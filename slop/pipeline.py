from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .analyzer import analyze
from .config import Settings
from .downloader import download_video
from .frames import Frame, extract_frames, shot_stats
from .groq_client import get_client
from .models import Analysis
from .transcribe import Transcript, transcribe
from .utils import transcript_stats
from .vision import caption_frames

Progress = Callable[[float, str], None]


@dataclass
class Result:
    analysis: Analysis
    metadata: dict
    frames: list[Frame]
    transcript: Transcript
    signals: dict
    model_used: str
    vision_ok: bool
    timings: dict = field(default_factory=dict)
    vision_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "analysis": self.analysis.model_dump(),
            "metadata": self.metadata,
            "signals": self.signals,
            "transcript": {"language": self.transcript.language, "text": self.transcript.text},
            "frames": [{"index": f.index, "timestamp": f.timestamp, "caption": f.caption} for f in self.frames],
            "model_used": self.model_used,
            "vision_ok": self.vision_ok,
            "vision_error": self.vision_error,
            "timings_s": self.timings,
        }


def run_pipeline(source: str | Path, *, is_url: bool, api_key: str, cfg: Settings,
                 progress: Progress = lambda p, m: None) -> Result:
    client = get_client(api_key)
    t: dict[str, float] = {}
    clock = time.perf_counter
    wait = lambda m: progress(-1, m)  # noqa: E731  (-1 = "status text only")

    with tempfile.TemporaryDirectory(prefix="slop_") as tmp:
        work = Path(tmp)

        s = clock()
        if is_url:
            progress(0.05, "Downloading video…")
            video, meta = download_video(str(source), work, cfg)
        else:
            video, meta = Path(source), {"title": Path(source).name, "note": "user-uploaded file; no platform metadata"}
        t["download"] = clock() - s

        # Audio transcription runs in a thread while we do frames + vision on the main thread.
        progress(0.25, "Transcribing audio and extracting frames…")
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(transcribe, client, video, work, cfg.whisper_model, wait)

            s = clock()
            frames = extract_frames(video, cfg.num_frames, cfg.frame_width)
            vsignals = shot_stats(video, max_seconds=120)
            t["frames"] = clock() - s

            vision_ok, vision_error = False, None
            if cfg.use_vision:
                progress(0.45, f"Describing frames with {cfg.vision_model}…")
                s = clock()
                vision_ok, vision_error = caption_frames(client, frames, cfg, wait)
                t["vision"] = clock() - s

            progress(0.7, "Waiting for transcript…")
            s = clock()
            transcript = fut.result()
            t["transcribe"] = clock() - s

        signals = {**vsignals, **transcript_stats(transcript.text, meta.get("duration") or vsignals.get("duration_s"))}

        progress(0.8, f"Analyzing with {cfg.reasoning_model}…")
        s = clock()
        analysis, used = analyze(client, cfg, meta, transcript, frames, signals, wait)
        t["analysis"] = clock() - s
        progress(1.0, "Done")

    return Result(analysis, meta, frames, transcript, signals, used, vision_ok,
                  {k: round(v, 1) for k, v in t.items()}, vision_error)