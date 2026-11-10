from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .errors import TranscriptionError
from .groq_client import call_with_retry
from .utils import ffmpeg_path


@dataclass
class Transcript:
    text: str = ""
    language: str | None = None
    segments: list[dict] = field(default_factory=list)
    has_audio: bool = True


def _extract_audio(video: Path, workdir: Path) -> Path | None:
    out = workdir / "audio.mp3"
    cmd = [ffmpeg_path(), "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        raise TranscriptionError("Audio extraction timed out.") from e
    return out if r.returncode == 0 and out.exists() and out.stat().st_size > 1024 else None


def transcribe(client, video: Path, workdir: Path, model: str, on_wait=None) -> Transcript:
    audio = _extract_audio(video, workdir)
    if audio is None:
        return Transcript(has_audio=False)
    data = audio.read_bytes()  # 32 kbps mono => ~2.4 MB per 10 min, well under Groq's file limits
    try:
        resp = call_with_retry(
            lambda: client.audio.transcriptions.create(
                file=(audio.name, data), model=model, response_format="verbose_json", temperature=0.0
            ),
            on_wait=on_wait,
        )
    except TranscriptionError:
        raise
    d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    segs = [
        {"start": round(s.get("start", 0), 1), "end": round(s.get("end", 0), 1), "text": s.get("text", "").strip()}
        for s in d.get("segments", []) or []
        if s.get("no_speech_prob", 0) < 0.6  # drop Whisper's music/silence hallucinations
    ]
    text = " ".join(s["text"] for s in segs) if segs else (d.get("text") or "").strip()
    return Transcript(text=text, language=d.get("language"), segments=segs)
