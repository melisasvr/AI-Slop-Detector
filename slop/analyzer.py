from __future__ import annotations

import json

from pydantic import ValidationError

from .config import Settings
from .errors import ApiError, InvalidModelOutput, RateLimited
from .frames import Frame
from .groq_client import call_with_retry
from .models import Analysis, verdict_for
from .prompts import SYSTEM_PROMPT
from .transcribe import Transcript
from .utils import extract_json

MAX_TRANSCRIPT_CHARS = 12000


def _fmt_transcript(t: Transcript) -> str:
    if not t.has_audio:
        return "[no audio track]"
    if not t.text:
        return "[audio present but no speech detected - possibly music only]"
    text = t.text
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:7000] + "\n[...truncated...]\n" + text[-4000:]
    return f"(language: {t.language})\n{text}"


def _fmt_frames(frames: list[Frame]) -> str:
    lines = []
    for f in frames:
        if not f.caption:
            continue
        c = f.caption
        arts = "; ".join(c.get("ai_artifacts") or []) or "none noted"
        lines.append(f"Frame {f.index} @ {f.timestamp:.0f}s [{c.get('style', '?')}]: {c.get('description', '')} "
                     f"| artifacts: {arts} | on-screen text: {c.get('on_screen_text') or 'none'}")
    return "\n".join(lines) or "unavailable (vision step failed or was disabled)"


def build_user_message(meta: dict, transcript: Transcript, frames: list[Frame], signals: dict) -> str:
    return (
        "<METADATA>\n" + json.dumps(meta, ensure_ascii=False, indent=1) + "\n</METADATA>\n\n"
        "<TRANSCRIPT>\n" + _fmt_transcript(transcript) + "\n</TRANSCRIPT>\n\n"
        "<FRAME_NOTES>\n" + _fmt_frames(frames) + "\n</FRAME_NOTES>\n\n"
        "<SIGNALS>\n" + json.dumps(signals, indent=1) + "\n</SIGNALS>\n\n"
        "Return the JSON object now."
    )


def _parse(raw: str) -> Analysis:
    d = extract_json(raw)
    for k in ("slop_score", "confidence"):  # clamp instead of failing on 101 / -3 / "72"
        d[k] = max(0, min(100, int(float(d.get(k, 0)))))
    d["verdict"] = verdict_for(d["slop_score"])  # enforce consistency
    return Analysis.model_validate(d)


def _run(client, model: str, user_msg: str, on_wait) -> Analysis:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    last_err: Exception | None = None
    for _ in range(2):  # one repair attempt
        resp = call_with_retry(
            lambda: client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_completion_tokens=6000,
                reasoning_effort="medium", response_format={"type": "json_object"},
            ),
            on_wait=on_wait,
        )
        raw = resp.choices[0].message.content or ""
        try:
            return _parse(raw)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
            last_err = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Invalid output ({str(e)[:300]}). Return ONLY the corrected JSON."}]
    raise InvalidModelOutput(f"Model returned invalid JSON twice: {last_err}")


def analyze(client, cfg: Settings, meta, transcript, frames, signals, on_wait=None) -> tuple[Analysis, str]:
    """Returns (analysis, model_used). Falls back to the smaller model if the primary is limited/unavailable."""
    msg = build_user_message(meta, transcript, frames, signals)
    try:
        return _run(client, cfg.reasoning_model, msg, on_wait), cfg.reasoning_model
    except (RateLimited, ApiError, InvalidModelOutput):
        if cfg.reasoning_fallback and cfg.reasoning_fallback != cfg.reasoning_model:
            return _run(client, cfg.reasoning_fallback, msg, on_wait), cfg.reasoning_fallback
        raise
