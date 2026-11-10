from __future__ import annotations

import json
import re
import shutil
from collections import Counter

from .errors import SlopError


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise SlopError("ffmpeg not found. Install it (e.g. `brew install ffmpeg`, `apt install ffmpeg`).") from e


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles <think> tags and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return json.loads(text[start : end + 1])


def transcript_stats(text: str, duration_s: float | None) -> dict:
    words = re.findall(r"[\w']+", text.lower())
    if not words:
        return {"word_count": 0}
    grams = Counter(zip(words, words[1:], words[2:], words[3:]))
    top = grams.most_common(1)[0][1] if grams else 0
    out = {
        "word_count": len(words),
        "lexical_diversity": round(len(set(words)) / len(words), 3),
        "max_repeated_4gram_count": top,
    }
    if duration_s:
        out["words_per_minute"] = round(len(words) / (duration_s / 60))
    return out
