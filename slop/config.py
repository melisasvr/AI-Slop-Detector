"""Central settings. Models can be overridden via env vars (Groq's lineup changes often)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    reasoning_model: str = os.getenv("SLOP_REASONING_MODEL", "openai/gpt-oss-120b")
    reasoning_fallback: str = os.getenv("SLOP_REASONING_FALLBACK", "openai/gpt-oss-20b")
    vision_model: str = os.getenv("SLOP_VISION_MODEL", "qwen/qwen3.8-27b")  # Preview, image input
    whisper_model: str = os.getenv("SLOP_WHISPER_MODEL", "whisper-large-v3-turbo")

    num_frames: int = 10
    frames_per_vision_call: int = 3
    frame_width: int = 768
    use_vision: bool = True

    max_duration_s: int = 900       # refuse videos longer than 15 min
    max_download_mb: int = 150
    cookies_file: str | None = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser: str | None = os.getenv("YTDLP_COOKIES_BROWSER")  # e.g. "chrome", "firefox", "edge"
