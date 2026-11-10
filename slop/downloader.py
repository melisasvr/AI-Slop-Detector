from __future__ import annotations

import re
import shutil
from pathlib import Path

import yt_dlp
from yt_dlp.version import __version__ as YTDLP_VERSION

from .config import Settings
from .errors import DownloadFailed
from .utils import ffmpeg_path

_ANSI = re.compile(r"\x1b?\[[0-9;]*m")  # yt-dlp colour codes (ESC byte is sometimes already stripped)

_FMT_SPLIT = "bv*[height<=480]+ba/b[height<=480]/b"   # separate video+audio (DASH)
_FMT_SINGLE = "b[height<=480]/b/bv*+ba"               # single combined file, avoids DASH 403s

# Tried in order until one works. Different YouTube clients get different PO-token / 403 treatment.
_ATTEMPTS = [
    {"format": _FMT_SPLIT, "yt_args": None},
    {"format": _FMT_SINGLE, "yt_args": {"player_client": ["default", "-android_vr"]}},
    {"format": _FMT_SINGLE, "yt_args": {"player_client": ["web_safari", "mweb", "tv"]}},
]

# Errors where retrying with another client is pointless.
_PERMANENT = ("private video", "video unavailable", "has been removed", "members-only",
              "not available in your country", "confirm your age", "sign in to confirm")


def _js_runtimes() -> dict:
    """YouTube needs an external JS runtime. Pass every one we can find on PATH."""
    found = {}
    for name, exe in (("deno", "deno"), ("node", "node"), ("bun", "bun"), ("quickjs", "qjs")):
        path = shutil.which(exe)
        if path:
            found[name] = {"path": path}
    return found


def _meta(info: dict) -> dict:
    keys = ["title", "uploader", "channel", "upload_date", "duration", "view_count", "like_count",
            "comment_count", "channel_follower_count", "categories", "extractor_key"]
    meta = {k: info.get(k) for k in keys}
    meta["description"] = (info.get("description") or "")[:1500]
    meta["tags"] = (info.get("tags") or [])[:15]
    return {k: v for k, v in meta.items() if v not in (None, "", [])}


def _build_opts(out_dir: Path, cfg: Settings, attempt: dict, rejected: list) -> dict:
    def too_long(info, *, incomplete=False):
        d = info.get("duration")
        if d and d > cfg.max_duration_s:
            rejected.append(d)
            return "too long"
        return None

    opts = {
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "format": attempt["format"],
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
        "max_filesize": cfg.max_download_mb * 1024 * 1024,
        "ffmpeg_location": ffmpeg_path(),
        "match_filter": too_long,
    }
    runtimes = _js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes
    if attempt["yt_args"]:
        opts["extractor_args"] = {"youtube": attempt["yt_args"]}
    if cfg.cookies_file:
        opts["cookiefile"] = cfg.cookies_file
    elif cfg.cookies_browser:
        opts["cookiesfrombrowser"] = (cfg.cookies_browser,)
    return opts


def _media_files(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("video.*") if p.suffix not in (".part", ".ytdl", ".temp"))


def _clear(out_dir: Path) -> None:
    for p in out_dir.glob("video.*"):
        p.unlink(missing_ok=True)


def _explain(url: str, err: str) -> str:
    if any(h in url for h in ("youtube.com", "youtu.be")):
        hint = ("YouTube usually needs: (1) an up-to-date yt-dlp: pip install -U --pre \"yt-dlp[default]\"; "
                "(2) Deno or Node.js installed (then restart the app); (3) optionally YTDLP_COOKIES_BROWSER=chrome "
                "in .env. Or download the clip yourself and use the Upload tab.")
    else:
        hint = ("Instagram, TikTok and X often require login or block scrapers. Set YTDLP_COOKIES_BROWSER=chrome "
                "(or YTDLP_COOKIES_FILE), or download the clip and use the Upload tab.")
    runtimes = ", ".join(_js_runtimes()) or "none found"
    return f"Download failed: {err}. {hint} [yt-dlp {YTDLP_VERSION}; JS runtimes: {runtimes}]"


def download_video(url: str, out_dir: Path, cfg: Settings) -> tuple[Path, dict]:
    last_err = "unknown error"
    for attempt in _ATTEMPTS:
        _clear(out_dir)
        rejected: list = []
        try:
            with yt_dlp.YoutubeDL(_build_opts(out_dir, cfg, attempt, rejected)) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            last_err = _ANSI.sub("", str(e)).replace("ERROR: ", "").strip().splitlines()[0][:220]
            if any(p in last_err.lower() for p in _PERMANENT):
                break
            continue
        except Exception as e:  # noqa: BLE001
            raise DownloadFailed(f"Unexpected download error: {e}") from e

        if rejected:
            raise DownloadFailed(f"Video is {rejected[0] / 60:.1f} min; limit is {cfg.max_duration_s / 60:.0f} min.")
        if info and info.get("_type") == "playlist":
            info = (info.get("entries") or [None])[0]
        files = _media_files(out_dir)
        if files and info:
            return files[0], _meta(info)
        last_err = f"no file was produced (it may exceed {cfg.max_download_mb} MB)"

    raise DownloadFailed(_explain(url, last_err))