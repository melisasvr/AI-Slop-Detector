from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .errors import FrameExtractionError


@dataclass
class Frame:
    index: int
    timestamp: float
    jpeg: bytes
    caption: dict | None = None


def _encode(img, width: int) -> bytes:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise FrameExtractionError("Failed to encode frame.")
    return buf.tobytes()


def extract_frames(path: Path, n: int, width: int) -> list[Frame]:
    """Evenly spaced frames (segment midpoints), skipping near-black ones."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FrameExtractionError("OpenCV could not open the video file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        raise FrameExtractionError("Video has no readable frames.")
    out: list[Frame] = []
    for i in range(n):
        base = int((i + 0.5) / n * total)
        for off in (0, int(fps), -int(fps), int(2 * fps)):
            pos = min(max(base + off, 0), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, img = cap.read()
            if ok and img is not None and img.mean() > 12:
                out.append(Frame(len(out), pos / fps, _encode(img, width)))
                break
    cap.release()
    if len(out) < 3:
        raise FrameExtractionError("Could not extract enough usable frames.")
    return out


def shot_stats(path: Path, max_seconds: int = 300) -> dict:
    """Cheap editing-rhythm signal: histogram-correlation cuts sampled at ~2 fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step, limit = max(int(round(fps / 2)), 1), int(max_seconds * fps)
    prev, cuts, idx = None, 0, 0
    while idx < limit and cap.grab():
        if idx % step == 0:
            ok, img = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < 0.5:
                    cuts += 1
                prev = hist
        idx += 1
    cap.release()
    analyzed = max(idx / fps, 1e-6)
    return {
        "resolution": f"{w}x{h}",
        "fps": round(fps, 1),
        "duration_s": round(total / fps, 1) if total else None,
        "estimated_cuts_per_minute": round(cuts / analyzed * 60, 1),
        "avg_shot_length_s": round(analyzed / (cuts + 1), 1),
    }
