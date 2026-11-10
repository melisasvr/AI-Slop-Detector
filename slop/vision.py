"""Frame captioning with Groq's multimodal Qwen. Optional: failure degrades gracefully."""
from __future__ import annotations

import base64
import json

import groq

from .config import Settings
from .errors import SlopError
from .frames import Frame
from .groq_client import call_with_retry
from .prompts import VISION_PROMPT
from .utils import extract_json
from .errors import ApiError, RateLimited, SlopError

def _data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _caption_batch(client, model: str, batch: list[Frame], on_wait) -> list[dict]:
    content: list[dict] = []
    for f in batch:
        content.append({"type": "text", "text": f"Frame {f.index} @ {f.timestamp:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": _data_uri(f.jpeg)}})
    content.append({"type": "text", "text": VISION_PROMPT.format(n=len(batch))})
    kwargs = dict(model=model, messages=[{"role": "user", "content": content}], temperature=0.1,
                  max_completion_tokens=2500)

    def call():
        try:
            return client.chat.completions.create(reasoning_format="hidden", **kwargs)
        except groq.BadRequestError:  # older SDK/model without reasoning_format
            return client.chat.completions.create(**kwargs)

    resp = call_with_retry(call, on_wait=on_wait)
    return extract_json(resp.choices[0].message.content).get("frames", [])


def caption_frames(client, frames: list[Frame], cfg: Settings, on_wait=None) -> tuple[bool, str | None]:
    """Fills frame.caption in place. Returns (any_batch_succeeded, last_error_message)."""
    ok_any, last_error = False, None
    by_index = {f.index: f for f in frames}
    for i in range(0, len(frames), cfg.frames_per_vision_call):
        batch = frames[i : i + cfg.frames_per_vision_call]
        try:
            for item in _caption_batch(client, cfg.vision_model, batch, on_wait):
                f = by_index.get(item.get("index"))
                if f:
                    f.caption = item
                    ok_any = True
        except (SlopError, json.JSONDecodeError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:300]}"
            if isinstance(e, (ApiError, RateLimited)):
                break  # model/API problem: don't waste time retrying the other batches
    return ok_any, last_error