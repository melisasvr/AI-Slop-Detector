# AI Slop Detector
A Streamlit application that estimates the likelihood that a short-form video is **AI slop**: low-effort,
mass-produced content built primarily with generative AI. It combines speech transcription, frame-level visual
analysis, platform metadata and editing-rhythm signals, and returns a structured, evidence-backed assessment.
All model inference runs through the **Groq API**.

## Features

- **Flexible input:** YouTube, TikTok, Instagram and X URLs, or a direct video upload
- **Multimodal analysis:** transcript, sampled key frames, platform metadata and computed signals
- **Structured output:** slop score (0–100), verdict, confidence, and categorized evidence with strength ratings
- **Graceful degradation:** continues with transcript and metadata if frame analysis is unavailable
- **Exportable results:** downloadable JSON report for every analysis

## Architecture

```
URL / upload ─► yt-dlp ─► video ─┬─► ffmpeg audio ─► Groq Whisper ─────────┐
                                 ├─► OpenCV: 8–12 frames ─► Qwen3.8 vision ─┼─► gpt-oss-120b ─► JSON
                                 ├─► shot-rhythm + transcript stats ────────┤   (validated; verdict
                                 └─► platform metadata ─────────────────────┘    derived from score)
```

| Stage | Model | Notes |
|---|---|---|
| Transcription | `whisper-large-v3-turbo` | Production model; runs in parallel with frame processing |
| Frame captioning | `qwen/qwen3.8-27b` | Preview model with image input; limited to 3 images per request, so frames are sent in batches of 3. Optional |
| Final judgment | `openai/gpt-oss-120b` | Text-only reasoning; falls back to `gpt-oss-20b` on rate limits or errors |

## Requirements

- Python 3.10 or later
- A [Groq API key](https://console.groq.com)
- ffmpeg (the system binary is used if present, otherwise the bundled `imageio-ffmpeg` binary)
- Deno or Node.js, required by yt-dlp for YouTube downloads

## Quick start

```bash
git clone <your-repo-url>
cd slop-detector

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# Edit .env and set GROQ_API_KEY

streamlit run app.py
```

## Configuration

Settings are read from `.env`.

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key |
| `SLOP_REASONING_MODEL` | No | Override the reasoning model (default `openai/gpt-oss-120b`) |
| `SLOP_VISION_MODEL` | No | Override the vision model (default `qwen/qwen3.8-27b`) |
| `SLOP_WHISPER_MODEL` | No | Override the transcription model (default `whisper-large-v3-turbo`) |
| `YTDLP_COOKIES_BROWSER` | No | Browser to read cookies from, e.g. `chrome` or `firefox` |
| `YTDLP_COOKIES_FILE` | No | Path to a cookies file, as an alternative to the browser option |

## Troubleshooting

- **YouTube returns HTTP 403:** install Deno or Node.js and restart the app, then update yt-dlp with
  `pip install -U --pre "yt-dlp[default]"`.
- **Instagram, TikTok or X downloads fail:** these platforms often require authentication. Set
  `YTDLP_COOKIES_BROWSER`, or download the clip manually and use the Upload tab.
- **Frame analysis unavailable:** the reason is shown in the application. The assessment continues using transcript
  and metadata only.

## Testing
- `pytest`
- The test suite covers the scoring and parsing logic and requires no network access.

## Design decisions

- **Verdict is derived from the score in code** (`<35` Likely Real, `<65` Suspicious, else Likely AI Slop), so the model can't contradict itself.
- **Strict JSON + validation**: `json_object` mode, Pydantic schema, score clamping, one automatic repair retry.
- **Prompt-injection hardening**: transcript/metadata/captions are delimited as untrusted data.
- **Calibration rules in the prompt**: missing data isn't evidence; confidence is capped when a modality is missing; a single-frame vision claim can only be weak evidence; "Likely AI Slop" requires support from at least two sources.
- **Resilience**: `Retry-After`-aware backoff on 429s, retries on 5xx/network, multi-attempt YouTube downloads, friendly errors for failed downloads, long videos, no-audio clips, bad API keys, and vision failures.
- **Cost control**: 480p downloads, 16 kHz mono 32 kbps audio, ≤12 frames, 15-min cap.

## Limitations

- It's a **heuristic likelihood**, not ground truth; there is no labeled-benchmark evaluation yet. It has only been spot-checked manually on a handful of videos.
- Audio is judged via its **transcript** only: synthetic voice cues (prosody) are not directly observed.
- Vision models misread blurred hands and small on-screen text, so frame-artifact claims are treated as weak evidence unless repeated across frames (see rule 8 in `slop/prompts.py`).
- Instagram/TikTok/X downloads break often; use cookies or the Upload tab.
- `qwen/qwen3.8-27b` is a Preview model and may change; override with `SLOP_VISION_MODEL`.

## Ideas to extend

Evaluate on a labeled set (precision/recall, calibration curve) · scene-change-based frame selection ·
batch mode + CSV export · FastAPI backend · caching by video ID.

## Contributing

Contributions are welcome.

1. Fork the repo and create a branch: `git checkout -b my-change`
2. Make your change and run `pytest`.
3. Keep pull requests small and focused, and explain what changed and why.
4. Never commit secrets: `.env` is git-ignored, so keep your Groq API key there.

Good first areas: a labeled evaluation set, better frame selection, support for more video platforms, and prompt calibration.
Bug reports and ideas are welcome as GitHub issues. Please include the video type (not necessarily the link), the error message, and your yt-dlp version.

## License
- MIT License
```
Copyright (c) 2026 YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```