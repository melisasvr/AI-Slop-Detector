import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from slop.config import Settings
from slop.errors import SlopError
from slop.pipeline import run_pipeline

st.set_page_config(page_title="AI Slop Detector", page_icon="🧪", layout="wide")

COLORS = {"Likely Real": "green", "Suspicious": "orange", "Likely AI Slop": "red"}
ICONS = {"slop": "🔴", "authentic": "🟢", "neutral": "⚪"}

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    base = Settings()
    reasoning = st.selectbox("Reasoning model", [base.reasoning_model, "openai/gpt-oss-20b"])
    use_vision = st.toggle(f"Frame analysis ({base.vision_model})", value=True,
                           help="Preview model. Disable to run on transcript + metadata only.")
    n_frames = st.slider("Key frames", 8, 12, 10)
    st.caption("Whisper: " + base.whisper_model)
cfg = replace(base, reasoning_model=reasoning, use_vision=use_vision, num_frames=n_frames)

st.title("🧪 AI Slop Detector")
st.caption("Transcript + frames + metadata → a likelihood estimate, with evidence. Not proof.")

tab_url, tab_file = st.tabs(["Paste URL", "Upload video"])
with tab_url:
    url = st.text_input("YouTube / TikTok / Instagram / X URL", placeholder="https://…")
with tab_file:
    upload = st.file_uploader("Video file", type=["mp4", "mov", "webm", "mkv"])

if st.button("Analyze", type="primary", disabled=not (url or upload)):
    bar, status = st.progress(0.0), st.empty()

    def progress(p: float, msg: str):
        if p >= 0:
            bar.progress(min(p, 1.0))
        status.info(msg)

    try:
        if upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as f:
                f.write(upload.getbuffer())
            tmp_path = Path(f.name)
            try:
                res = run_pipeline(tmp_path, is_url=False, api_key=api_key, cfg=cfg, progress=progress)
            finally:
                tmp_path.unlink(missing_ok=True)
            res.metadata["title"] = upload.name
        else:
            res = run_pipeline(url.strip(), is_url=True, api_key=api_key, cfg=cfg, progress=progress)
        st.session_state["result"] = res
        status.empty(); bar.empty()
    except SlopError as e:
        bar.empty(); status.error(str(e))
    except Exception as e:  # noqa: BLE001
        bar.empty(); status.error(f"Unexpected error: {e}")

res = st.session_state.get("result")
if res:
    a = res.analysis
    st.divider()
    st.markdown(f"## :{COLORS[a.verdict]}[{a.verdict}]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Slop score", f"{a.slop_score}/100")
    c2.metric("Confidence", f"{a.confidence}%")
    c3.metric("Evidence items", len(a.evidence))
    st.progress(a.slop_score / 100)
    st.write(a.summary)

    if not res.vision_ok:
        st.warning("Frame analysis was unavailable, so this verdict rests on transcript and metadata only."
                   + (f"\n\n**Reason:** `{res.vision_error}`" if res.vision_error else ""))
    st.subheader("Evidence")
    for ev in sorted(a.evidence, key=lambda e: {"strong": 0, "moderate": 1, "weak": 2}[e.strength]):
        st.markdown(f"{ICONS[ev.points_to]} **{ev.signal}** · _{ev.source}, {ev.strength}_  \n{ev.detail}")

    if a.limitations:
        with st.expander("Limitations"):
            for l in a.limitations:
                st.write("• " + l)

    st.subheader("Sampled frames")
    cols = st.columns(5)
    for i, f in enumerate(res.frames):
        with cols[i % 5]:
            st.image(f.jpeg, caption=f"#{f.index} · {f.timestamp:.0f}s")
            if f.caption:
                with st.popover("details"):
                    st.write(f.caption.get("description", ""))
                    if f.caption.get("ai_artifacts"):
                        st.write("**Artifacts:** " + "; ".join(f.caption["ai_artifacts"]))

    with st.expander("Transcript"):
        st.write(res.transcript.text or "_No speech detected._")
    with st.expander("Metadata & signals"):
        st.json({"metadata": res.metadata, "signals": res.signals})
    st.caption(f"Reasoning: {res.model_used} · timings (s): {res.timings}")
    st.download_button("Download JSON report", json.dumps(res.to_dict(), indent=2, default=str),
                       "slop_report.json", "application/json")
