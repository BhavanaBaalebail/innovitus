# dashboard.py
# ---------------------------------------------------------
# NeuroPulse AI - Cyberpunk HUD Edition
# High-Fidelity Sci-Fi / Neural-Link Aesthetics
# Theme: Void Black + Neon Cyan + Hazard Orange + Matrix Green
#
# Run:
#   streamlit run dashboard.py
# ---------------------------------------------------------

import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
from groq import Groq

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from predict import predict_fatigue
try:
    from rag.llm_coach import generate_coaching_recommendation
    from rag.retriever import SessionRetriever

    RAG_AVAILABLE = True
    RAG_IMPORT_ERROR = ""
except Exception as rag_exc:  # pragma: no cover - keeps dashboard running without RAG deps
    generate_coaching_recommendation = None
    SessionRetriever = None
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = str(rag_exc)

# Project root (parent of src/) — app.py lives here
_VISION_APP = _PROJECT_ROOT / "app.py"
_VISION_OUTPUT_JSON = _PROJECT_ROOT / "vision_output.json"
_SESSION_HISTORY_PATH = _PROJECT_ROOT / "data" / "session_history.json"


@st.cache_resource
def get_session_retriever() -> SessionRetriever | None:
    if not RAG_AVAILABLE or SessionRetriever is None:
        return None
    return SessionRetriever(db_path=_SESSION_HISTORY_PATH)


def to_rag_fatigue_level(predicted_label: str) -> str:
    if predicted_label == "High Fatigue":
        return "high"
    if predicted_label == "Mild Fatigue":
        return "moderate"
    return "mild"


def score_badge_style(score_pct: int) -> str:
    if score_pct > 80:
        return "background:rgba(0,255,65,0.18);color:#00ff41;border:1px solid rgba(0,255,65,0.45);"
    if score_pct >= 60:
        return "background:rgba(255,204,0,0.18);color:#ffd24d;border:1px solid rgba(255,204,0,0.45);"
    return "background:rgba(255,140,0,0.18);color:#ff8c00;border:1px solid rgba(255,140,0,0.45);"


def top_intervention(summary: dict[str, float]) -> tuple[str, float]:
    if not summary:
        return "No history yet", 0.0
    name = max(summary, key=summary.get)
    return name, summary[name]

# Evidence-aligned wellness copy (general health, sleep, ergonomics, attention science).
# Not medical advice; persistent symptoms warrant a clinician.
_W_ITEMS_MAINTAIN = [
    (
        "Timed work / rest cycles",
        "Short breaks every 25–30 minutes during demanding work match common Pomodoro-style practice and help sustain attention across the day.",
    ),
    (
        "Movement micro-breaks",
        "WHO guidance highlights that breaking up long sitting with light activity supports cardiovascular and metabolic health—brief standing or walking counts.",
    ),
    (
        "Vision recovery pauses",
        "The 20-20-20 rule (every ~20 minutes, look ~20 feet away for ~20 seconds) is a practical cue to reduce sustained near-focus strain; it aligns with eye-care recommendations to vary viewing distance.",
    ),
    (
        "Sleep regularity",
        "Consistent sleep and wake times are a core CDC / National Sleep Foundation recommendation for next-day alertness and stress resilience.",
    ),
    (
        "Daytime caffeine cut-off",
        "Caffeine late in the day can delay sleep onset; limiting it after early afternoon supports recovery sleep that underpins alert performance.",
    ),
]

_W_ITEMS_MULTITASK = [
    (
        "Reduce task switching",
        "Controlled experiments on “switch costs” show interleaving unrelated tasks can slow reaction time and increase errors—time-boxing one primary task reduces cognitive load.",
    ),
    (
        "Attention restoration",
        "Kaplan’s attention-restoration theory and follow-on studies associate brief exposure to natural scenes with improved directed-attention recovery versus uninterrupted urban stimuli.",
    ),
    (
        "Slow breathing (e.g., 4-6 breaths/min)",
        "Meta-analyses of slow-paced breathing report modest reductions in self-reported stress and blood pressure in healthy adults; it is a low-risk adjunct, not a substitute for care when needed.",
    ),
    (
        "Notification hygiene",
        "HCI research links frequent interruptions with longer task resumption time—batching notifications supports deeper focus blocks.",
    ),
    (
        "Hydration",
        "Controlled dehydration studies (≈1–2% body-mass loss) show measurable hits on concentration and mood in some cohorts—regular water intake during long sessions is sensible.",
    ),
]

_W_ITEMS_MILD = [
    (
        "Strategic short nap",
        "Sleep-medicine reviews note that 10–20 minute naps can improve alertness and psychomotor speed without heavy sleep inertia when sleep debt is mild.",
    ),
    (
        "Morning bright light",
        "Circadian research shows timed bright-light exposure after waking advances alertness rhythms—outdoor daylight is especially effective.",
    ),
    (
        "Gentle mobility breaks",
        "Occupational health guidelines encourage stretching or posture changes every 45–60 minutes to reduce musculoskeletal discomfort during desk work.",
    ),
    (
        "Front-load demanding work",
        "Ultradian rhythm literature suggests planning cognitively heavy tasks earlier in a wake episode when homeostatic sleep pressure is lower.",
    ),
    (
        "Wind-down for sleep",
        "Evidence-based sleep hygiene includes dimming screens and reducing stimulating tasks 1–2 hours before bed to protect sleep quality.",
    ),
]

_W_ITEMS_HIGH = [
    (
        "Prioritize sleep opportunity tonight",
        "Chronic short sleep is strongly linked to attention lapses and safety risk; CDC recommends most adults aim for ≥7 hours per night when feasible.",
    ),
    (
        "Avoid safety-critical tasks",
        "NHTSA and sleep societies warn that severe drowsiness markedly raises accident risk—defer driving or machinery until rested.",
    ),
    (
        "Immediate rest or nap window",
        "If circumstances allow, a protected 20–30 minute nap can partially restore alertness; longer naps may cause grogginess (sleep-inertia literature).",
    ),
    (
        "Hydrate and fuel lightly",
        "Large heavy meals increase post-meal sleepiness; lighter snacks plus fluids can avoid compounding fatigue.",
    ),
    (
        "Seek care if fatigue is persistent",
        "Unexplained ongoing exhaustion warrants medical evaluation (thyroid, mood, sleep disorders, etc.)—this dashboard is not diagnostic.",
    ),
]

PROFILE_WELLNESS = {
    "Focused Coder": {
        "title": "Maintain peak alertness",
        "subtitle": "Profile: steady, high-tempo work — keep recovery habits that preserve performance.",
        "items": _W_ITEMS_MAINTAIN,
    },
    "Distracted Multitasker": {
        "title": "Lower stress & rebuild focus",
        "subtitle": "Profile: fragmented attention — reduce load and restore directed attention.",
        "items": _W_ITEMS_MULTITASK,
    },
    "Mild Fatigue": {
        "title": "Recover before fatigue deepens",
        "subtitle": "Profile: early warning signs — evidence-based resets.",
        "items": _W_ITEMS_MILD,
    },
    "High Fatigue": {
        "title": "Urgent recovery & safety",
        "subtitle": "Profile: high strain — prioritize rest and risk reduction.",
        "items": _W_ITEMS_HIGH,
    },
}

_MANUAL_BY_LABEL = {
    "Alert": PROFILE_WELLNESS["Focused Coder"],
    "Mild Fatigue": PROFILE_WELLNESS["Mild Fatigue"],
    "High Fatigue": PROFILE_WELLNESS["High Fatigue"],
}


def build_wellness_suggestions_html(profile: str, predicted_label: str) -> str:
    if profile == "Manual Input":
        block = _MANUAL_BY_LABEL.get(predicted_label, _MANUAL_BY_LABEL["Alert"])
    else:
        block = PROFILE_WELLNESS[profile]
    title = html.escape(block["title"])
    subtitle = html.escape(block["subtitle"])
    lis = "".join(
        f"<li><strong>{html.escape(h)}</strong> — {html.escape(t)}</li>"
        for h, t in block["items"]
    )
    foot = (
        "General wellness information from public health and cognitive-ergonomics literature; "
        "not individualized medical advice."
    )
    return f"""
<div class="hud-module" style="margin-top:4px;">
  <div class="mod-label">Evidence-based suggestions</div>
  <div style="font-family:Orbitron,monospace;font-size:14px;font-weight:700;letter-spacing:2px;color:#00f2ff;margin-bottom:4px;">{title}</div>
  <div style="font-size:9px;letter-spacing:2px;color:rgba(0,242,255,0.45);margin-bottom:10px;">{subtitle}</div>
  <ul class="wellness-list">{lis}</ul>
  <div class="wellness-foot">{html.escape(foot)}</div>
</div>
"""


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="NeuroPulse AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# CYBERPUNK CSS INJECTION
# ---------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── ROOT PALETTE ─────────────────────────────────────── */
:root {
    --void:       #010a0f;
    --cyan:       #00f2ff;
    --cyan-dim:   rgba(0,242,255,0.12);
    --cyan-glow:  rgba(0,242,255,0.40);
    --orange:     #ff8c00;
    --orange-dim: rgba(255,140,0,0.12);
    --green:      #00ff41;
    --green-dim:  rgba(0,255,65,0.10);
    --red-alert:  #ff003c;
    --glass:      rgba(0,20,30,0.72);
    --border:     rgba(0,242,255,0.22);
}

/* ── GLOBAL RESET ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #010a0f !important;
    color: rgba(0,242,255,0.85) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── SCANLINES OVERLAY ────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.07) 2px,
        rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 9999;
    animation: scanmove 10s linear infinite;
}

/* Moving horizontal scan beam */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,242,255,0.35), transparent);
    z-index: 9998;
    animation: scanbeam 7s linear infinite;
    pointer-events: none;
}

/* ── GRID BACKGROUND ──────────────────────────────────── */
[data-testid="stMain"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,242,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,242,255,0.035) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none;
    z-index: 0;
}

/* ── ANIMATIONS ───────────────────────────────────────── */
@keyframes scanmove  { to { background-position: 0 100px; } }
@keyframes scanbeam  { 0%{ top:-2px; } 100%{ top:100vh; } }
@keyframes pulse-cyan   { 0%,100%{ box-shadow: 0 0 8px rgba(0,242,255,0.4), 0 0 20px rgba(0,242,255,0.12); } 50%{ box-shadow: 0 0 18px rgba(0,242,255,0.6), 0 0 45px rgba(0,242,255,0.20); } }
@keyframes pulse-orange { 0%,100%{ box-shadow: 0 0 8px rgba(255,140,0,0.4),  0 0 20px rgba(255,140,0,0.12); } 50%{ box-shadow: 0 0 18px rgba(255,140,0,0.6),  0 0 45px rgba(255,140,0,0.20); } }
@keyframes pulse-green  { 0%,100%{ box-shadow: 0 0 6px rgba(0,255,65,0.35),  0 0 16px rgba(0,255,65,0.10); } 50%{ box-shadow: 0 0 14px rgba(0,255,65,0.55), 0 0 35px rgba(0,255,65,0.18); } }
@keyframes blink     { 0%,100%{ opacity:1; } 50%{ opacity:0; } }

/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(0,8,14,0.95) !important;
    border-right: 1px solid rgba(0,242,255,0.18) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace !important;
    color: rgba(0,242,255,0.75) !important;
}
[data-testid="stSidebar"] .stRadio label { font-size: 11px !important; letter-spacing: 2px !important; }
[data-testid="stSidebar"] .stSlider label { font-size: 9px !important; letter-spacing: 2px !important; text-transform: uppercase !important; }

/* ── HUD MODULE (glassmorphism + corner brackets) ──────── */
.hud-module {
    background: rgba(0,20,30,0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0,242,255,0.22);
    border-radius: 4px;
    padding: 18px 20px;
    position: relative;
    margin-bottom: 12px;
}
/* Top-left bracket */
.hud-module::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 14px; height: 14px;
    border-top: 2px solid #00f2ff;
    border-left: 2px solid #00f2ff;
}
/* Bottom-right bracket */
.hud-module::after {
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 14px; height: 14px;
    border-bottom: 2px solid #00f2ff;
    border-right: 2px solid #00f2ff;
}

/* ── STATE-SPECIFIC GLOWS ─────────────────────────────── */
.hud-state-alert  { border-color: rgba(0,255,65,0.4)  !important; animation: pulse-green  2.5s ease-in-out infinite; }
.hud-state-mild   { border-color: rgba(255,140,0,0.38) !important; animation: pulse-orange 2.5s ease-in-out infinite; }
.hud-state-high   { border-color: rgba(255,0,60,0.50)  !important; animation: pulse-orange 1.6s ease-in-out infinite; }

/* ── ORBITRON HEADERS ─────────────────────────────────── */
.orb-header {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 3px;
    color: #00f2ff;
    text-shadow: 0 0 14px rgba(0,242,255,0.5);
    margin: 0;
}
.orb-sub {
    font-size: 9px;
    letter-spacing: 4px;
    color: rgba(0,242,255,0.40);
    margin-top: 2px;
    text-transform: uppercase;
}
.mod-label {
    font-size: 8px;
    letter-spacing: 3px;
    color: rgba(0,242,255,0.40);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.mod-label::before { content: '── '; }

/* ── METRIC DISPLAY ───────────────────────────────────── */
.metric-orb {
    font-family: 'Orbitron', monospace;
    font-size: 30px;
    font-weight: 900;
    line-height: 1;
}
.metric-cyan   { color: #00f2ff; text-shadow: 0 0 14px rgba(0,242,255,0.55); }
.metric-green  { color: #00ff41; text-shadow: 0 0 14px rgba(0,255,65,0.55); }
.metric-orange { color: #ff8c00; text-shadow: 0 0 14px rgba(255,140,0,0.55); }
.metric-red    { color: #ff003c; text-shadow: 0 0 14px rgba(255,0,60,0.55); }

/* ── NEON PROGRESS BARS ───────────────────────────────── */
.neon-bar-track {
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 6px;
}
.neon-bar-fill-alert  { height:100%; border-radius:3px; background:#00ff41; box-shadow: 0 0 8px #00ff41; }
.neon-bar-fill-mild   { height:100%; border-radius:3px; background:#ff8c00; box-shadow: 0 0 8px #ff8c00; }
.neon-bar-fill-high   { height:100%; border-radius:3px; background:#ff003c; box-shadow: 0 0 8px #ff003c; }

/* ── BIOMARKER / FEATURE TABLES ───────────────────────── */
.cyber-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}
.cyber-table td {
    padding: 7px 6px;
    border-bottom: 1px solid rgba(0,242,255,0.07);
    color: rgba(0,242,255,0.55);
    letter-spacing: 1px;
}
.cyber-table td:first-child { color: #00f2ff; font-weight: 700; letter-spacing: 2px; }
.cyber-table td:last-child  { color: #00f2ff; text-align: right; font-weight: 600; }

/* ── STATUS BANNER ────────────────────────────────────── */
.banner-alert { background:rgba(0,255,65,0.07);  border:1px solid rgba(0,255,65,0.35);  color:#00ff41;  padding:12px 18px; border-radius:3px; font-size:11px; letter-spacing:2px; text-transform:uppercase; }
.banner-mild  { background:rgba(255,140,0,0.07); border:1px solid rgba(255,140,0,0.35); color:#ff8c00;  padding:12px 18px; border-radius:3px; font-size:11px; letter-spacing:2px; text-transform:uppercase; }
.banner-high  { background:rgba(255,0,60,0.08);  border:1px solid rgba(255,0,60,0.45);  color:#ff003c;  padding:12px 18px; border-radius:3px; font-size:11px; letter-spacing:2px; text-transform:uppercase; animation: pulse-orange 1.6s ease-in-out infinite; }

/* ── TERMINAL LOG ─────────────────────────────────────── */
.terminal {
    font-size: 10px;
    line-height: 1.9;
    color: #00ff41;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,8,4,0.6);
    border: 1px solid rgba(0,255,65,0.15);
    border-radius: 3px;
    padding: 12px 14px;
}
.t-dim    { color: rgba(0,242,255,0.35); }
.t-cyan   { color: #00f2ff; }
.t-orange { color: #ff8c00; }
.t-red    { color: #ff003c; }
.t-green  { color: #00ff41; }

/* ── STREAMLIT COMPONENT OVERRIDES ───────────────────── */
div[data-testid="metric-container"] {
    background: rgba(0,20,30,0.6) !important;
    border: 1px solid rgba(0,242,255,0.2) !important;
    border-radius: 4px !important;
    padding: 14px !important;
}
div[data-testid="metric-container"] label {
    font-size: 8px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(0,242,255,0.45) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    color: #00f2ff !important;
    text-shadow: 0 0 12px rgba(0,242,255,0.5) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,242,255,0.18) !important;
    border-radius: 4px !important;
}
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #00f2ff !important;
    letter-spacing: 3px !important;
    text-shadow: 0 0 12px rgba(0,242,255,0.35) !important;
}
p, li { color: rgba(0,242,255,0.65) !important; font-size: 11px !important; letter-spacing: 1px !important; }
.stInfo, .stSuccess, .stWarning, .stError {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
}

/* ── WELLNESS SUGGESTIONS PANEL ───────────────────────── */
.wellness-list {
    margin: 8px 0 0 0;
    padding-left: 18px;
    list-style-type: square;
}
.wellness-list li {
    margin-bottom: 11px;
    line-height: 1.55;
    color: rgba(0,242,255,0.78);
    font-size: 11px;
    letter-spacing: 0.5px;
}
.wellness-list li strong {
    color: #9efcff;
    font-weight: 600;
}
.wellness-foot {
    font-size: 8px;
    letter-spacing: 1.5px;
    color: rgba(0,242,255,0.36);
    margin-top: 12px;
    border-top: 1px solid rgba(0,242,255,0.12);
    padding-top: 10px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;padding:14px 20px;
     background:rgba(0,10,16,0.85);border:1px solid rgba(0,242,255,0.22);border-radius:4px;position:relative;">
  <div style="position:absolute;top:-1px;left:-1px;width:14px;height:14px;
       border-top:2px solid #00f2ff;border-left:2px solid #00f2ff;"></div>
  <div style="position:absolute;bottom:-1px;right:-1px;width:14px;height:14px;
       border-bottom:2px solid #00f2ff;border-right:2px solid #00f2ff;"></div>
  <div style="width:44px;height:44px;border:1.5px solid #00f2ff;border-radius:50%;
       display:flex;align-items:center;justify-content:center;font-size:22px;
       box-shadow:0 0 14px rgba(0,242,255,0.45),inset 0 0 10px rgba(0,242,255,0.12);">🧠</div>
  <div>
    <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:900;letter-spacing:4px;
         color:#00f2ff;text-shadow:0 0 14px rgba(0,242,255,0.5);">NEUROPULSE <span style="color:rgba(0,242,255,0.45);font-size:13px;letter-spacing:2px;">AI</span></div>
    <div style="font-size:8px;letter-spacing:4px;color:rgba(0,242,255,0.38);margin-top:2px;">
      COGNITIVE FATIGUE DETECTION SYSTEM &nbsp;// &nbsp;BEHAVIORAL ML ENGINE &nbsp;// &nbsp;XGBoost v2.4.1
    </div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
    <div style="width:7px;height:7px;border-radius:50%;background:#00ff41;
         box-shadow:0 0 8px #00ff41;animation:pulse-green 1.5s infinite;"></div>
    <span style="font-size:9px;letter-spacing:3px;color:#00ff41;">NEURAL LINK ONLINE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("""
<div style="font-family:Orbitron,monospace;font-size:11px;font-weight:700;letter-spacing:3px;
     color:#00f2ff;padding:8px 0 12px;border-bottom:1px solid rgba(0,242,255,0.2);margin-bottom:12px;">
  ◆ APPLICATION MODE
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Mode",
    ["Neural Dashboard", "Vision Mode"],
    index=0,
    label_visibility="collapsed",
    help="Neural Dashboard: behavioral fatigue ML. Vision Mode: webcam EAR/MAR + ONNX (app.py).",
)

if app_mode == "Vision Mode":
    st.markdown("""
    <div class="hud-module" style="margin-bottom:16px;">
      <div class="mod-label">Vision Mode</div>
      <p style="margin:0 0 8px 0;">
        Runs the OpenCV + MediaPipe / cascade + ONNX pipeline from <strong>app.py</strong>
        in a <strong>separate process</strong>. A camera window opens outside the browser.
        Press <strong>Q</strong> in that window to stop capture. Live metrics are read from
        <code>vision_output.json</code> in the project root.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not _VISION_APP.is_file():
        st.error(f"Could not find **app.py** at `{_VISION_APP}`. Place `app.py` in the project root.")
    else:
        st.code(str(_VISION_APP), language="text")

        if "vision_proc" not in st.session_state:
            st.session_state.vision_proc = None

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Launch vision pipeline", type="primary", use_container_width=True):
                proc = st.session_state.vision_proc
                if proc is not None and proc.poll() is None:
                    st.warning("Vision pipeline is already running. Close the OpenCV window (press Q) first, or stop the other process.")
                else:
                    st.session_state.vision_proc = subprocess.Popen(
                        [sys.executable, str(_VISION_APP)],
                        cwd=str(_PROJECT_ROOT),
                    )
                    st.success("Started **app.py**. Check the OpenCV window; press **Q** to quit.")
        with col_b:
            if st.button("↻ Refresh data", use_container_width=True):
                st.rerun()

        proc = st.session_state.vision_proc
        if proc is not None:
            code = proc.poll()
            if code is None:
                st.info("Vision process: **running** (PID %d)." % proc.pid)
            else:
                st.caption("Vision process: exited with code `%s`." % code)

        st.sidebar.markdown("---")
        auto_refresh = st.sidebar.checkbox(
            "Auto-refresh vision (~2/s)",
            value=False,
            help="Reruns the page periodically while in Vision Mode so metrics stay current.",
        )

        st.subheader("Live Vision Analysis")
        if os.path.isfile(_VISION_OUTPUT_JSON):
            try:
                with open(_VISION_OUTPUT_JSON, encoding="utf-8") as f:
                    data = json.load(f)
                fs = float(data.get("fatigue_score", 0.0))
                lbl = str(data.get("label", "—"))

                st.metric("Fatigue score", f"{fs * 100:.1f}%")
                st.caption(f"Internal label: **{lbl}** · file `{_VISION_OUTPUT_JSON.name}`")

                if fs > 0.7:
                    st.error("⚠️ DROWSY")
                elif fs > 0.4:
                    st.warning("⚠️ Moderate fatigue")
                else:
                    st.success("✅ Alert")

                st.progress(min(max(fs, 0.0), 1.0))
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                st.warning(f"Could not read vision output: `{e}`")
        else:
            st.caption(
                f"No `{_VISION_OUTPUT_JSON.name}` yet — launch the pipeline and wait for the first frame."
            )

        if auto_refresh:
            time.sleep(0.45)
            st.rerun()

    st.stop()

st.sidebar.markdown("""
<div style="font-family:Orbitron,monospace;font-size:11px;font-weight:700;letter-spacing:3px;
     color:#00f2ff;padding:8px 0 12px;border-bottom:1px solid rgba(0,242,255,0.2);margin-bottom:12px;">
  ◈ DEMO PROFILES
</div>
""", unsafe_allow_html=True)

profile = st.sidebar.radio(
    "Select Scenario",
    ["Focused Coder", "Distracted Multitasker", "Mild Fatigue", "High Fatigue", "Manual Input"],
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# PROFILE PRESETS
# ---------------------------------------------------------

if profile == "Focused Coder":
    vals = dict(typing_speed_kpm=230, backspace_rate=0.02, mean_key_interval_ms=115,
                std_key_interval_ms=28, mouse_distance_px=9500, mouse_speed_px_per_sec=160,
                click_rate_per_min=16, mean_click_latency_ms=90, idle_time_sec=1.5, session_minutes=40)
elif profile == "Distracted Multitasker":
    vals = dict(typing_speed_kpm=160, backspace_rate=0.08, mean_key_interval_ms=210,
                std_key_interval_ms=95, mouse_distance_px=7000, mouse_speed_px_per_sec=140,
                click_rate_per_min=20, mean_click_latency_ms=220, idle_time_sec=7, session_minutes=55)
elif profile == "Mild Fatigue":
    vals = dict(typing_speed_kpm=135, backspace_rate=0.10, mean_key_interval_ms=320,
                std_key_interval_ms=120, mouse_distance_px=4200, mouse_speed_px_per_sec=75,
                click_rate_per_min=9, mean_click_latency_ms=340, idle_time_sec=8, session_minutes=150)
elif profile == "High Fatigue":
    vals = dict(typing_speed_kpm=72, backspace_rate=0.17, mean_key_interval_ms=510,
                std_key_interval_ms=185, mouse_distance_px=1800, mouse_speed_px_per_sec=30,
                click_rate_per_min=4, mean_click_latency_ms=620, idle_time_sec=15, session_minutes=310)
else:
    vals = dict(typing_speed_kpm=150, backspace_rate=0.05, mean_key_interval_ms=250,
                std_key_interval_ms=80, mouse_distance_px=5000, mouse_speed_px_per_sec=100,
                click_rate_per_min=10, mean_click_latency_ms=250, idle_time_sec=5, session_minutes=60)

# ---------------------------------------------------------
# SIDEBAR SLIDERS
# ---------------------------------------------------------

st.sidebar.markdown("""
<div style="font-family:Orbitron,monospace;font-size:11px;font-weight:700;letter-spacing:3px;
     color:#00f2ff;padding:12px 0 12px;border-bottom:1px solid rgba(0,242,255,0.2);margin-bottom:12px;">
  ◉ FEATURE CONTROLS
</div>
""", unsafe_allow_html=True)

typing_speed_kpm        = st.sidebar.slider("Typing Speed (KPM)",      0,   350,  int(vals["typing_speed_kpm"]))
backspace_rate          = st.sidebar.slider("Backspace Rate",           0.0, 0.30, float(vals["backspace_rate"]),      step=0.01)
mean_key_interval_ms    = st.sidebar.slider("Mean Key Interval (ms)",   50,  1000, int(vals["mean_key_interval_ms"]))
std_key_interval_ms     = st.sidebar.slider("Typing Variability",       0,   300,  int(vals["std_key_interval_ms"]))
mouse_distance_px       = st.sidebar.slider("Mouse Distance (px)",      0,   15000,int(vals["mouse_distance_px"]))
mouse_speed_px_per_sec  = st.sidebar.slider("Mouse Speed (px/s)",       0,   300,  int(vals["mouse_speed_px_per_sec"]))
click_rate_per_min      = st.sidebar.slider("Clicks / Min",             0,   30,   int(vals["click_rate_per_min"]))
mean_click_latency_ms   = st.sidebar.slider("Click Latency (ms)",       0,   1200, int(vals["mean_click_latency_ms"]))
idle_time_sec           = st.sidebar.slider("Idle Time (sec)",          0.0, 30.0, float(vals["idle_time_sec"]),       step=0.5)
session_minutes         = st.sidebar.slider("Session Duration (min)",   0,   480,  int(vals["session_minutes"]))

manual_ear = 0.28
manual_mar = 0.30
rag_time_of_day = "afternoon"
rag_session_type = "coding"
manual_rag_analyze = True
if profile == "Manual Input":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 RAG Demo Controls")
    manual_ear = st.sidebar.slider("EAR", 0.10, 0.40, 0.22, step=0.01)
    manual_mar = st.sidebar.slider("MAR", 0.10, 0.60, 0.36, step=0.01)
    MCR = st.sidebar.slider("MCR (RAG)", 0.40, 1.0, 0.65, step=0.01)
    MPTI = st.sidebar.slider("MPTI (RAG)", 0.30, 0.90, 0.58, step=0.01)
    ITSL = st.sidebar.slider("ITSL (RAG, ms)", 100, 500, 320)
    TRCV = st.sidebar.slider("TRCV (RAG)", 0.30, 0.90, 0.54, step=0.01)
    DCR = st.sidebar.slider("DCR (RAG)", 0.10, 0.60, 0.28, step=0.01)
    rag_time_of_day = st.sidebar.selectbox("Time of day", ["morning", "afternoon", "evening"], index=1)
    rag_session_type = st.sidebar.selectbox(
        "Session type", ["deep_work", "meetings", "coding", "writing", "studying"], index=2
    )
    manual_rag_analyze = st.sidebar.button("🚀 Analyze", use_container_width=True)

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

result     = predict_fatigue(
    typing_speed_kpm, backspace_rate, mean_key_interval_ms, std_key_interval_ms,
    mouse_distance_px, mouse_speed_px_per_sec, click_rate_per_min,
    mean_click_latency_ms, idle_time_sec, session_minutes
)

label      = result["label"]
confidence = result["confidence"]
probs      = result["probabilities"]

# ---------------------------------------------------------
# BIOMARKERS
# ---------------------------------------------------------

if profile != "Manual Input":
    MCR  = round(backspace_rate, 4)
    MPTI = round(1 + (std_key_interval_ms / 300), 3)
    ITSL = round(idle_time_sec * 1000, 2)
    TRCV = round(std_key_interval_ms / max(mean_key_interval_ms, 1), 4)
    DCR  = round(click_rate_per_min * backspace_rate, 2)

auto_ear_by_label = {"Alert": 0.29, "Mild Fatigue": 0.22, "High Fatigue": 0.17}
auto_mar_by_label = {"Alert": 0.27, "Mild Fatigue": 0.37, "High Fatigue": 0.46}
rag_ear = round(manual_ear if profile == "Manual Input" else auto_ear_by_label.get(label, 0.24), 2)
rag_mar = round(manual_mar if profile == "Manual Input" else auto_mar_by_label.get(label, 0.34), 2)
rag_time_of_day = rag_time_of_day if profile == "Manual Input" else "afternoon"
rag_session_type = rag_session_type if profile == "Manual Input" else profile.lower().replace(" ", "_")
rag_fatigue_level = to_rag_fatigue_level(label)

# ---------------------------------------------------------
# STATE COLOUR MAPPING
# ---------------------------------------------------------

if label == "Alert":
    state_color    = "#00ff41"
    state_shadow   = "rgba(0,255,65,0.55)"
    state_bg       = "rgba(0,255,65,0.07)"
    state_border   = "rgba(0,255,65,0.35)"
    hud_class      = "hud-state-alert"
    banner_class   = "banner-alert"
    banner_icon    = "◈"
    banner_msg     = "COGNITIVE STATE STABLE — OPERATING AT FULL CAPACITY"
elif label == "Mild Fatigue":
    state_color    = "#ff8c00"
    state_shadow   = "rgba(255,140,0,0.55)"
    state_bg       = "rgba(255,140,0,0.07)"
    state_border   = "rgba(255,140,0,0.35)"
    hud_class      = "hud-state-mild"
    banner_class   = "banner-mild"
    banner_icon    = "◉"
    banner_msg     = "MILD FATIGUE RISING — EFFICIENCY DEGRADATION IMMINENT"
else:
    state_color    = "#ff003c"
    state_shadow   = "rgba(255,0,60,0.55)"
    state_bg       = "rgba(255,0,60,0.08)"
    state_border   = "rgba(255,0,60,0.45)"
    hud_class      = "hud-state-high"
    banner_class   = "banner-high"
    banner_icon    = "⚠"
    banner_msg     = "HIGH FATIGUE DETECTED — IMMEDIATE BREAK RECOMMENDED"

# ---------------------------------------------------------
# MAIN LAYOUT — ROW 1: STATE + CONFIDENCE
# ---------------------------------------------------------

st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div class="hud-module {hud_class}">
      <div class="mod-label">Neural State</div>
      <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:900;letter-spacing:4px;
           text-align:center;padding:14px 12px;border-radius:3px;
           color:{state_color};background:{state_bg};border:1px solid {state_border};
           text-shadow:0 0 14px {state_shadow};">
        {label.upper()}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    conf_bar_w = confidence
    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Confidence Index</div>
      <div style="text-align:center;">
        <div style="font-size:8px;letter-spacing:3px;color:rgba(0,242,255,0.4);">MODEL CERTAINTY</div>
        <div style="font-family:Orbitron,monospace;font-size:32px;font-weight:900;
             color:#00f2ff;text-shadow:0 0 14px rgba(0,242,255,0.5);margin-top:4px;">
          {confidence}<span style="font-size:16px;">%</span>
        </div>
      </div>
      <svg width="100%" height="4" style="margin-top:10px;display:block;">
        <rect width="100%" height="4" rx="2" fill="rgba(0,242,255,0.08)"/>
        <rect width="{conf_bar_w}%" height="4" rx="2" fill="#00f2ff"
              style="filter:drop-shadow(0 0 4px rgba(0,242,255,0.6));"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# ROW 2: PREDICTION BREAKDOWN + GAUGES
# ---------------------------------------------------------

col3, col4 = st.columns([1, 1])

with col3:
    p_alert = int(probs["Alert"])
    p_mild  = int(probs["Mild Fatigue"])
    p_high  = int(probs["High Fatigue"])
    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Prediction Breakdown</div>

      <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:9px;letter-spacing:2px;color:rgba(0,242,255,0.45);text-transform:uppercase;">Alert</span>
          <span style="font-size:10px;font-weight:700;color:#00ff41;">{p_alert}%</span>
        </div>
        <svg width="100%" height="5"><rect width="100%" height="5" rx="2" fill="rgba(0,255,65,0.08)"/>
          <rect width="{p_alert}%" height="5" rx="2" fill="#00ff41"
                style="filter:drop-shadow(0 0 5px rgba(0,255,65,0.7));"/></svg>
      </div>

      <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:9px;letter-spacing:2px;color:rgba(0,242,255,0.45);text-transform:uppercase;">Mild Fatigue</span>
          <span style="font-size:10px;font-weight:700;color:#ff8c00;">{p_mild}%</span>
        </div>
        <svg width="100%" height="5"><rect width="100%" height="5" rx="2" fill="rgba(255,140,0,0.08)"/>
          <rect width="{p_mild}%" height="5" rx="2" fill="#ff8c00"
                style="filter:drop-shadow(0 0 5px rgba(255,140,0,0.7));"/></svg>
      </div>

      <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:9px;letter-spacing:2px;color:rgba(0,242,255,0.45);text-transform:uppercase;">High Fatigue</span>
          <span style="font-size:10px;font-weight:700;color:#ff003c;">{p_high}%</span>
        </div>
        <svg width="100%" height="5"><rect width="100%" height="5" rx="2" fill="rgba(255,0,60,0.08)"/>
          <rect width="{p_high}%" height="5" rx="2" fill="#ff003c"
                style="filter:drop-shadow(0 0 5px rgba(255,0,60,0.7));"/></svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Arc gauge SVG helper
    import math

    def arc_gauge_svg(val, max_val, label, unit="", color="#00f2ff"):
        pct   = min(val / max_val, 1.0)
        r     = 30
        cx, cy = 38, 38
        start  = -220 * math.pi / 180
        sweep  = 260 * math.pi / 180
        sweep_partial = sweep * pct
        end    = start + sweep_partial
        x1     = cx + r * math.cos(start)
        y1     = cy + r * math.sin(start)
        x2     = cx + r * math.cos(start + sweep)
        y2     = cy + r * math.sin(start + sweep)
        ex     = cx + r * math.cos(end)
        ey     = cy + r * math.sin(end)
        la0    = 1 if sweep > math.pi else 0
        # Value arc must run from track *start* to partial *end* (same direction as track).
        # Drawing end→full-end made SVG pick the wrong arc and stroke through the dial interior.
        la_partial = 1 if sweep_partial > math.pi else 0
        disp   = f"{val:.2f}" if isinstance(val, float) and val < 1 else str(int(round(val)))
        arc_path = (
            f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 {la_partial} 1 {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="4.5" stroke-linecap="round" '
            f'style="filter:drop-shadow(0 0 5px {color});"/>'
        ) if pct > 0.002 else ""
        return f"""
        <svg width="76" height="76" viewBox="0 0 76 76">
          <path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 {la0} 1 {x2:.1f} {y2:.1f}"
                fill="none" stroke="rgba(0,242,255,0.08)" stroke-width="4.5" stroke-linecap="round"/>
          {arc_path}
          <text x="38" y="33" text-anchor="middle"
                font-family="Orbitron,monospace" font-size="10" font-weight="700" fill="{color}">{disp}</text>
          <text x="38" y="48" text-anchor="middle"
                font-family="JetBrains Mono,monospace" font-size="6"
                fill="rgba(0,242,255,0.38)" letter-spacing="1">{unit}</text>
        </svg>
        <div style="font-size:7px;letter-spacing:2px;color:rgba(0,242,255,0.38);
             text-align:center;margin-top:2px;text-transform:uppercase;">{label}</div>"""

    def gauge_color(pct, invert=False):
        if invert:
            return "#00ff41" if pct > 0.6 else ("#ff8c00" if pct > 0.3 else "#ff003c")
        return "#ff003c" if pct > 0.75 else ("#ff8c00" if pct > 0.45 else "#00f2ff")

    gauges = [
        (typing_speed_kpm,       350,  "KPM",       "kpm",   True),
        (mean_click_latency_ms,  1200, "Latency",   "ms",    False),
        (idle_time_sec,          30,   "Idle",      "sec",   False),
        (backspace_rate,         0.30, "Err Rate",  "",      False),
        (session_minutes,        480,  "Session",   "min",   False),
        (mouse_speed_px_per_sec, 300,  "Cursor",    "px/s",  True),
    ]

    gauge_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">'
    for val, mx, lbl, unit, inv in gauges:
        pct = min(val / mx, 1.0)
        col = gauge_color(pct, invert=inv)
        gauge_html += f'<div>{arc_gauge_svg(val, mx, lbl, unit, col)}</div>'
    gauge_html += '</div>'

    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Behavioral Sensors</div>
      {gauge_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown(build_wellness_suggestions_html(profile, label), unsafe_allow_html=True)

if rag_fatigue_level in {"moderate", "high"} and (profile != "Manual Input" or manual_rag_analyze):
    st.markdown("---")
    st.markdown("## 🧠 RAG Fatigue Coach")

    current_state = {
        "fatigue_level": rag_fatigue_level,
        "EAR": rag_ear,
        "MAR": rag_mar,
        "MCR": round(MCR, 3),
        "MPTI": round(MPTI, 3),
        "ITSL": int(ITSL),
        "TRCV": round(TRCV, 3),
        "DCR": round(DCR, 3),
        "time_of_day": rag_time_of_day,
        "session_type": rag_session_type,
    }

    summary = {}
    retrieved = []
    with st.spinner("🔍 Searching your fatigue history..."):
        retriever = get_session_retriever()
        if retriever is not None:
            retrieved = retriever.retrieve(current_state, top_k=5)

    st.markdown("### 📂 Retrieved from your session history")
    left_col, right_col = st.columns([1.7, 1.1])
    with left_col:
        for idx, row in enumerate(retrieved, start=1):
            sim_pct = int(round(row["similarity"] * 100))
            style = score_badge_style(sim_pct)
            st.markdown(
                f"<div style='margin-bottom:6px;'>• Match {idx}: {row['date']} at {row['time_of_day']} "
                f"<span style='padding:2px 8px;border-radius:999px;font-size:11px;letter-spacing:1px;{style}'>{sim_pct}%</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["similarity"], 0.0), 1.0))
            pretty_row = {
                "when": f"{row['date']} {row['time_of_day']}",
                "fatigue_level": row.get("fatigue_level"),
                "session_type": row.get("session_type"),
                "intervention_tried": row.get("intervention_suggested"),
                "recovery_pct": row.get("recovery_pct"),
                "time_to_recover_mins": row.get("time_to_recover_mins"),
                "pattern": row.get("fatigue_pattern", {}),
                "notes": row.get("notes", ""),
            }
            with st.expander(f"View match {idx} details (JSON)", expanded=False):
                st.json(pretty_row)
    with right_col:
        if retriever is not None:
            summary = retriever.get_intervention_summary(retrieved)
        if summary:
            chart_df = pd.DataFrame(
                {
                    "Intervention": list(summary.keys()),
                    "Avg Recovery %": list(summary.values()),
                }
            ).set_index("Intervention")
            st.markdown("#### 📊 Retrieved intervention impact")
            st.bar_chart(chart_df)
        else:
            st.info("Not enough session history yet.")

    if generate_coaching_recommendation is None:
        ai_text = (
            "RAG dependencies are unavailable in this environment. "
            f"Install `requirements_rag.txt` to enable Claude coaching. ({RAG_IMPORT_ERROR})"
        )
    else:
        ai_text = generate_coaching_recommendation(current_state, retrieved)
    #st.markdown("### 💬 AI Recommendation")
    #st.markdown(
    #    f"<div class='hud-module' style='margin-top:0;'><div style='font-size:13px;line-height:1.6;'>{ai_text}</div></div>",
    #    unsafe_allow_html=True,
    #)

    best_name, best_avg = top_intervention(summary)
    st.markdown("###  Top intervention for YOU")
    m1, m2 = st.columns([1.2, 1])
    with m1:
        st.metric("Best intervention", best_name)
    with m2:
        st.metric("Avg recovery", f"{best_avg:.1f}%")
    st.markdown(f"**{best_name}**  \nBased on your history: **{best_avg:.1f}% avg recovery**")

    st.markdown("### 📝 Session logging")
    if st.button("📝 Log this session", key="rag_log_btn"):
        if retriever is None:
            st.error("RAG retriever is unavailable. Install `requirements_rag.txt` first.")
        else:
            new_record = {
                "date": time.strftime("%Y-%m-%d"),
                "time_of_day": time.strftime("%H:%M"),
                "day_type": "weekday" if time.localtime().tm_wday < 5 else "weekend",
                "fatigue_pattern": {
                    "EAR": current_state["EAR"],
                    "MAR": current_state["MAR"],
                    "MCR": current_state["MCR"],
                    "MPTI": current_state["MPTI"],
                    "ITSL": current_state["ITSL"],
                    "TRCV": current_state["TRCV"],
                    "DCR": current_state["DCR"],
                },
                "fatigue_level": current_state["fatigue_level"],
                "intervention_suggested": best_name,
                "intervention_taken": False,
                "pre_fatigue_score": round(float(confidence) / 100, 2),
                "post_fatigue_score": round(max(0.0, (float(confidence) / 100) - 0.25), 2),
                "recovery_pct": int(round(best_avg)) if best_avg > 0 else 0,
                "time_to_recover_mins": 12,
                "session_type": current_state["session_type"],
                "notes": "Logged from dashboard RAG panel",
            }
            logged = retriever.append_session(new_record)
            st.session_state["last_logged_session_id"] = logged["session_id"]
            st.success(f"Logged session `{logged['session_id']}` to session history.")

    interventions = ["7-min walk", "coffee break", "power nap", "breathing exercise", "screen break"]
    chosen = st.selectbox("Did you take a break? What did you do?", ["Not yet"] + interventions, key="rag_taken_action")
    last_id = st.session_state.get("last_logged_session_id")
    if last_id and chosen != "Not yet" and st.button("✅ Update latest logged session", key="rag_update_btn"):
        if retriever is None:
            st.error("RAG retriever is unavailable. Install `requirements_rag.txt` first.")
        else:
            ok = retriever.update_session_intervention(last_id, intervention=chosen, intervention_taken=True)
            if ok:
                st.success(f"Updated `{last_id}` with intervention: {chosen}.")
            else:
                st.warning("Could not update the latest logged session.")

# ---------------------------------------------------------
# ROW 3: BIOMARKERS + INPUT VECTOR + TERMINAL
# ---------------------------------------------------------

col5, col6, col7 = st.columns([1, 1, 1])

with col5:
    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Cognitive Biomarkers</div>
      <table class="cyber-table">
        <tr><td>MCR</td>  <td style="color:rgba(0,242,255,0.45);font-size:9px;">Micro-Correction Rate</td>     <td>{MCR}</td></tr>
        <tr><td>MPTI</td> <td style="color:rgba(0,242,255,0.45);font-size:9px;">Mouse Path Tortuosity</td>     <td>{MPTI}</td></tr>
        <tr><td>ITSL</td> <td style="color:rgba(0,242,255,0.45);font-size:9px;">Inter-Task Switch Latency</td>  <td>{ITSL} ms</td></tr>
        <tr><td>TRCV</td> <td style="color:rgba(0,242,255,0.45);font-size:9px;">Typing Rhythm Coeff. Var.</td>  <td>{TRCV}</td></tr>
        <tr><td>DCR</td>  <td style="color:rgba(0,242,255,0.45);font-size:9px;">Double Click Rate</td>         <td>{DCR}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with col6:
    feat_rows = "".join([
        f"<tr><td>{k.replace('_',' ').title()}</td><td>{round(v,3) if isinstance(v,float) else v}</td></tr>"
        for k, v in [
            ("typing_speed_kpm",       typing_speed_kpm),
            ("backspace_rate",         backspace_rate),
            ("mean_key_interval_ms",   mean_key_interval_ms),
            ("std_key_interval_ms",    std_key_interval_ms),
            ("mouse_distance_px",      mouse_distance_px),
            ("mouse_speed_px_per_sec", mouse_speed_px_per_sec),
            ("click_rate_per_min",     click_rate_per_min),
            ("mean_click_latency_ms",  mean_click_latency_ms),
            ("idle_time_sec",          idle_time_sec),
            ("session_minutes",        session_minutes),
        ]
    ])
    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Input Vector</div>
      <table class="cyber-table">{feat_rows}</table>
    </div>
    """, unsafe_allow_html=True)

with col7:
    t_state_cls = "t-green" if label == "Alert" else ("t-orange" if label == "Mild Fatigue" else "t-red")
    t_status    = ("✓ NOMINAL" if label == "Alert"
                   else ("! FATIGUE WARNING" if label == "Mild Fatigue"
                         else "!! FATIGUE CRITICAL"))
    st.markdown(f"""
    <div class="hud-module">
      <div class="mod-label">Diagnostic Log</div>
      <div class="terminal">
        <span class="t-dim">[SYS]</span> <span class="t-cyan">INIT</span> xgboost.predict()<br>
        <span class="t-dim">[DAT]</span> loading 10 behavioral features...<br>
        <span class="t-dim">[BIO]</span> MCR={MCR} &nbsp;TRCV={TRCV}<br>
        <span class="t-dim">[BIO]</span> ITSL={ITSL}ms &nbsp;sess={session_minutes}min<br>
        <span class="t-dim">[MDL]</span> <span class="t-cyan">ensemble vote</span> → <span class="{t_state_cls}">{label.upper()}</span><br>
        <span class="t-dim">[OUT]</span> confidence={confidence}%<br>
        <span class="t-dim">[OUT]</span> p_alert={probs['Alert']}% &nbsp;p_mild={probs['Mild Fatigue']}%<br>
        <span class="{t_state_cls}">STATUS: {t_status}</span><br>
        <span class="t-dim">_</span><span style="animation:blink 1s step-end infinite;color:#00f2ff;">█</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL INSIGHT
# ---------------------------------------------------------

st.markdown("""
<div class="hud-module" style="margin-top:4px;">
  <div class="mod-label">Model Insight — Top Learned Predictors</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;font-size:9px;letter-spacing:1px;">
    <div style="color:rgba(0,242,255,0.5);">01 / Idle Time<div style="color:#00f2ff;margin-top:4px;font-size:11px;">████████░</div></div>
    <div style="color:rgba(0,242,255,0.5);">02 / Click Hesitation<div style="color:#00f2ff;margin-top:4px;font-size:11px;">███████░░</div></div>
    <div style="color:rgba(0,242,255,0.5);">03 / Session Duration<div style="color:#00f2ff;margin-top:4px;font-size:11px;">██████░░░</div></div>
    <div style="color:rgba(0,242,255,0.5);">04 / Typing Variability<div style="color:#00f2ff;margin-top:4px;font-size:11px;">█████░░░░</div></div>
    <div style="color:rgba(0,242,255,0.5);">05 / Error Rate<div style="color:#00f2ff;margin-top:4px;font-size:11px;">████░░░░░</div></div>
  </div>
  <div style="margin-top:10px;font-size:9px;letter-spacing:2px;color:rgba(0,242,255,0.4);">
    CROSS-VALIDATED XGBoost ACCURACY: <span style="color:#00ff41;font-weight:700;">96.84%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# STATUS BANNER
# ---------------------------------------------------------

st.markdown(f"""
<div class="{banner_class}" style="margin-top:6px;">
  {banner_icon} &nbsp; {banner_msg}
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MENTAL HEALTH WELLNESS BUDDY
# ---------------------------------------------------------

import streamlit as st
import os
from groq import Groq

# ---------------------------------------------------------
# MENTAL HEALTH WELLNESS BUDDY (GROQ VERSION)
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## 🤖 Wellness Buddy ")
st.markdown("A mental health wellness companion to converse with you and support you.")


load_dotenv() 

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)

        # Initialize chat history
        if "buddy_messages" not in st.session_state:
            st.session_state.buddy_messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your Mental Health Wellness Buddy. I'm here to listen, support, and console you. How are you feeling right now?"
                }
            ]

        # Display chat history
        for msg in st.session_state.buddy_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        if prompt := st.chat_input("Share what's on your mind..."):
            st.session_state.buddy_messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    # Convert history to Groq format
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a warm, empathetic, and supportive mental health wellness buddy. Help users process stress and fatigue. Be kind, gentle, and human-like. Keep responses short. Do not give medical advice."
                        }
                    ]

                    # Add chat history
                    for m in st.session_state.buddy_messages:
                        messages.append({
                            "role": m["role"],
                            "content": m["content"]
                        })

                    # Generate response
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",  # ✅ best free model on Groq
                        messages=messages
                    )

                    full_response = response.choices[0].message.content

                except Exception as e:
                    # Fallback (never crash)
                    full_response = (
                        "Hey… I’m here with you. It sounds like you might be going through something heavy. "
                        "Take your time — do you want to tell me a bit more about what’s been draining you? 💛"
                    )

                message_placeholder.markdown(full_response)

                st.session_state.buddy_messages.append({
                    "role": "assistant",
                    "content": full_response
                })

    except ImportError:
        st.error("Please install Groq SDK using: pip install groq")

else:
    st.info("The Wellness Buddy is offline. Please set the GROQ_API_KEY environment variable.")