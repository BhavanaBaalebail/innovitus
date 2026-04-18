# Innovitus / NeuroPulse — Full Project Context (for reports, PPTs, LLM prompts)

This document summarizes the **fatigue detection system** as implemented in this repository. Use it as **single-source context** when generating presentations, technical reports, documentation, or when onboarding other tools or LLMs.

---

## 1. Purpose & positioning

The project is a **cognitive / physical fatigue awareness system** with **two complementary tracks**:

| Track | What it measures | Typical input | Output |
|--------|------------------|---------------|--------|
| **Vision / physiological** | Eye closure, mouth openness, face-appearance “drowsy” cues | Webcam (`app.py`) | Continuous **fatigue score** 0–1 (shown as 0–100%), binary internal label DROWSY vs ALERT |
| **Behavioral / interaction** (Streamlit dashboard) | Typing, mouse, idle, session patterns | Sliders simulating user behavior | **XGBoost** 3-class: Alert, Mild Fatigue, High Fatigue + probabilities |

They are **independent models** and **different modalities**; the Streamlit UI can expose both **Neural Dashboard** (behavioral ML) and **Vision Mode** (launch `app.py` + read live JSON).

---

## 2. Repository layout (conceptual)

```
project_root/
├── app.py                    # Main OpenCV + optional MediaPipe + ONNX vision pipeline
├── train.py                  # PyTorch training → best_model.pth
├── export_model_onnx.py      # best_model.pth → fatigue_model.onnx (+ optional .data)
├── capture.py                # Optional: capture training images
├── best_model.pth            # CNN weights (PyTorch state dict)
├── fatigue_model.onnx        # Exported CNN (ONNX opset 18)
├── fatigue_model.onnx.data   # External weights blob (keep next to .onnx if present)
├── training_history.json     # Train/val curves from last train.py run
├── vision_output.json        # Written at runtime by app.py for dashboard bridge
├── data/train/drowsy/        # Class 1: drowsy face crops / images
├── data/train/notdrowsy/     # Class 0: alert / non-drowsy
├── data/val/                 # Optional separate validation folders (same structure)
├── README.md
└── src/                      # If present: Streamlit dashboard + behavioral predict
    ├── dashboard.py          # NeuroPulse UI: modes + Vision integration
    ├── predict.py            # Loads XGBoost + scaler from models/
    └── ...
models/
    ├── behavior_xgb.pkl
    └── xgb_scaler.pkl
```

Paths may vary; **`predict.py`** resolves `models/` relative to project root.

---

## 3. Vision pipeline (`app.py`) — detailed

### 3.1 Face & signals

- **Preferred path:** **MediaPipe Face Mesh** (if import succeeds) for landmarks.
  - **EAR** (eye aspect ratio): computed from **6 points per eye** (standard vertical/horizontal geometry) on **pixel** landmark coordinates; left and right averaged.
  - **MAR** (mouth aspect): vertical / horizontal span of **outer mouth** landmark set (ratio in pixel space).
- **Fallback:** **OpenCV Haar cascades** (face, eyes, smile/mouth proxy) with **continuous proxies** for EAR-like and MAR-like values when MediaPipe is unavailable.

### 3.2 CNN (ONNX)

- Crops the face region, resizes to **64×64**, **BGR→RGB**, normalizes **(x/255 − 0.5) / 0.5** per channel (matches training).
- Runs **ONNX Runtime**; input tensor name resolved dynamically (`image` in export).
- **Softmax** over 2 logits → **`model_score` = P(class “drowsy”)** (index 1).

### 3.3 Score fusion (continuous — not a pure rule engine)

**Normalized components (clamped to [0,1]):**

- `eye_score = clip((0.5 − EAR) / (0.5 − 0.18), 0, 1)` — stronger emphasis on closed eyes vs older 0.15 denominator.
- `mouth_score = clip(MAR / 0.5, 0, 1)`.

**Base fusion:**

`fatigue_score = 0.5 × eye_score + 0.3 × mouth_score + 0.2 × model_score`

**Heuristic adjustments (then clipped to [0,1]):**

- If `EAR < 0.22`: +0.25  
- If `EAR < 0.18`: +0.35 (stacks with above for very closed eyes)  
- If `MAR > 0.35`: +0.20 (open mouth / yawning proxy)  
- If `EAR > 0.4` and `MAR < 0.15`: ×0.30 (strong “alert-looking” dampening)

**Temporal smoothing:** last **5** instantaneous (post-adjustment) scores in a **deque**; **displayed fatigue_score** = **rolling mean** (reduces flicker).

**Decision:** internal **`DROWSY`** if `fatigue_score > 0.5`, else **`ALERT`**.

**Confidence shown:** `fatigue_score × 100` (0–100%).

### 3.4 Display quirk (important for demos)

**`draw_status`** may **swap** the **on-screen** label vs internal state (internal logic uses true DROWSY/ALERT; HUD text/colors can show the opposite for legacy UX). When describing “what the user sees,” distinguish **internal label** vs **display label** if screenshots are used.

### 3.5 Audio (macOS)

- Optional **afplay** beeps when **confidence** (0–100 scale) is **below** thresholds, with **cooldown** (e.g. Glass / Sosumi sounds), non-blocking (`&`).

### 3.6 Bridge file for Streamlit

Each processed frame (with a face), **`app.py`** writes **`vision_output.json`** at project root, e.g.:

```json
{
  "fatigue_score": 0.0-1.0,
  "label": "DROWSY|ALERT|NO FACE",
  "confidence_pct": 0-100
}
```

The Streamlit **Vision Mode** reads this file to show **live metrics**, **progress bar**, and status bands, with optional **auto-refresh** or manual refresh.

---

## 4. Training pipeline (`train.py`)

- **Data:** `data/train/drowsy/` (label 1), `data/train/notdrowsy/` (label 0); glob `*.*`.
- **Load:** OpenCV, RGB, **64×64** base size.
- **Augmentation (train):** PIL path: resize, random resized crop (0.8–1.0), flip, rotation ±15°, color jitter, **ToTensor**, **Normalize(0.5, 0.5)** per channel.
- **Split:** 80% train / 20% val (random shuffle, seed 42).
- **Balance:** **WeightedRandomSampler** if class counts differ in training split.
- **Model:** `FatigueDetectionCNN` — 3× (Conv→ReLU→MaxPool) with channels **32→64→128**, then **Linear(8192→256)** + ReLU + **Dropout(0.4)** + **Linear(256→2)**.
- **Optimizer:** Adam **lr = 3e-4**; **CrossEntropyLoss**; **8 epochs**; save **`best_model.pth`** on best **validation accuracy**.
- **Artifact:** **`training_history.json`** stores train/val loss and accuracy per epoch.

**Reported example performance** (from a completed run in `training_history.json`):

- **Best validation accuracy ≈ 76.3%** (0.7632) on the held-out split.

---

## 5. Export (`export_model_onnx.py`)

- Loads **`best_model.pth`** into the **same** `FatigueDetectionCNN` class.
- Exports **`fatigue_model.onnx`**, **opset 18**, dynamic batch axes optional; input name **`image`** aligned with runtime.
- May produce **`fatigue_model.onnx.data`** for large weights (ONNX external data).

---

## 6. Behavioral / dashboard track (Streamlit)

- **UI theme:** “NeuroPulse AI” cyberpunk HUD (Streamlit + custom CSS).
- **`predict.py`:** Loads **joblib** **`behavior_xgb.pkl`** + **`xgb_scaler.pkl`** from **`models/`**.
- **Input:** 10 behavioral features (typing speed, backspace rate, key intervals, mouse distance/speed, click rate, click latency, idle time, session minutes).
- **Output:** 3-class **Alert / Mild Fatigue / High Fatigue** + **probabilities** + **confidence** = max softmax × 100.
- **Profiles:** Preset scenarios (e.g. Focused Coder, High Fatigue) + manual sliders.
- **Vision Mode (if implemented in `src/dashboard.py`):** launches **`app.py`** via **`subprocess`** with **`cwd`** = project root; reads **`vision_output.json`** for live vision metrics; optional auto-refresh.

**Note:** Marketing copy in the dashboard may cite **~96.84%** cross-validated XGBoost accuracy — that refers to the **behavioral** model in **`models/`**, **not** the vision CNN.

---

## 7. Dependencies (typical)

- **Vision:** `opencv-python`, `numpy`, `onnxruntime`, optional `mediapipe`
- **Train/export:** `torch`, `torchvision`
- **Dashboard:** `streamlit`, `pandas`, `joblib`, `scikit-learn` (for XGB path as used by saved model)

---

## 8. How to describe the system in one paragraph (elevator pitch)

> The system combines **webcam-based physiological cues** (eye and mouth geometry, plus a small **CNN** on face crops) into a **single continuous fatigue score**, with **temporal smoothing** for stable feedback. Separately, a **behavioral analytics** dashboard uses **keyboard and mouse features** with an **XGBoost** classifier for **three fatigue levels**. A **Streamlit** interface can present the behavioral engine and optionally **launch** the real-time vision app while **mirroring** its output via a small **JSON** file.

---

## 9. Suggested slide outline (PPT)

1. Title — NeuroPulse / Innovitus fatigue intelligence  
2. Problem — monitoring cognitive & visual fatigue during computer use  
3. Two modalities — Vision vs Behavioral  
4. Vision pipeline diagram — Webcam → EAR/MAR + CNN → fusion → score  
5. CNN architecture table — conv layers + FC  
6. Training data & augmentation  
7. Metrics — val acc ~76% (vision); behavioral model metrics from `predict`/dashboard  
8. Live demo — OpenCV window + optional Streamlit Vision panel  
9. Limitations — lighting, pose, cascade fallback, not medical-grade  
10. Future work — more data, calibration, user-specific thresholds  

---

## 10. Limitations & ethics (for reports)

- **Not a medical device**; for awareness / UX research / productivity only.  
- Performance depends on **dataset diversity**, **lighting**, and **camera quality**.  
- **Display label inversion** in the OpenCV HUD can confuse viewers unless explained.  
- **Privacy:** webcam data should be processed locally; clarify data retention policy in real deployments.

---

## 11. File quick reference

| File | Role |
|------|------|
| `app.py` | Real-time vision fatigue detection + JSON bridge + optional beeps |
| `train.py` | Train CNN |
| `export_model_onnx.py` | PyTorch → ONNX |
| `capture.py` | Optional dataset collection |
| `vision_output.json` | Runtime live state for dashboard (generated) |
| `src/dashboard.py` | Streamlit NeuroPulse UI (if present) |
| `src/predict.py` | Behavioral XGBoost inference |
| `models/*.pkl` | Behavioral ML artifacts |

---

*Last aligned with codebase concepts: vision fusion with EAR denom 0.32, MAR scale 0.5, boosts, smoothing deque=5, ONNX `FatigueDetectionCNN`, Streamlit Vision JSON integration.*
