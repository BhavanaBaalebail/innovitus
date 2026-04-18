# Fatigue detection (minimal project)

## Layout

```
app.py                  # Webcam + scoring + UI
train.py                # Retrain CNN → best_model.pth
export_model_onnx.py    # best_model.pth → fatigue_model.onnx
best_model.pth          # PyTorch weights (optional after export)
fatigue_model.onnx      # ONNX graph
fatigue_model.onnx.data # Large weights (keep with .onnx)
training_history.json   # Last train metrics (optional)
capture.py              # Optional image capture helper
data/train/drowsy/      # Positive class images
data/train/notdrowsy/   # Negative class images
data/val/               # Optional holdout images (same subfolders if used)
README.md
```

## Run

```bash
python3 app.py
```

Dependencies: `opencv-python`, `numpy`, `onnxruntime`, `torch` / `torchvision` (train + export only), optional `mediapipe` for landmark-based EAR/MAR.

---

## End-to-end flow

1. **Capture** (optional): `capture.py` can populate `data/train/...` with stills from the camera.

2. **Training** (`train.py`): Reads all images under `data/train/drowsy` (label 1) and `data/train/notdrowsy` (label 0). Each image is loaded with OpenCV, converted RGB, resized to **64×64**. Augmentation: resize, random resized crop (0.8–1.0), horizontal flip, rotation ±15°, color jitter, then **ToTensor** and **Normalize(mean=0.5, std=0.5)** per channel. An **80/20** split is used; if class counts differ in the training split, **WeightedRandomSampler** balances batches. Optimizer **Adam** lr **3e-4**, **CrossEntropyLoss**, **8** epochs; best **validation accuracy** checkpoint is written to **`best_model.pth`**. History is saved to **`training_history.json`**.

3. **Export** (`export_model_onnx.py`): Loads **`best_model.pth`** into the same **`FatigueDetectionCNN`** definition, exports **`fatigue_model.onnx`** (opset **18**, input name **`image`**, logits shape **(batch, 2)**). Large weights may live in **`fatigue_model.onnx.data`**.

4. **Runtime** (`app.py`): Reads webcam frames. **Face / eyes / mouth**: if **MediaPipe** Face Mesh is importable, EAR is computed from the first six indices per eye (standard vertical/horizontal ratio), MAR from outer mouth landmark spread in pixels. If MediaPipe is unavailable, **Haar cascades** supply continuous **EAR-like** and **MAR-like** proxies from eye boxes and lower-face / smile regions. **ONNX**: face crop is resized to **64×64**, RGB, normalized **(x/255 − 0.5)/0.5** to match training; **softmax** on logits gives **`model_score` = P(drowsy)** (class index 1).

5. **Fusion (continuous, no hard yawning/eye rules)**  
   - **`eye_score`** = clamp\(((0.5 - \mathrm{EAR}) / (0.5 - 0.15)), 0, 1\) — higher when eyes are more closed.  
   - **`mouth_score`** = clamp\((\mathrm{MAR} / 0.4), 0, 1\) — higher when mouth is more open.  
   - **`fatigue_score`** = **0.5 × eye_score + 0.3 × mouth_score + 0.2 × model_score**  
   - Internal label: **`DROWSY`** if `fatigue_score > 0.5`, else **`ALERT`**.  
   - **Confidence** shown is **`fatigue_score × 100`** (0–100).  
   - **Note:** The **on-screen text and colors intentionally swap** internal `DROWSY`/`ALERT` (see `draw_status` in `app.py`) so the HUD does not match the internal string; logic and score are unchanged.

---

## CNN architecture (custom)

**`FatigueDetectionCNN`** in `train.py` / `export_model_onnx.py`:

| Block | Details |
|--------|---------|
| Conv + ReLU + MaxPool | 3→32, 3×3, pad 1; pool 2 |
| Conv + ReLU + MaxPool | 32→64 |
| Conv + ReLU + MaxPool | 64→128 |
| Flatten | 128 × 8 × 8 |
| FC + ReLU + Dropout(0.4) | → 256 |
| FC | → 2 logits (notdrowsy, drowsy) |

Weights live in **`best_model.pth`** (state dict only). After export, inference uses **`fatigue_model.onnx`** (+ **`.data`** if present).

---

## Accuracy (last recorded train)

From **`training_history.json`** (same 8-epoch schedule as `train.py`):

- **Best validation accuracy:** **0.7632** (~**76.3%**) on the held-out 20% split.  
- Final epoch train accuracy ~**0.761**, validation ~**0.746** (see file for full curves).

Re-running `train.py` on new images will refresh **`best_model.pth`** and **`training_history.json`**; then run **`export_model_onnx.py`** before deploying a new **`fatigue_model.onnx`**.

---

## Dataset

- **Classes:** binary **drowsy** vs **notdrowsy** (alert).  
- **Location:** `data/train/drowsy/` and `data/train/notdrowsy/` (JPG/PNG, etc.).  
- **Optional:** `data/val/` with the same subfolder names if you add a separate validation pipeline later; current `train.py` only uses `data/train` with a random split.

---

## Debug

In `app.py`, **`DEBUG_PRINT_EVERY_N_FRAMES`** controls how often EAR, eye_score, MAR, mouth_score, model score, and final fatigue are printed to the console.
