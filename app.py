"""
Continuous fatigue scoring: EAR + MAR + ONNX (weighted), with recalibration,
strong-drowsy boosts, alert dampening, and a short temporal average for stability.
"""

from collections import deque
import json
import os
import sys
import time
from pathlib import Path

import cv2
import math
import numpy as np
import onnxruntime as ort

# Audio alert (macOS afplay); cooldown in main loop
last_beep_time = 0.0

# Streamlit Vision Mode: live bridge file (project root, next to app.py)
VISION_OUTPUT_JSON = Path(__file__).resolve().parent / "vision_output.json"


def beep():
    """Short non-blocking system sound (macOS)."""
    if sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Glass.aiff &")


def beep_strong():
    """Optional stronger alert for very low confidence (macOS)."""
    if sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Sosumi.aiff &")

# --- MediaPipe (landmarks) when available ---
HAS_MEDIAPIPE = False
mp = None
try:
    import mediapipe as mp_pkg

    try:
        from mediapipe.python import _framework_bindings  # noqa: F401

        mp = mp_pkg
        HAS_MEDIAPIPE = True
    except Exception:
        pass
except ImportError:
    pass

if not HAS_MEDIAPIPE:
    print("ℹ️  MediaPipe not available. Using cascade-based EAR/MAR proxies.")

# ==================== CONFIG ====================

# EAR → eye_score = (0.5 - ear) / (0.5 - 0.18), clamped [0, 1]
EAR_OPEN_REF = 0.5
EAR_CLOSED_REF = 0.15  # still used in cascade proxy clipping
EYE_SCORE_DENOM = EAR_OPEN_REF - 0.18  # 0.32 — stronger sensitivity vs closed eyes

# MAR → mouth_score = mar / MAR_SCALE
MAR_SCALE = 0.5

# Temporal smoothing of final fatigue (post recalibration)
SCORE_BUFFER_MAXLEN = 5

# Weighted fusion
W_EYE = 0.5
W_MOUTH = 0.3
W_MODEL = 0.2

FATIGUE_DECISION_THRESHOLD = 0.5

# STEP 8 debug lines; set to 1 for every frame
DEBUG_PRINT_EVERY_N_FRAMES = 15

COLOR_ALERT = (0, 255, 0)
COLOR_DROWSY = (0, 0, 255)

# MediaPipe indices (face mesh) — same lists as app_clean; EAR uses first 6 per eye
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
MOUTH_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 178, 179,
    181, 176,
]


# ==================== DETECTOR ====================


class FatigueDetector:
    def __init__(self, onnx_model_path="fatigue_model.onnx"):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

        self.face_mesh = None
        if HAS_MEDIAPIPE and mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        try:
            self.session = ort.InferenceSession(
                onnx_model_path,
                providers=["CPUExecutionProvider"],
            )
            self.has_onnx = True
            self._onnx_input_name = self.session.get_inputs()[0].name
            print(f"✅ ONNX model loaded: {onnx_model_path}")
        except Exception as e:
            print(f"⚠️  ONNX model not available: {e}")
            self.session = None
            self.has_onnx = False
            self._onnx_input_name = "image"

        self.frame_index = 0
        self.score_buffer = deque(maxlen=SCORE_BUFFER_MAXLEN)

    def compute_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_landmark_coords(self, landmarks, indices, w, h):
        coords = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                coords.append((lm.x * w, lm.y * h))
        return coords

    def compute_ear_from_coords(self, eye_coords):
        """Standard EAR from 6 points (pixel coords)."""
        if len(eye_coords) < 6:
            return EAR_OPEN_REF
        p1 = eye_coords[1]
        p2 = eye_coords[2]
        p3 = eye_coords[3]
        p4 = eye_coords[4]
        p5 = eye_coords[5]
        p6 = eye_coords[0]
        vert1 = self.compute_distance(p1, p3)
        vert2 = self.compute_distance(p2, p4)
        horiz = self.compute_distance(p5, p6)
        if horiz <= 1e-6:
            return EAR_OPEN_REF
        return (vert1 + vert2) / (2.0 * horiz)

    def compute_mar_from_coords(self, mouth_coords):
        """MAR = vertical / horizontal span of outer mouth (pixel coords)."""
        if len(mouth_coords) < 2:
            return 0.0
        top = min(c[1] for c in mouth_coords)
        bottom = max(c[1] for c in mouth_coords)
        left = min(c[0] for c in mouth_coords)
        right = max(c[0] for c in mouth_coords)
        vert = bottom - top
        horiz = right - left
        if horiz <= 1e-6:
            return 0.0
        return vert / horiz

    def cascade_ear_continuous(self, face_roi_gray, face_w, face_h):
        """Smooth EAR-like proxy from eye boxes vs face area (no landmarks)."""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        face_area = max(face_w * face_h, 1)
        if len(eyes) == 0:
            return EAR_CLOSED_REF
        total = sum(float(ew * eh) for (_, _, ew, eh) in eyes)
        ratio = total / face_area
        # Map area ratio into ~[EAR_CLOSED_REF, EAR_OPEN_REF]
        ear = float(np.clip(0.12 + ratio * 3.2, EAR_CLOSED_REF, 0.55))
        return ear

    def cascade_mar_continuous(self, mouths, face_roi_gray):
        """MAR-like openness in a scale compatible with mouth_score = mar / 0.4."""
        fh, fw = face_roi_gray.shape[:2]
        face_area = max(fh * fw, 1)
        if len(mouths) > 0:
            ratios = [(mw * mh) / face_area for (_, _, mw, mh) in mouths]
            r = max(ratios)
            return float(np.clip(0.15 + r * 2.8, 0.05, 0.95))
        lower = face_roi_gray[int(fh * 0.55) :, :]
        if lower.size == 0:
            return 0.12
        blur = cv2.GaussianBlur(lower, (5, 5), 0)
        std = float(np.std(blur))
        t = np.clip((std - 8.0) / 28.0, 0.0, 1.0)
        return float(np.clip(0.08 + 0.22 * t, 0.05, 0.38))

    def preprocess_face_onnx(self, frame, x1, y1, x2, y2):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        face_img = cv2.resize(face_img, (64, 64))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, 0)
        return face_img.astype(np.float32)

    def get_model_score(self, frame, face_bbox):
        if not self.has_onnx:
            return 0.0
        x1, y1, x2, y2 = face_bbox
        try:
            face_img = self.preprocess_face_onnx(frame, x1, y1, x2, y2)
            if face_img is None:
                return 0.0
            outputs = self.session.run(None, {self._onnx_input_name: face_img})
            logits = np.asarray(outputs[0][0], dtype=np.float64).reshape(-1)
            if logits.size < 2:
                return 0.0
            z = logits - np.max(logits)
            e = np.exp(z)
            probs = e / np.sum(e)
            return float(probs[1])
        except Exception:
            return 0.0

    def detect(self, frame):
        self.frame_index += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- MediaPipe path ----------
        if self.face_mesh is not None:
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                self.score_buffer.clear()
                return "NO FACE", 0.0, {}

            landmarks = results.multi_face_landmarks[0].landmark
            left = self.get_landmark_coords(landmarks, LEFT_EYE, w, h)
            right = self.get_landmark_coords(landmarks, RIGHT_EYE, w, h)
            mouth_pts = self.get_landmark_coords(landmarks, MOUTH_OUTER, w, h)

            left_ear = self.compute_ear_from_coords(left)
            right_ear = self.compute_ear_from_coords(right)
            ear = (left_ear + right_ear) / 2.0
            mar = self.compute_mar_from_coords(mouth_pts)
            eye_source = "MediaPipe"

            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            pad = 20
            x1 = int(min(xs)) - pad
            y1 = int(min(ys)) - pad
            x2 = int(max(xs)) + pad
            y2 = int(max(ys)) + pad
            face_bbox = (x1, y1, x2, y2)
        else:
            # ---------- Cascade fallback ----------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) == 0:
                self.score_buffer.clear()
                return "NO FACE", 0.0, {}

            fx, fy, fw, fh = faces[0]
            face_roi = gray[fy : fy + fh, fx : fx + fw]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            mouths = self.mouth_cascade.detectMultiScale(
                face_roi, scaleFactor=1.3, minNeighbors=8, minSize=(20, 20)
            )

            ear = self.cascade_ear_continuous(face_roi, fw, fh)
            mar = self.cascade_mar_continuous(mouths, face_roi)
            eye_source = "Cascade"
            face_bbox = (fx, fy, fx + fw, fy + fh)

        # ---------- Scores (continuous) ----------
        eye_score = (EAR_OPEN_REF - ear) / EYE_SCORE_DENOM
        eye_score = float(max(0.0, min(1.0, eye_score)))

        mouth_score = mar / MAR_SCALE
        mouth_score = float(max(0.0, min(1.0, mouth_score)))

        model_score = self.get_model_score(frame, face_bbox)
        model_score = float(max(0.0, min(1.0, model_score)))

        fatigue_score = W_EYE * eye_score + W_MOUTH * mouth_score + W_MODEL * model_score

        # Strong drowsy boosts (eyes very closed / yawning mouth)
        if ear < 0.22:
            fatigue_score += 0.25
        if ear < 0.18:
            fatigue_score += 0.35
        if mar > 0.35:
            fatigue_score += 0.2

        fatigue_score = float(max(0.0, min(1.0, fatigue_score)))

        # Pull down clearly alert frames (wide eyes, closed mouth)
        if ear > 0.4 and mar < 0.15:
            fatigue_score *= 0.3

        fatigue_score = float(max(0.0, min(1.0, fatigue_score)))

        # Temporal smoothing (reduces frame-to-frame flicker)
        self.score_buffer.append(fatigue_score)
        smoothed_score = sum(self.score_buffer) / len(self.score_buffer)
        fatigue_score_raw = fatigue_score
        fatigue_score = float(smoothed_score)

        if fatigue_score > FATIGUE_DECISION_THRESHOLD:
            status = "DROWSY"
        else:
            status = "ALERT"

        confidence = fatigue_score * 100.0

        features = {
            "ear": ear,
            "mar": mar,
            "eye_score": eye_score,
            "mouth_score": mouth_score,
            "model_score": model_score,
            "fatigue_score": fatigue_score,
            "fatigue_score_raw": fatigue_score_raw,
            "face_bbox": face_bbox,
            "eye_source": eye_source,
        }

        if self.frame_index % max(1, DEBUG_PRINT_EVERY_N_FRAMES) == 0:
            print(f"EAR: {ear:.3f} | MAR: {mar:.3f} | Score: {fatigue_score:.2f}")

        return status, confidence, features


# ==================== VISUALIZATION ====================


def draw_status(frame, status, confidence, features):
    # Display-only inversion: internal logic unchanged; swap what the user sees.
    if status == "DROWSY":
        display_label = "ALERT"
    elif status == "ALERT":
        display_label = "DROWSY"
    else:
        display_label = status

    if display_label == "ALERT":
        color = (0, 255, 0)
    elif display_label == "DROWSY":
        color = (0, 0, 255)
    else:
        color = COLOR_ALERT

    if "face_bbox" in features:
        x1, y1, x2, y2 = features["face_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"{display_label} ({confidence:.0f}%)"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    ear = features.get("ear", 0.0)
    mar = features.get("mar", 0.0)
    es = features.get("eye_score", 0.0)
    ms = features.get("mouth_score", 0.0)
    mscore = features.get("model_score", 0.0)
    fs = features.get("fatigue_score", 0.0)
    src = features.get("eye_source", "")

    cv2.putText(
        frame,
        f"EAR {ear:.3f} eye_s {es:.2f} [{src}]",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        frame,
        f"MAR {mar:.3f} mouth_s {ms:.2f} | model {mscore:.2f} | fatigue {fs:.2f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )


# ==================== MAIN ====================


def main():
    global last_beep_time

    print("\n" + "=" * 75)
    print("Continuous fatigue scoring (EAR + MAR + model)")
    print("=" * 75)
    print(f"Weights: eye={W_EYE} mouth={W_MOUTH} model={W_MODEL}")
    print(f"Label: DROWSY if fatigue > {FATIGUE_DECISION_THRESHOLD}")
    print("=" * 75 + "\n")

    cap = cv2.VideoCapture(0)
    detector = FatigueDetector()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        status, confidence, features = detector.detect(frame)
        draw_status(frame, status, confidence, features)

        fatigue_score = float(
            features.get("fatigue_score", confidence / 100.0)
            if features
            else confidence / 100.0
        )
        vision_data = {
            "fatigue_score": fatigue_score,
            "label": status,
            "confidence_pct": float(confidence),
        }
        try:
            with open(VISION_OUTPUT_JSON, "w", encoding="utf-8") as vf:
                json.dump(vision_data, vf)
        except OSError:
            pass

        current_time = time.time()
        if confidence < 35 and (current_time - last_beep_time > 3):
            if confidence < 20:
                beep_strong()
            else:
                beep()
            last_beep_time = current_time

        cv2.imshow("Fatigue Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection stopped")


if __name__ == "__main__":
    main()
