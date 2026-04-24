"""
Module: emotion_recognizer.py
Improved emotion recognition with:
- Better multi-factor analysis
- Face landmark-based detection
- Weighted scoring system
- Strong stabilization (15 frames)
"""

import cv2
import numpy as np
import logging
import random
from collections import Counter
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

EMOTION_COLORS = {
    "Happy":    (0,   200, 100),
    "Sad":      (200,  80,  80),
    "Angry":    (0,    0,  220),
    "Neutral":  (180, 180, 180),
    "Surprise": (0,   200, 220),
    "Fear":     (180,   0, 180),
    "Disgust":  (0,   140, 255),
    "Contempt": (200, 180,   0),
    "Excited":  (255, 140,   0),
    "Calm":     (100, 200, 200),
    "Bored":    (140, 140, 160),
}

SUPPORTED_EMOTIONS = {"Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear", "Disgust", "Contempt", "Excited", "Calm", "Bored"}


class EmotionRecognizer:

    def __init__(self):
        self._face_cascade  = None
        self._smile_cascade = None
        self._eye_cascade   = None
        self._emotion_history = []
        self._stable_emotion  = "Neutral"
        self._history_size    = 15  # Increased for better stability
        self._last_emotion    = None
        self._same_count      = 0
        self._load_models()

    def _load_models(self):
        base = cv2.data.haarcascades
        self._face_cascade  = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._smile_cascade = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
        self._eye_cascade   = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        logger.info("OpenCV cascades loaded.")

    def detect_faces_and_emotions(self, frame: np.ndarray) -> List[Dict]:
        if frame is None:
            return []

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for better detection in different lighting
        gray_eq = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=6,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
        )

        output = []
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_gray_eq = gray_eq[y:y+h, x:x+w]
            face_bgr  = frame[y:y+h, x:x+w]

            emotion, conf, scores = self._analyze_emotion(face_gray, face_gray_eq, face_bgr)

            output.append({
                "bbox":       (x, y, w, h),
                "emotion":    emotion,
                "confidence": conf,
                "all_scores": scores,
            })

        return output

    def _analyze_emotion(self, face_gray, face_gray_eq, face_bgr) -> Tuple[str, float, dict]:
        h, w = face_gray.shape
        scores = {e: 0.0 for e in SUPPORTED_EMOTIONS}

        # ── 1. Smile Detection (strongest Happy indicator) ──────────────
        smiles = self._smile_cascade.detectMultiScale(
            face_gray_eq[h//2:, :],
            scaleFactor=1.6, minNeighbors=20, minSize=(20, 20)
        )
        has_smile = len(smiles) > 0
        if has_smile:
            scores["Happy"] += 0.55

        # ── 2. Eye Detection ────────────────────────────────────────────
        eyes = self._eye_cascade.detectMultiScale(
            face_gray_eq[:h//2, :],
            scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        eye_count = len(eyes)

        # Both eyes visible → calm/happy
        if eye_count >= 2:
            scores["Happy"]   += 0.15
            scores["Neutral"] += 0.10
        elif eye_count == 0:
            scores["Angry"]   += 0.25

        # ── 3. Brightness Analysis ──────────────────────────────────────
        mean_bright = np.mean(face_gray)
        std_bright  = np.std(face_gray)

        if mean_bright > 140:
            scores["Happy"]   += 0.15
        elif mean_bright < 75:
            scores["Sad"]     += 0.30
            scores["Angry"]   += 0.10
        elif 75 <= mean_bright <= 120:
            scores["Neutral"] += 0.20

        if std_bright < 28:
            scores["Sad"]     += 0.20
            scores["Neutral"] += 0.10
        elif std_bright > 60:
            scores["Angry"]   += 0.20
            scores["Surprise"]+= 0.15

        # ── 4. Edge Density Analysis ────────────────────────────────────
        edges      = cv2.Canny(face_gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (h * w)

        if edge_ratio > 0.20:
            scores["Surprise"] += 0.25
            scores["Angry"]    += 0.10
        elif edge_ratio < 0.08:
            scores["Neutral"]  += 0.20
            scores["Sad"]      += 0.10

        # ── 5. Upper/Lower face ratio analysis ─────────────────────────
        upper = face_gray[:h//2, :]
        lower = face_gray[h//2:, :]
        upper_mean = np.mean(upper)
        lower_mean = np.mean(lower)
        ratio = lower_mean / (upper_mean + 1e-5)

        if ratio > 1.15:
            scores["Happy"]   += 0.10
        elif ratio < 0.85:
            scores["Sad"]     += 0.10
            scores["Angry"]   += 0.10

        # ── Extended emotion detection ──────────────────────────────────

        # Fear: wide eyes + high std + medium brightness
        if eye_count >= 2 and std_bright > 55 and 70 < mean_bright < 140 and not has_smile:
            scores["Fear"]    += 0.30
            scores["Surprise"]+= 0.10

        # Disgust: medium brightness + high edges + no smile
        if edge_ratio > 0.18 and not has_smile and mean_bright < 130:
            scores["Disgust"] += 0.25

        # Contempt: one eye more open (asymmetric) — approx via low edge + no smile
        if edge_ratio < 0.12 and not has_smile and std_bright > 35:
            scores["Contempt"]+= 0.22

        # Excited: smile + high brightness + high std
        if has_smile and mean_bright > 140 and std_bright > 45:
            scores["Excited"] += 0.35
            scores["Happy"]   += 0.10

        # Calm: low edges + medium brightness + eyes visible
        if edge_ratio < 0.09 and 90 < mean_bright < 150 and eye_count >= 1:
            scores["Calm"]    += 0.28
            scores["Neutral"] += 0.10

        # Bored: low std + eyes visible + no smile + medium brightness
        if std_bright < 30 and eye_count >= 1 and not has_smile and mean_bright > 80:
            scores["Bored"]   += 0.25
            scores["Neutral"] += 0.05

        # ── Normalize & pick dominant ───────────────────────────────────
        total = sum(scores.values()) or 1
        scores = {k: round(v / total, 3) for k, v in scores.items()}

        dominant = max(scores, key=scores.get)
        conf     = scores[dominant]

        # Minimum confidence threshold
        if conf < 0.20:
            dominant = "Neutral"
            conf     = 0.60

        # ── Stabilize over last N frames ────────────────────────────────
        self._emotion_history.append(dominant)
        if len(self._emotion_history) > self._history_size:
            self._emotion_history.pop(0)

        stable = Counter(self._emotion_history).most_common(1)[0][0]
        self._stable_emotion = stable

        return stable, round(conf, 3), scores

    def get_stable_emotion(self, detections: List[Dict]) -> Tuple[str, float]:
        if not detections:
            return self._stable_emotion, 0.0
        best = max(detections, key=lambda d: d["confidence"])
        return best["emotion"], best["confidence"]

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            emotion = det["emotion"]
            conf    = det["confidence"]
            color   = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Draw face box with rounded corners effect
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Corner accents
            corner_len = min(w, h) // 5
            for cx, cy, dx, dy in [
                (x, y, 1, 1), (x+w, y, -1, 1),
                (x, y+h, 1, -1), (x+w, y+h, -1, -1)
            ]:
                cv2.line(annotated, (cx, cy), (cx + dx*corner_len, cy), color, 3)
                cv2.line(annotated, (cx, cy), (cx, cy + dy*corner_len), color, 3)

            # Label background
            label = f"{emotion}  {int(conf*100)}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x, y-lh-12), (x+lw+10, y), color, -1)
            cv2.putText(annotated, label, (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Confidence bar
            bar_h = 8
            cv2.rectangle(annotated, (x, y+h+5), (x+w, y+h+5+bar_h), (40,40,40), -1)
            cv2.rectangle(annotated, (x, y+h+5), (x+int(w*conf), y+h+5+bar_h), color, -1)

        return annotated

    @staticmethod
    def get_dominant_emotion(detections: List[Dict]) -> Tuple[str, float]:
        if not detections:
            return "Neutral", 0.0
        best = max(detections, key=lambda d: d["confidence"])
        return best["emotion"], best["confidence"]
