"""
gui.py — v5 Modern Edition
Improvements:
  ✅ Modern UI with gradients & better cards
  ✅ Toast notifications when emotion detected
  ✅ Improved layout & typography
  ✅ Better color scheme
  ✅ Smooth animations
"""

import os, sys, logging, datetime
from typing import Optional
import cv2, numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QDialog, QFileDialog,
    QGridLayout, QProgressBar, QSlider, QStackedWidget,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QSplashScreen, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QLinearGradient, QPainterPath, QPen, QBrush

from modules.emotion_recognizer import EmotionRecognizer
from modules.recommender import SongRecommender
from modules.database import (log_emotion_event, get_emotion_analytics,
    get_recent_history, start_session, end_session, clear_history)
from modules.analytics import SessionAnalytics, EMOTION_BAR_COLORS
from modules.notification import ToastNotification

logger = logging.getLogger(__name__)
SONGS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "songs")

# ── Themes ────────────────────────────────────────────────────────────────────
DARK = dict(
    BG="#0a0a18", SIDEBAR="#0f0f22", CARD="#13132a", CARD2="#1a1a35",
    TEXT="#eeeef8", DIM="#6868a0", BORDER="#222245",
    BLUE="#4a90d9", GREEN="#00c864", RED="#e03030",
    PURPLE="#8860d0", ORANGE="#d07020",
    NAV_ACTIVE="#1e3a5f", NAV_HOVER="#1a1a35",
)
LIGHT = dict(
    BG="#f0f2f8", SIDEBAR="#e0e4f0", CARD="#ffffff", CARD2="#eaecf8",
    TEXT="#1a1a2e", DIM="#5a5a7a", BORDER="#c8cad8",
    BLUE="#2060b0", GREEN="#007840", RED="#c02020",
    PURPLE="#6040a0", ORANGE="#a05010",
    NAV_ACTIVE="#d0e0f8", NAV_HOVER="#d0d4e8",
)
_T = DARK.copy()
def C(k): return _T[k]

EMOTION_HEX = {
    "Happy":"#00c864",   "Sad":"#5080c8",    "Angry":"#e03030",
    "Neutral":"#a0a0a0", "Surprise":"#f0c030","Fear":"#b400b4",
    "Disgust":"#008cff", "Contempt":"#c8b400","Excited":"#ff8c00",
    "Calm":"#64c8c8",    "Bored":"#8c8ca0",
}

# ─────────────────────────────────────────────────────────────────────────────
# Splash Screen
# ─────────────────────────────────────────────────────────────────────────────
class SplashScreen(QSplashScreen):
    def __init__(self):
        pix = QPixmap(680, 380); pix.fill(Qt.transparent)
        p = QPainter(pix)
        g = QLinearGradient(0, 0, 680, 380)
        g.setColorAt(0, QColor("#0a0a18"))
        g.setColorAt(1, QColor("#1a1a3a"))
        p.fillRect(0, 0, 680, 380, g)

        # Decorative circles
        p.setBrush(QBrush(QColor(74, 144, 217, 30)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(500, -60, 280, 280)
        p.drawEllipse(-80, 260, 200, 200)

        p.setPen(QColor("#4a90d9"))
        p.setFont(QFont("Segoe UI", 24, QFont.Bold))
        p.drawText(0, 90, 680, 55, Qt.AlignCenter, "Face Expression Based")

        p.setPen(QColor("#00c864"))
        p.setFont(QFont("Segoe UI", 22, QFont.Bold))
        p.drawText(0, 148, 680, 50, Qt.AlignCenter, "Song Recommendation System")

        p.setPen(QColor("#7878a0"))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(0, 230, 680, 30, Qt.AlignCenter, "AI + Music Player")
        p.drawText(0, 268, 680, 30, Qt.AlignCenter, "Loading, please wait...")
        p.end()

        super().__init__(pix)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

# ─────────────────────────────────────────────────────────────────────────────
# Camera Worker
# ─────────────────────────────────────────────────────────────────────────────
class CameraWorker(QThread):
    frame_ready  = pyqtSignal(np.ndarray, list)
    camera_error = pyqtSignal(str)

    def __init__(self, recognizer):
        super().__init__()
        self._recognizer  = recognizer
        self._running     = False
        self._frame_count = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.camera_error.emit("Cannot open webcam."); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        last_det = []
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.camera_error.emit("Failed to read from camera."); break
            self._frame_count += 1
            frame = cv2.flip(frame, 1)
            if self._frame_count % 3 == 0:
                last_det = self._recognizer.detect_faces_and_emotions(frame)
            annotated = self._recognizer.annotate_frame(frame, last_det)
            self.frame_ready.emit(annotated, last_det)
            self.msleep(30)
        cap.release()

    def stop(self):
        self._running = False; self.wait()

# ─────────────────────────────────────────────────────────────────────────────
# Emotion Pie Chart
# ─────────────────────────────────────────────────────────────────────────────
class PieChartWidget(QWidget):
    COLORS = {
        "Happy":    QColor("#00c864"),
        "Sad":      QColor("#5080c8"),
        "Angry":    QColor("#e03030"),
        "Neutral":  QColor("#a0a0a0"),
        "Surprise": QColor("#f0c030"),
        "Fear":     QColor("#b400b4"),
        "Disgust":  QColor("#008cff"),
        "Contempt": QColor("#c8b400"),
        "Excited":  QColor("#ff8c00"),
        "Calm":     QColor("#64c8c8"),
        "Bored":    QColor("#8c8ca0"),
    }

    def __init__(self, session_analytics, parent=None):
        super().__init__(parent)
        self._sa = session_analytics
        self.setMinimumSize(420, 360)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        from PyQt5.QtCore import QRectF
        counts = self._sa.get_counts()
        total  = sum(counts.values()) if counts else 0
        w, h   = self.width(), self.height()
        cx, cy = w // 2 - 70, h // 2
        radius = min(cx, cy, 130)
        rect   = QRectF(cx-radius, cy-radius, radius*2, radius*2)

        if total == 0:
            painter.setPen(QPen(QColor("#3a3a60"), 2))
            painter.setBrush(QBrush(QColor("#1e1e35")))
            painter.drawEllipse(rect)
            painter.setPen(QColor("#7878a0"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(rect, Qt.AlignCenter, "No data yet\nStart Camera!")
            painter.end(); return

        start = 0
        legend_y = cy - len(counts) * 20

        for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            color = self.COLORS.get(emotion, QColor("#888"))
            span  = int(360 * 16 * count / total)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#0a0a18"), 2))
            painter.drawPie(rect, start, span)
            start += span

        # Legend
        lx = cx + radius + 25
        for i, (emotion, count) in enumerate(sorted(counts.items(), key=lambda x: x[1], reverse=True)):
            color = self.COLORS.get(emotion, QColor("#888"))
            pct   = round(100 * count / total, 1)
            ly    = legend_y + i * 44
            painter.setBrush(QBrush(color)); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(lx, ly, 16, 16, 3, 3)
            painter.setPen(QColor("#eeeef8"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(lx + 24, ly + 13, emotion)
            painter.setPen(QColor("#9090b0"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(lx + 24, ly + 28, f"{pct}%  ({count}x)")

        # Center donut hole
        painter.setBrush(QBrush(QColor("#0a0a18")))
        painter.setPen(Qt.NoPen)
        inner_r = radius * 0.42
        from PyQt5.QtCore import QPointF
        painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        painter.setPen(QColor("#eeeef8"))
        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        from PyQt5.QtCore import QRectF
        painter.drawText(QRectF(cx-inner_r, cy-inner_r, inner_r*2, inner_r*2),
                         Qt.AlignCenter, f"{total}\ndetections")
        painter.end()

# ─────────────────────────────────────────────────────────────────────────────
# Emotion Chart Panel
# ─────────────────────────────────────────────────────────────────────────────
class EmotionChartPanel(QWidget):
    def __init__(self, session_analytics, parent=None):
        super().__init__(parent)
        self._sa = session_analytics
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)

        title = QPushButton("🥧  Emotion Distribution Chart")
        title.setEnabled(False)
        title.setStyleSheet(
            "QPushButton { color: #eeeef8; font-size: 22px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0px; }"
        )
        lay.addWidget(title)

        sub = QPushButton("Real-time emotion distribution from this session")
        sub.setEnabled(False)
        sub.setStyleSheet(
            "QPushButton { color: #6868a0; font-size: 13px;"
            " background: transparent; border: none; text-align: left; padding: 0px; }"
        )
        lay.addWidget(sub)

        self._chart = PieChartWidget(self._sa)
        lay.addWidget(self._chart)

        refresh = QPushButton("Refresh Chart")
        refresh.setStyleSheet(
            "QPushButton { background: #1e3a5f; color: #eeeef8;"
            " border: 1px solid #4a90d9; border-radius: 10px;"
            " padding: 10px 28px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background: #2a4a70; }"
        )
        refresh.clicked.connect(lambda: (self._chart.repaint(), self._chart.update()))
        lay.addWidget(refresh, alignment=Qt.AlignCenter)
        lay.addStretch()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Button
# ─────────────────────────────────────────────────────────────────────────────
class SidebarBtn(QPushButton):
    def __init__(self, icon, label, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedHeight(52)
        self.setText(f"  {icon}  {label}" if icon else f"    {label}")
        self.setFont(QFont("Segoe UI", 11))
        self._refresh()

    def _refresh(self):
        active = self.isChecked()
        if active:
            self.setStyleSheet("""
                QPushButton { background: #1e3a5f; color: #4a90d9;
                    text-align: left; padding-left: 14px; border: none;
                    border-left: 3px solid #4a90d9; font-weight: bold; font-size: 13px; }
            """)
        else:
            self.setStyleSheet("""
                QPushButton { background: transparent; color: #6868a0;
                    text-align: left; padding-left: 17px; border: none;
                    font-size: 13px; }
                QPushButton:hover { background: #1a1a35; color: #eeeef8; }
            """)

    def setChecked(self, v):
        super().setChecked(v); self._refresh()

# ─────────────────────────────────────────────────────────────────────────────
# Home Panel
# ─────────────────────────────────────────────────────────────────────────────
class HomePanel(QWidget):
    def __init__(self, recognizer, recommender, session_analytics, main_window, parent=None):
        super().__init__(parent)
        self._rec   = recognizer
        self._recom = recommender
        self._sa    = session_analytics
        self._main  = main_window
        self._worker      = None
        self._running     = False
        self._last_frame  = None
        self._last_emotion = None
        self._screenshots = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "screenshots")
        os.makedirs(self._screenshots, exist_ok=True)
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        # ── Left: Camera ─────────────────────────────────────────────
        cam_col = QVBoxLayout()
        cam_col.setSpacing(12)

        # Camera frame with modern border
        cam_frame = QFrame()
        cam_frame.setStyleSheet(
            "QFrame { background: #050510; border-radius: 16px;"
            " border: 2px solid #222245; }"
        )
        cam_fl = QVBoxLayout(cam_frame)
        cam_fl.setContentsMargins(0, 0, 0, 0)

        self._cam_lbl = QLabel()
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setMinimumSize(540, 390)
        self._cam_lbl.setText("Camera stopped.\nPress Start Camera to begin.")
        self._cam_lbl.setStyleSheet(
            "color: #6868a0; font-size: 14px; background: transparent; border: none;"
        )
        cam_fl.addWidget(self._cam_lbl)
        cam_col.addWidget(cam_frame)

        # Status row
        status_row = QHBoxLayout()
        self._live_badge = QLabel("● STOPPED")
        self._live_badge.setStyleSheet(
            "color: #e03030; font-size: 12px; font-weight: bold; border: none;"
        )
        status_row.addWidget(self._live_badge)
        status_row.addStretch()
        self._face_lbl = QLabel("Faces: 0")
        self._face_lbl.setStyleSheet("color: #6868a0; font-size: 12px; border: none;")
        status_row.addWidget(self._face_lbl)
        cam_col.addLayout(status_row)

        # Control buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self._start_btn = self._btn("Start Camera", "#00c864", self._toggle_cam)
        self._pause_btn = self._btn("Pause",        "#8860d0", self._pause)
        self._next_btn  = self._btn("Next Song",    "#4a90d9", self._next)
        self._snap_btn  = self._btn("Snapshot",     "#d07020", self._snapshot)
        for b in [self._start_btn, self._pause_btn, self._next_btn, self._snap_btn]:
            btn_row.addWidget(b)
        cam_col.addLayout(btn_row)

        # Volume
        vol_row = QHBoxLayout()
        vol_lbl = QLabel("Volume")
        vol_lbl.setStyleSheet("color: #6868a0; font-size: 12px; border: none;")
        vol_row.addWidget(vol_lbl)
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100); self._vol_slider.setValue(75)
        self._vol_slider.setFixedHeight(20)
        self._vol_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #222245; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4a90d9; width: 16px; height: 16px;
                border-radius: 8px; margin: -5px 0; }
            QSlider::sub-page:horizontal { background: #4a90d9; border-radius: 3px; }
        """)
        self._vol_slider.valueChanged.connect(self._vol_change)
        vol_row.addWidget(self._vol_slider)
        self._vol_lbl = QLabel("75%")
        self._vol_lbl.setFixedWidth(38)
        self._vol_lbl.setStyleSheet("color: #6868a0; font-size: 12px; border: none;")
        vol_row.addWidget(self._vol_lbl)
        cam_col.addLayout(vol_row)

        root.addLayout(cam_col, stretch=3)

        # ── Right: Info Cards ─────────────────────────────────────────
        info_col = QVBoxLayout(); info_col.setSpacing(14)

        # Emotion card
        ec = self._card()
        el = QVBoxLayout(ec); el.setSpacing(6)
        el.addWidget(self._lbl("🎭  Detected Emotion", dim=True, size=11))
        self._emo_lbl  = self._lbl("—", color="#00c864", size=44, bold=True)
        self._conf_lbl = self._lbl("Confidence: —", dim=True, size=12)
        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100); self._conf_bar.setValue(0)
        self._conf_bar.setFixedHeight(6); self._conf_bar.setTextVisible(False)
        self._conf_bar.setStyleSheet("""
            QProgressBar { background: #222245; border-radius: 3px; border: none; }
            QProgressBar::chunk { background: #00c864; border-radius: 3px; }
        """)
        for w in [self._emo_lbl, self._conf_lbl, self._conf_bar]:
            el.addWidget(w)
        info_col.addWidget(ec)

        # Song card
        sc = self._card()
        sl = QVBoxLayout(sc); sl.setSpacing(6)
        sl.addWidget(self._lbl("🎵  Now Playing", dim=True, size=11))
        self._song_lbl = self._lbl("No song playing", color="#4a90d9", size=13, bold=True)
        self._song_lbl.setWordWrap(True)
        self._status_lbl = self._lbl("Status: Stopped", dim=True, size=11)
        sl.addWidget(self._song_lbl); sl.addWidget(self._status_lbl)
        info_col.addWidget(sc)

        # Stats card
        stc = self._card()
        stl = QVBoxLayout(stc); stl.setSpacing(6)
        stl.addWidget(self._lbl("📊  Session Stats", dim=True, size=11))
        self._stats_lbl = self._lbl("Songs played: 0\nDominant: —", size=12)
        stl.addWidget(self._stats_lbl)
        info_col.addWidget(stc)

        info_col.addStretch()
        root.addLayout(info_col, stretch=2)

    # ── Camera ────────────────────────────────────────────────────────
    def _toggle_cam(self):
        if self._running: self._stop_cam()
        else: self._start_cam()

    def _start_cam(self):
        self._worker = CameraWorker(self._rec)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.camera_error.connect(self._on_err)
        self._worker.start()
        self._running = True
        self._start_btn.setText("Stop Camera")
        self._start_btn.setStyleSheet(
            self._start_btn.styleSheet().replace("#00c864", "#e03030"))
        self._live_badge.setText("● LIVE")
        self._live_badge.setStyleSheet(
            "color: #00c864; font-size: 12px; font-weight: bold; border: none;")

    def _stop_cam(self):
        if self._worker: self._worker.stop(); self._worker = None
        self._recom.stop()
        self._running = False
        self._start_btn.setText("Start Camera")
        self._start_btn.setStyleSheet(
            self._start_btn.styleSheet().replace("#e03030", "#00c864"))
        self._live_badge.setText("● STOPPED")
        self._live_badge.setStyleSheet(
            "color: #e03030; font-size: 12px; font-weight: bold; border: none;")
        self._cam_lbl.setText("Camera stopped.\nPress Start Camera to begin.")
        self._emo_lbl.setText("—")
        self._song_lbl.setText("No song playing")

    def stop_camera(self):
        if self._running: self._stop_cam()

    def _on_frame(self, frame, detections):
        self._last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._cam_lbl.setPixmap(
            QPixmap.fromImage(img).scaled(
                self._cam_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._face_lbl.setText(f"Faces: {len(detections)}")

        if detections:
            from modules.emotion_recognizer import EmotionRecognizer as ER
            emotion, conf = ER.get_dominant_emotion(detections)
            col = EMOTION_HEX.get(emotion, "#4a90d9")

            self._emo_lbl.setText(emotion)
            self._emo_lbl.setStyleSheet(
                f"color: {col}; font-size: 44px; font-weight: bold; border: none;")
            self._conf_lbl.setText(f"Confidence: {int(conf*100)}%")
            self._conf_bar.setValue(int(conf*100))
            self._conf_bar.setStyleSheet(f"""
                QProgressBar {{ background: #222245; border-radius: 3px; border: none; }}
                QProgressBar::chunk {{ background: {col}; border-radius: 3px; }}
            """)

            song_path = self._recom.on_emotion_detected(emotion, conf)
            song_name = self._recom.current_song_name
            self._sa.record(emotion)

            # Toast notification when new emotion detected & song changes
            if song_path and emotion != self._last_emotion:
                self._last_emotion = emotion
                if hasattr(self._main, '_toast'):
                    self._main._toast.show_emotion(emotion, song_name, self._main)
                log_emotion_event(emotion, conf, song_name)

            self._song_lbl.setText(song_name)
            self._status_lbl.setText(
                "Status: Playing" if self._recom.is_playing else "Status: Paused")
            self._stats_lbl.setText(
                f"Songs played: {self._recom.songs_played_count}\n"
                f"Dominant: {self._sa.get_dominant()}")
        else:
            self._emo_lbl.setText("No Face")
            self._emo_lbl.setStyleSheet(
                "color: #6868a0; font-size: 44px; font-weight: bold; border: none;")
            self._conf_lbl.setText("Confidence: —")
            self._conf_bar.setValue(0)

    def _on_err(self, msg):
        QMessageBox.critical(self, "Camera Error", msg); self._stop_cam()

    def _pause(self):
        self._recom.play_pause()
        self._pause_btn.setText("Pause" if self._recom.is_playing else "Resume")

    def _next(self):
        self._recom.next_song()
        self._song_lbl.setText(self._recom.current_song_name)

    def _vol_change(self, val):
        self._vol_lbl.setText(f"{val}%")
        if self._recom._pygame_ok:
            try:
                import pygame; pygame.mixer.music.set_volume(val/100.0)
            except: pass

    def _snapshot(self):
        if self._last_frame is None:
            QMessageBox.warning(self, "Snapshot", "Start the camera first!"); return
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self._screenshots, f"snapshot_{ts}.jpg")
        cv2.imwrite(fname, self._last_frame)
        QMessageBox.information(self, "Snapshot Saved!", f"Saved:\n{fname}")

    # ── Helpers ───────────────────────────────────────────────────────
    def _card(self):
        f = QFrame()
        f.setStyleSheet(
            "QFrame { background: #13132a; border-radius: 14px;"
            " border: 1px solid #222245; padding: 8px; }")
        return f

    def _btn(self, text, color, slot):
        b = QPushButton(text)
        b.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; border-radius: 10px;"
            f" padding: 10px 12px; font-size: 12px; font-weight: bold; border: none; }}"
            f"QPushButton:hover {{ background: {color}cc; }}"
        )
        b.clicked.connect(slot); return b

    def _lbl(self, text="", dim=False, color=None, size=13, bold=False):
        l = QLabel(text)
        col = color or ("#6868a0" if dim else "#eeeef8")
        fw  = "bold" if bold else "normal"
        l.setStyleSheet(f"color:{col}; font-size:{size}px; font-weight:{fw}; border:none;")
        return l

# ─────────────────────────────────────────────────────────────────────────────
# Analytics Panel
# ─────────────────────────────────────────────────────────────────────────────
class AnalyticsPanel(QWidget):
    def __init__(self, session_analytics, parent=None):
        super().__init__(parent)
        self._sa = session_analytics
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(16)

        title = QPushButton("📊  Analytics")
        title.setEnabled(False)
        title.setStyleSheet(
            "QPushButton { color: #eeeef8; font-size: 22px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget(); inner.setStyleSheet("background: transparent;")
        il = QVBoxLayout(inner); il.setSpacing(16)
        il.addWidget(self._section("This Session", self._sa.get_counts()))
        il.addWidget(self._section("All-Time", get_emotion_analytics()))
        il.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def _section(self, title, counts):
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #13132a; border-radius: 14px;"
            " border: 1px solid #222245; padding: 12px; }")
        cl = QVBoxLayout(card)
        hdr = QPushButton(title)
        hdr.setEnabled(False)
        hdr.setStyleSheet(
            "QPushButton { color: #4a90d9; font-size: 14px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        cl.addWidget(hdr)
        if not counts:
            e = QPushButton("No data yet.")
            e.setEnabled(False)
            e.setStyleSheet(
                "QPushButton { color: #6868a0; background: transparent; border: none; }")
            cl.addWidget(e); return card
        mx = max(counts.values()) or 1
        for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            color = EMOTION_BAR_COLORS.get(emotion, "#888")
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,3,0,3)
            lbl = QPushButton(emotion)
            lbl.setEnabled(False)
            lbl.setFixedWidth(90)
            lbl.setStyleSheet(
                f"QPushButton {{ color: {color}; font-size: 13px; font-weight: bold;"
                " background: transparent; border: none; text-align: left; padding: 0; }")
            rl.addWidget(lbl)
            bar = QProgressBar(); bar.setRange(0, mx); bar.setValue(count)
            bar.setFixedHeight(20); bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar {{ background: #0a0a18; border-radius: 5px; border: none; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}")
            rl.addWidget(bar)
            cnt = QPushButton(str(count))
            cnt.setEnabled(False); cnt.setFixedWidth(34)
            cnt.setStyleSheet(
                "QPushButton { color: #6868a0; font-size: 11px;"
                " background: transparent; border: none; }")
            rl.addWidget(cnt); cl.addWidget(row)
        return card

# ─────────────────────────────────────────────────────────────────────────────
# Playlists Panel
# ─────────────────────────────────────────────────────────────────────────────
class PlaylistsPanel(QWidget):
    def __init__(self, recommender, parent=None):
        super().__init__(parent)
        self._recom = recommender; self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(16)
        title = QPushButton("🎵  Song Playlists")
        title.setEnabled(False)
        title.setStyleSheet(
            "QPushButton { color: #eeeef8; font-size: 22px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        lay.addWidget(title)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget(); inner.setStyleSheet("background: transparent;")
        grid  = QGridLayout(inner); grid.setSpacing(16)

        playlists = self._recom.get_all_playlists()
        emotion_data = [
            ("Happy","#00c864"),   ("Sad","#5080c8"),
            ("Angry","#e03030"),   ("Neutral","#a0a0a0"),
            ("Surprise","#f0c030"),("Fear","#b400b4"),
            ("Disgust","#008cff"), ("Contempt","#c8b400"),
            ("Excited","#ff8c00"), ("Calm","#64c8c8"),
            ("Bored","#8c8ca0"),
        ]
        emoji_map = {"Happy":"😊","Sad":"😢","Angry":"😠","Neutral":"😐",
                     "Surprise":"😲","Fear":"😨","Disgust":"🤢",
                     "Contempt":"🙄","Excited":"🤩","Calm":"😌","Bored":"🥱"}

        for i, (emotion, color) in enumerate(emotion_data):
            songs = playlists.get(emotion, [])
            card  = QFrame()
            card.setStyleSheet(
                "QFrame { background: #13132a; border-radius: 14px;"
                f"border: 1px solid {color}44; padding: 12px; }}")

            cl = QVBoxLayout(card); cl.setSpacing(6)

            hdr = QPushButton(f"{emotion} {emoji_map.get(emotion,'')}  —  {len(songs)} songs")
            hdr.setEnabled(False)
            hdr.setStyleSheet(
                f"QPushButton {{ color: {color}; font-size: 15px; font-weight: bold;"
                " background: transparent; border: none; text-align: left; padding: 0; }")
            cl.addWidget(hdr)

            div = QFrame(); div.setFixedHeight(1)
            div.setStyleSheet(f"background: {color}55; border: none;")
            cl.addWidget(div)

            if songs:
                for j, s in enumerate(songs[:5], 1):
                    sl = QPushButton(f"  {j}.  {s}")
                    sl.setEnabled(False)
                    sl.setStyleSheet(
                        "QPushButton { color: #ccccdd; font-size: 13px;"
                        " background: transparent; border: none; text-align: left; padding: 2px 0; }")
                    cl.addWidget(sl)
                if len(songs) > 5:
                    more = QPushButton(f"  + {len(songs)-5} more songs...")
                    more.setEnabled(False)
                    more.setStyleSheet(
                        "QPushButton { color: #6868a0; font-size: 11px;"
                        " background: transparent; border: none; text-align: left; padding: 0; }")
                    cl.addWidget(more)
            else:
                empty = QPushButton("  No songs added yet.")
                empty.setEnabled(False)
                empty.setStyleSheet(
                    "QPushButton { color: #6868a0; font-size: 12px;"
                    " background: transparent; border: none; text-align: left; padding: 0; }")
                cl.addWidget(empty)

            grid.addWidget(card, i//2, i%2)

        scroll.setWidget(inner); lay.addWidget(scroll)

# ─────────────────────────────────────────────────────────────────────────────
# History Panel
# ─────────────────────────────────────────────────────────────────────────────
class HistoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(16)
        hdr_row = QHBoxLayout()
        title = QPushButton("📋  Detection History")
        title.setEnabled(False)
        title.setStyleSheet(
            "QPushButton { color: #eeeef8; font-size: 22px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        hdr_row.addWidget(title); hdr_row.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet(
            "QPushButton { background: #4a90d9; color: white; border-radius: 8px;"
            " padding: 8px 20px; font-size: 13px; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #5aa0e9; }")
        refresh_btn.clicked.connect(self._refresh_table)
        hdr_row.addWidget(refresh_btn)

        clear_btn = QPushButton("Clear History")
        clear_btn.setStyleSheet(
            "QPushButton { background: #e03030; color: white; border-radius: 8px;"
            " padding: 8px 20px; font-size: 13px; font-weight: bold; border: none; }"
            "QPushButton:hover { background: #f04040; }")
        clear_btn.clicked.connect(self._clear_history)
        hdr_row.addWidget(clear_btn)
        lay.addLayout(hdr_row)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Timestamp","Emotion","Confidence","Song Played"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget { background: #13132a; color: #eeeef8;
                gridline-color: #222245; border: 1px solid #222245; border-radius: 10px; }
            QHeaderView::section { background: #1a1a35; color: #4a90d9;
                font-weight: bold; padding: 8px; border: none; }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:alternate { background: #1a1a35; }
        """)
        self._refresh_table()
        lay.addWidget(self._table)

    def _clear_history(self):
        reply = QMessageBox.question(self, "Clear History",
            "Delete all detection history?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            clear_history(); self._refresh_table()

    def _refresh_table(self):
        rows = get_recent_history(100)
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            emotion = row.get("emotion","")
            items = [
                QTableWidgetItem(row.get("timestamp","")),
                QTableWidgetItem(f"  {emotion}"),
                QTableWidgetItem(f'{row["confidence"]:.0%}' if row.get("confidence") else "—"),
                QTableWidgetItem(row.get("song_played") or "—"),
            ]
            color = EMOTION_HEX.get(emotion)
            if color: items[1].setForeground(QColor(color))
            for j, item in enumerate(items):
                self._table.setItem(i, j, item)

# ─────────────────────────────────────────────────────────────────────────────
# Add Songs Panel
# ─────────────────────────────────────────────────────────────────────────────
class AddSongsPanel(QWidget):
    def __init__(self, recommender, parent=None):
        super().__init__(parent)
        self._recom = recommender; self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(20)

        title = QPushButton("➕  Add Songs to Playlists")
        title.setEnabled(False)
        title.setStyleSheet(
            "QPushButton { color: #eeeef8; font-size: 22px; font-weight: bold;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        lay.addWidget(title)

        sub = QPushButton("Select an emotion folder, then browse and add your .mp3 / .wav files.")
        sub.setEnabled(False)
        sub.setStyleSheet(
            "QPushButton { color: #6868a0; font-size: 13px;"
            " background: transparent; border: none; text-align: left; padding: 0; }")
        lay.addWidget(sub)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget(); inner.setStyleSheet("background: transparent;")
        grid  = QGridLayout(inner); grid.setSpacing(14)

        emotion_data = [
            ("Happy","#00c864"),   ("Sad","#5080c8"),
            ("Angry","#e03030"),   ("Neutral","#a0a0a0"),
            ("Surprise","#f0c030"),("Fear","#b400b4"),
            ("Disgust","#008cff"), ("Contempt","#c8b400"),
            ("Excited","#ff8c00"), ("Calm","#64c8c8"),
            ("Bored","#8c8ca0"),
        ]
        emoji_map = {"Happy":"😊","Sad":"😢","Angry":"😠","Neutral":"😐",
                     "Surprise":"😲","Fear":"😨","Disgust":"🤢",
                     "Contempt":"🙄","Excited":"🤩","Calm":"😌","Bored":"🥱"}

        for i, (emotion, color) in enumerate(emotion_data):
            songs = self._recom.get_playlist(emotion)
            card  = QFrame()
            card.setStyleSheet(
                "QFrame { background: #13132a; border-radius: 14px;"
                f"border: 1px solid {color}44; padding: 14px; }}")
            cl = QVBoxLayout(card); cl.setSpacing(10)

            emoji = emoji_map.get(emotion, "")
            emo_btn = QPushButton(f"{emoji}  {emotion}  —  {len(songs)} songs")
            emo_btn.setEnabled(False)
            emo_btn.setStyleSheet(
                f"QPushButton {{ color: {color}; font-size: 14px; font-weight: bold;"
                " background: transparent; border: none; text-align: left; padding: 0; }")
            cl.addWidget(emo_btn)

            add_btn = QPushButton(f"Add Songs to {emotion}")
            add_btn.setStyleSheet(
                "QPushButton { background: #1e1e35; color: #eeeef8;"
                f"border: 1px solid {color}; border-radius: 8px;"
                " padding: 10px; font-size: 13px; font-weight: bold; }"
                f"QPushButton:hover {{ background: {color}33; }}")
            add_btn.clicked.connect(lambda checked, e=emotion: self._add_songs(e))
            cl.addWidget(add_btn)
            grid.addWidget(card, i//3, i%3)

        scroll.setWidget(inner)
        lay.addWidget(scroll)

        self._status = QLabel("")
        self._status.setStyleSheet(
            "color: #00c864; font-size: 13px; font-weight: bold; border: none;")
        self._status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._status); lay.addStretch()

    def _add_songs(self, emotion):
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Add Songs to {emotion}", "",
            "Audio Files (*.mp3 *.wav *.ogg *.flac)")
        if not files: return
        folder = os.path.join(SONGS_BASE_DIR, emotion.lower())
        os.makedirs(folder, exist_ok=True)
        import shutil; added = 0
        for f in files:
            dest = os.path.join(folder, os.path.basename(f))
            if not os.path.exists(dest): shutil.copy2(f, dest); added += 1
        self._recom.reload_playlists()
        self._status.setText(f"{added} song(s) added to {emotion}!")

# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Expression Based Song Recommendation System")
        self.setMinimumSize(1200, 720)

        self._recognizer        = EmotionRecognizer()
        self._recommender       = SongRecommender(cooldown_seconds=8)
        self._session_analytics = SessionAnalytics()
        self._session_id        = start_session()
        self._is_dark           = True
        self._toast             = ToastNotification(self)

        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        global _T
        _T = DARK.copy() if self._is_dark else LIGHT.copy()
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {C('BG')}; color: {C('TEXT')};
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }}
            QFrame {{ border: none; }} QLabel {{ border: none; }}
            QPushButton {{ border: none; }}
        """)

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._topbar())
        body = QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)
        body.addWidget(self._sidebar())
        body.addWidget(self._content_area())
        root.addLayout(body)

    def _topbar(self):
        bar = QFrame(); bar.setFixedHeight(56)
        bar.setStyleSheet(
            f"background: {C('SIDEBAR')}; border-bottom: 1px solid {C('BORDER')};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(20,0,20,0)
        logo = QLabel("Face Expression Based Song Recommendation System")
        logo.setStyleSheet(
            f"color: {C('TEXT')}; font-size: 14px; font-weight: bold;")
        lay.addWidget(logo); lay.addStretch()
        self._theme_btn = QPushButton("Light Mode")
        self._theme_btn.setStyleSheet(
            f"QPushButton {{ background: {C('CARD2')}; color: {C('TEXT')};"
            f" border-radius: 8px; padding: 6px 16px; font-size: 12px;"
            f" font-weight: bold; border: 1px solid {C('BORDER')}; }}"
            f"QPushButton:hover {{ background: {C('NAV_HOVER')}; }}")
        self._theme_btn.clicked.connect(self._toggle_theme)
        lay.addWidget(self._theme_btn)
        return bar

    def _sidebar(self):
        side = QFrame(); side.setFixedWidth(210)
        side.setStyleSheet(
            f"background: {C('SIDEBAR')}; border-right: 1px solid {C('BORDER')};")
        lay = QVBoxLayout(side); lay.setContentsMargins(0,16,0,16); lay.setSpacing(4)

        nav_items = [
            ("🏠", "Home"),
            ("📊", "Analytics"),
            ("🥧", "Emotion Chart"),
            ("🎵", "Playlists"),
            ("📋", "History"),
            ("➕", "Add Songs"),
        ]
        self._nav_btns = []
        for icon, label in nav_items:
            btn = SidebarBtn(icon, label)
            btn.clicked.connect(lambda _, l=label: self._nav_to(l))
            self._nav_btns.append(btn); lay.addWidget(btn)

        lay.addStretch()
        ver = QLabel("© 2026 Expression Music Explorer")
        ver.setWordWrap(True)
        ver.setStyleSheet(f"color: {C('DIM')}; font-size: 10px; padding: 8px 16px;")
        lay.addWidget(ver)
        self._nav_btns[0].setChecked(True)
        return side

    def _content_area(self):
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {C('BG')};")
        self._home_panel      = HomePanel(
            self._recognizer, self._recommender, self._session_analytics, self)
        self._analytics_panel = AnalyticsPanel(self._session_analytics)
        self._chart_panel     = EmotionChartPanel(self._session_analytics)
        self._playlists_panel = PlaylistsPanel(self._recommender)
        self._history_panel   = HistoryPanel()
        self._addsongs_panel  = AddSongsPanel(self._recommender)
        for p in [self._home_panel, self._analytics_panel, self._chart_panel,
                  self._playlists_panel, self._history_panel, self._addsongs_panel]:
            self._stack.addWidget(p)
        return self._stack

    def _nav_to(self, label):
        idx_map = {"Home":0,"Analytics":1,"Emotion Chart":2,
                   "Playlists":3,"History":4,"Add Songs":5}
        idx = idx_map.get(label, 0)
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == idx)

    def _toggle_theme(self):
        current_idx = self._stack.currentIndex()
        self._home_panel.stop_camera()
        self._is_dark = not self._is_dark
        global _T
        _T = DARK.copy() if self._is_dark else LIGHT.copy()
        old = self.centralWidget()
        self._apply_style()
        self._build_ui()
        old.deleteLater()
        self._stack.setCurrentIndex(current_idx)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == current_idx)
        self._theme_btn.setText("Light Mode" if self._is_dark else "Dark Mode")

    def closeEvent(self, event):
        self._home_panel.stop_camera()
        end_session(self._session_id, self._recommender.songs_played_count)
        event.accept()
