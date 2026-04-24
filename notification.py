"""
Module: notification.py
Toast notification popup that appears when emotion is detected.
Auto-dismisses after 3 seconds.
"""

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPainterPath

EMOTION_COLORS = {
    "Happy":"#00c864",   "Sad":"#5080c8",    "Angry":"#e03030",
    "Neutral":"#a0a0a0", "Surprise":"#f0c030","Fear":"#b400b4",
    "Disgust":"#008cff", "Contempt":"#c8b400","Excited":"#ff8c00",
    "Calm":"#64c8c8",    "Bored":"#8c8ca0",
}

EMOTION_EMOJI = {
    "Happy":"😊",   "Sad":"😢",    "Angry":"😠",
    "Neutral":"😐", "Surprise":"😲","Fear":"😨",
    "Disgust":"🤢", "Contempt":"🙄","Excited":"🤩",
    "Calm":"😌",    "Bored":"🥱",
}


class ToastNotification(QWidget):
    """
    Floating toast notification — appears bottom-right,
    auto-dismisses after 3 seconds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._opacity = 0.0
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(300, 80)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(16, 12, 16, 12)
        self._layout.setSpacing(12)

        self._icon_lbl = QLabel()
        self._icon_lbl.setFixedSize(40, 40)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        self._icon_lbl.setStyleSheet("background: transparent; border: none; font-size: 22px;")
        self._layout.addWidget(self._icon_lbl)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        self._title_lbl = QLabel("Emotion Detected!")
        self._title_lbl.setStyleSheet(
            "color: white; font-size: 13px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(self._title_lbl)

        self._body_lbl = QLabel("")
        self._body_lbl.setStyleSheet(
            "color: rgba(255,255,255,200); font-size: 11px;"
            " background: transparent; border: none;"
        )
        text_col.addWidget(self._body_lbl)
        self._layout.addLayout(text_col)

        self._color = QColor("#4a90d9")

        self._fade_in  = QPropertyAnimation(self, b"opacity")
        self._fade_in.setDuration(300)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)

        self._fade_out = QPropertyAnimation(self, b"opacity")
        self._fade_out.setDuration(400)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out.finished.connect(self.hide)

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out.start)

    # ── Property for animation ─────────────────────────────────────────
    def get_opacity(self):
        return self._opacity

    def set_opacity(self, val):
        self._opacity = val
        self.setWindowOpacity(val)

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    # ── Public API ─────────────────────────────────────────────────────
    def show_emotion(self, emotion: str, song_name: str, parent_widget=None):
        """Show a toast for the detected emotion."""
        color_hex = EMOTION_COLORS.get(emotion, "#4a90d9")
        emoji     = EMOTION_EMOJI.get(emotion, "🎭")
        self._color = QColor(color_hex)

        self._icon_lbl.setText(emoji)
        self._title_lbl.setText(f"Emotion: {emotion}")
        self._body_lbl.setText(f"Now Playing: {song_name[:30]}{'...' if len(song_name) > 30 else ''}")

        # Position: bottom-right of parent or screen
        if parent_widget:
            pr = parent_widget.geometry()
            x  = pr.x() + pr.width()  - self.width()  - 20
            y  = pr.y() + pr.height() - self.height()  - 60
        else:
            x, y = 1200, 700

        self.move(x, y)
        self.show()
        self._fade_in.start()
        self._timer.start(3000)   # Auto-dismiss after 3 seconds

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        # Background
        bg = QColor(20, 20, 35, 230)
        painter.fillPath(path, bg)

        # Left accent bar
        accent_path = QPainterPath()
        accent_path.addRoundedRect(0, 0, 5, self.height(), 3, 3)
        painter.fillPath(accent_path, self._color)

        painter.end()
