import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from modules.database import get_emotion_analytics

logger = logging.getLogger(__name__)

EMOTION_BAR_COLORS = {
    "Happy":"#00c864",   "Sad":"#5080c8",    "Angry":"#e03030",
    "Neutral":"#a0a0a0", "Surprise":"#f0c030","Fear":"#b400b4",
    "Disgust":"#008cff", "Contempt":"#c8b400","Excited":"#ff8c00",
    "Calm":"#64c8c8",    "Bored":"#8c8ca0",
}

class SessionAnalytics:
    def __init__(self):
        self._counts = defaultdict(int)
        self._total  = 0

    def record(self, emotion):
        self._counts[emotion] += 1
        self._total += 1

    def get_counts(self):
        return dict(self._counts)

    def get_dominant(self):
        if not self._counts: return "N/A"
        return max(self._counts, key=self._counts.get)

    def get_percentage(self, emotion):
        if self._total == 0: return 0.0
        return round(100.0 * self._counts.get(emotion, 0) / self._total, 1)

    def get_summary_lines(self):
        if not self._counts: return ["No data yet."]
        lines = [f"{'Emotion':<12} {'Count':>6}  {'Share':>7}", "─" * 28]
        for e, c in sorted(self._counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{e:<12} {c:>6}  {self.get_percentage(e):>6.1f}%")
        lines.extend(["─" * 28, f"{'Total':<12} {self._total:>6}"])
        return lines

    def reset(self):
        self._counts.clear()
        self._total = 0

def build_bar_data(counts):
    if not counts: return []
    return [(e, c, EMOTION_BAR_COLORS.get(e, "#888"))
            for e, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
