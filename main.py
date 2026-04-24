"""
Entry Point: main.py
Face Expression Based Song Recommendation System
"""

import sys, os, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.app_logger import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

from modules.database import initialize_db
initialize_db()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

def main():
    logger.info("=" * 60)
    logger.info("  Face Expression Song Recommender  —  Starting Up")
    logger.info("=" * 60)

    # Fix Windows DPI
    import ctypes
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_Use96Dpi, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Face Expression Song Recommender")
    app.setApplicationVersion("3.0.0")
    app.setFont(QFont("Segoe UI", 10))

    # Splash Screen
    from modules.gui import SplashScreen, MainWindow
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    time.sleep(2.5)

    window = MainWindow()
    splash.finish(window)
    window.show()

    logger.info("GUI launched successfully.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
