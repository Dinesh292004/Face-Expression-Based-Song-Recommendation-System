import logging, os
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE  = os.path.join(LOGS_DIR, "app.log")

def setup_logging(level=logging.INFO):
    os.makedirs(LOGS_DIR, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(fh)
