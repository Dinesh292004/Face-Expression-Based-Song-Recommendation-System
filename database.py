import sqlite3, os, logging
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "app_data.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS emotion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, emotion TEXT NOT NULL,
        confidence REAL, song_played TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS session_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_start TEXT NOT NULL, session_end TEXT,
        dominant_emotion TEXT, total_songs_played INTEGER DEFAULT 0)""")
    conn.commit(); conn.close()
    logger.info("Database initialized at: %s", DB_PATH)

def log_emotion_event(emotion, confidence, song_played=None):
    try:
        conn = get_connection(); c = conn.cursor()
        c.execute("INSERT INTO emotion_logs (timestamp,emotion,confidence,song_played) VALUES (?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, confidence, song_played))
        conn.commit(); conn.close()
    except Exception as e:
        logger.error("Failed to log emotion: %s", e)

def get_emotion_analytics():
    try:
        conn = get_connection(); c = conn.cursor()
        c.execute("SELECT emotion, COUNT(*) as count FROM emotion_logs GROUP BY emotion ORDER BY count DESC")
        rows = c.fetchall(); conn.close()
        return {row["emotion"]: row["count"] for row in rows}
    except: return {}

def get_most_detected_emotion():
    a = get_emotion_analytics()
    return max(a, key=a.get) if a else "N/A"

def get_recent_history(limit=50):
    try:
        conn = get_connection(); c = conn.cursor()
        c.execute("SELECT timestamp,emotion,confidence,song_played FROM emotion_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall(); conn.close()
        return [dict(row) for row in rows]
    except: return []

def clear_history():
    try:
        conn = get_connection(); c = conn.cursor()
        c.execute("DELETE FROM emotion_logs")
        conn.commit(); conn.close()
        logger.info("History cleared.")
        return True
    except Exception as e:
        logger.error("Failed to clear history: %s", e)
        return False

def start_session():
    try:
        conn = get_connection(); c = conn.cursor()
        c.execute("INSERT INTO session_summary (session_start) VALUES (?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
        sid = c.lastrowid; conn.commit(); conn.close()
        return sid
    except: return None

def end_session(session_id, total_songs):
    try:
        dominant = get_most_detected_emotion()
        conn = get_connection(); c = conn.cursor()
        c.execute("UPDATE session_summary SET session_end=?,dominant_emotion=?,total_songs_played=? WHERE id=?",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dominant, total_songs, session_id))
        conn.commit(); conn.close()
    except Exception as e:
        logger.error("Failed to end session: %s", e)
