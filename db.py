import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("events.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            dong TEXT NOT NULL,
            cctv_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_id TEXT NOT NULL
        )
        """)
        conn.commit()

def insert_event(
    lat: float,
    lon: float,
    dong: str,
    cctv_id: str,
    event_type: str,
    confidence: float,
    source_id: str
):
    ts = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO events (ts, lat, lon, dong, cctv_id, event_type, confidence, source_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, lat, lon, dong, cctv_id, event_type, confidence, source_id)
        )
        conn.commit()

def fetch_events(limit: int = 2000):
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT ts, lat, lon, dong, cctv_id, event_type, confidence, source_id
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        return cur.fetchall()
