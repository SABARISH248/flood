import os
import sqlite3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.utils import logger, load_config


def get_db_connection(db_path: str = "data/field_records/feedback.db") -> sqlite3.Connection:
    """Connect to SQLite field feedback database and initialize tables if missing."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            date TEXT NOT NULL,
            depth_m REAL,
            description TEXT,
            reporter_name TEXT,
            photo_path TEXT,
            submitted_at TEXT NOT NULL,
            expert_verified INTEGER DEFAULT 0,
            verification_notes TEXT
        );
    """)
    conn.commit()
    return conn


def submit_field_report(
    lat: float,
    lon: float,
    date_str: str,
    depth_m: float,
    description: str,
    reporter_name: str = "Anonymous",
    photo_path: Optional[str] = None,
    db_path: str = "data/field_records/feedback.db"
) -> int:
    """Insert a new field feedback record into SQLite database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO field_feedback (latitude, longitude, date, depth_m, description, reporter_name, photo_path, submitted_at, expert_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (lat, lon, date_str, depth_m, description, reporter_name, photo_path or "", now_str))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    logger.info(f"Submitted field report ID {record_id} at ({lat}, {lon})")
    return record_id


def verify_field_report(
    record_id: int,
    verified: bool = True,
    notes: str = "Verified by domain expert",
    db_path: str = "data/field_records/feedback.db"
) -> bool:
    """Update expert verification status of a field feedback record."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE field_feedback
        SET expert_verified = ?, verification_notes = ?
        WHERE id = ?
    """, (1 if verified else 0, notes, record_id))
    conn.commit()
    conn.close()
    return True


def get_all_field_reports(db_path: str = "data/field_records/feedback.db") -> pd.DataFrame:
    """Fetch all field reports as pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM field_feedback ORDER BY submitted_at DESC", conn)
    conn.close()
    return df


def get_field_reports_gdf(db_path: str = "data/field_records/feedback.db") -> gpd.GeoDataFrame:
    """Fetch all field reports as GeoPandas GeoDataFrame."""
    df = get_all_field_reports(db_path)
    if df.empty:
        return gpd.GeoDataFrame(columns=["id", "geometry"], crs="EPSG:4326")

    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf
