"""SQLite database setup and connection helpers for EquiCare."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "equicare.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cdc_health_outcomes (
                fips            TEXT PRIMARY KEY,
                county_name     TEXT,
                state           TEXT,
                diabetes_rate       REAL,
                hypertension_rate   REAL,
                mental_health_poor_days REAL,
                preventive_screening_rate REAL,
                obesity_rate    REAL
            );

            CREATE TABLE IF NOT EXISTS cms_spending (
                fips                            TEXT PRIMARY KEY,
                county_name                     TEXT,
                state                           TEXT,
                medicare_spending_per_beneficiary REAL
            );

            CREATE TABLE IF NOT EXISTS census_demographics (
                fips            TEXT PRIMARY KEY,
                county_name     TEXT,
                state           TEXT,
                median_income   REAL,
                poverty_rate    REAL,
                uninsured_rate  REAL,
                pct_black       REAL,
                pct_hispanic    REAL,
                total_population REAL
            );

            CREATE TABLE IF NOT EXISTS master_county (
                fips            TEXT PRIMARY KEY,
                county_name     TEXT,
                state           TEXT,
                median_income   REAL,
                poverty_rate    REAL,
                uninsured_rate  REAL,
                pct_black       REAL,
                pct_hispanic    REAL,
                diabetes_rate   REAL,
                hypertension_rate REAL,
                mental_health_poor_days REAL,
                preventive_screening_rate REAL,
                medicare_spending_per_beneficiary REAL,
                total_population REAL,
                equity_gap_score REAL,
                equity_quartile  INTEGER,
                primary_driver   TEXT
            );
        """)
    print(f"[DB] Initialized at {DB_PATH}")
