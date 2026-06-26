"""
Fetch CDC PLACES county-level health outcome data.
API endpoint: https://data.cdc.gov/resource/swc5-untb.json
No API key required. Uses Socrata Open Data API with $limit parameter.
"""

import logging
import requests
import pandas as pd
from src.database import get_connection

logger = logging.getLogger(__name__)

CDC_URL = "https://data.cdc.gov/resource/swc5-untb.json"

# Map CDC measure IDs to our column names
MEASURE_MAP = {
    "DIABETES":     "diabetes_rate",
    "BPHIGH":       "hypertension_rate",
    "MHLTH":        "mental_health_poor_days",
    "COLON_SCREEN": "preventive_screening_rate",
    "OBESITY":      "obesity_rate",
}


def fetch_cdc_places() -> pd.DataFrame:
    """
    Download CDC PLACES data. Returns a DataFrame with one row per county
    containing all health outcome measures we need.
    """
    logger.info("Fetching CDC PLACES data...")
    all_records = []
    offset = 0
    limit = 50000

    while True:
        params = {
            "$limit": limit,
            "$offset": offset,
            "$select": "locationid,locationname,stateabbr,measureid,data_value",
        }
        try:
            resp = requests.get(CDC_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # API returns {value: [...], Count: N} or plain list
            batch = data.get("value", data) if isinstance(data, dict) else data
        except Exception as exc:
            logger.error("CDC PLACES fetch failed at offset %d: %s", offset, exc)
            break

        if not batch:
            break
        all_records.extend(batch)
        logger.info("  Fetched %d records (total so far: %d)", len(batch), len(all_records))
        if len(batch) < limit:
            break
        offset += limit

    if not all_records:
        logger.warning("No CDC PLACES records retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.columns = df.columns.str.lower()

    # Keep only county-level rows (5-digit locationid) and target measures
    df = df[df["locationid"].astype(str).str.len() == 5].copy()
    df = df[df["measureid"].isin(MEASURE_MAP.keys())].copy()
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    # Pivot: one row per county, one column per measure
    pivoted = df.pivot_table(
        index=["locationid", "locationname", "stateabbr"],
        columns="measureid",
        values="data_value",
        aggfunc="mean",
    ).reset_index()

    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={
        "locationid": "fips",
        "locationname": "county_name",
        "stateabbr": "state",
        **MEASURE_MAP,
    })

    # Ensure FIPS is zero-padded to 5 digits
    pivoted["fips"] = pivoted["fips"].astype(str).str.zfill(5)

    # Rename any remaining raw measure columns
    for raw, clean in MEASURE_MAP.items():
        if raw in pivoted.columns:
            pivoted = pivoted.rename(columns={raw: clean})

    logger.info("CDC PLACES: %d counties processed", len(pivoted))
    return pivoted


def store_cdc_places(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["fips", "county_name", "state", "diabetes_rate", "hypertension_rate",
            "mental_health_poor_days", "preventive_screening_rate", "obesity_rate"]
    # Add missing columns as NaN
    for c in cols:
        if c not in df.columns:
            df[c] = None

    with get_connection() as conn:
        conn.execute("DELETE FROM cdc_health_outcomes")
        df[cols].to_sql("cdc_health_outcomes", conn, if_exists="append", index=False)
    logger.info("Stored %d rows in cdc_health_outcomes", len(df))
