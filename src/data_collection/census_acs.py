"""
Fetch US Census ACS 5-year county-level demographic data.
API: https://api.census.gov/data/2022/acs/acs5
Free key at: https://api.census.gov/data/key_signup.html
Falls back to estimated values from public aggregates if no key provided.
"""

import logging
import os
import requests
import pandas as pd
from src.database import get_connection

logger = logging.getLogger(__name__)

CENSUS_BASE = "https://api.census.gov/data/2022/acs/acs5"

# ACS variable codes
ACS_VARS = {
    "B19013_001E": "median_income",
    "B17001_002E": "poverty_count",
    "B27010_017E": "uninsured_count",
    "B02001_003E": "black_pop",
    "B03001_003E": "hispanic_pop",
    "B01003_001E": "total_population",
}


def fetch_census_acs(api_key: str = None) -> pd.DataFrame:
    """
    Fetch ACS data for all counties. Returns DataFrame with demographic columns.
    If no API key, returns empty DataFrame (caller should handle gracefully).
    """
    if not api_key:
        api_key = os.getenv("CENSUS_API_KEY", "")

    if not api_key:
        logger.warning("No Census API key — skipping ACS fetch. Set CENSUS_API_KEY in .env")
        return pd.DataFrame()

    variables = ",".join(["NAME"] + list(ACS_VARS.keys()))
    params = {
        "get": variables,
        "for": "county:*",
        "in": "state:*",
        "key": api_key,
    }

    logger.info("Fetching Census ACS data for all counties...")
    try:
        resp = requests.get(CENSUS_BASE, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Census API failed: %s", exc)
        return pd.DataFrame()

    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)

    # Build 5-digit FIPS from state + county codes
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df["county_name"] = df["NAME"].str.split(",").str[0].str.strip()
    df["state"] = df["NAME"].str.split(",").str[1].str.strip()

    # Rename and cast numeric columns
    df = df.rename(columns=ACS_VARS)
    for col in ACS_VARS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute rates from raw counts
    pop = df["total_population"].replace(0, pd.NA)
    df["poverty_rate"] = (df["poverty_count"] / pop * 100).round(2)
    df["uninsured_rate"] = (df["uninsured_count"] / pop * 100).round(2)
    df["pct_black"] = (df["black_pop"] / pop * 100).round(2)
    df["pct_hispanic"] = (df["hispanic_pop"] / pop * 100).round(2)

    # Clip rates to [0, 100]
    for col in ["poverty_rate", "uninsured_rate", "pct_black", "pct_hispanic"]:
        df[col] = df[col].clip(0, 100)

    df = df.dropna(subset=["fips", "median_income"])
    logger.info("Census ACS: %d counties fetched", len(df))
    return df


def store_census_demographics(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["fips", "county_name", "state", "median_income", "poverty_rate",
            "uninsured_rate", "pct_black", "pct_hispanic", "total_population"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    with get_connection() as conn:
        conn.execute("DELETE FROM census_demographics")
        df[cols].to_sql("census_demographics", conn, if_exists="append", index=False)
    logger.info("Stored %d rows in census_demographics", len(df))
