"""
Fetch CMS Medicare Geographic Variation county-level spending data.
Uses the CMS data.cms.gov Socrata API — no key required.
Dataset: Medicare Geographic Variation by National/State/County
"""

import logging
import requests
import pandas as pd
from src.database import get_connection

logger = logging.getLogger(__name__)

# CMS new data API endpoint (dataset UUID confirmed from data.cms.gov catalog)
CMS_API_URL = "https://data.cms.gov/data-api/v1/dataset/6219697b-8f6c-4164-bed4-cd9317c58ebc/data"


def fetch_cms_spending() -> pd.DataFrame:
    """
    Download CMS county-level Medicare spending. Returns DataFrame with
    fips, county_name, state, medicare_spending_per_beneficiary.
    """
    logger.info("Fetching CMS Medicare spending data (new CMS data API)...")

    all_records = []
    offset = 0
    size = 5000

    while True:
        params = {
            "filter[BENE_GEO_LVL]": "County",
            "filter[BENE_AGE_LVL]": "All",
            "size": size,
            "offset": offset,
        }
        try:
            resp = requests.get(CMS_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("data", data) if isinstance(data, dict) else data
        except Exception as exc:
            logger.error("CMS API failed at offset %d: %s", offset, exc)
            break

        if not batch:
            break
        all_records.extend(batch)
        logger.info("  CMS: fetched %d records (total: %d)", len(batch), len(all_records))
        if len(batch) < size:
            break
        offset += size

    if not all_records:
        logger.error("No CMS spending data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Column names are uppercase in the new API
    df.columns = df.columns.str.upper()
    logger.info("CMS columns sample: %s", list(df.columns)[:10])

    # Use most recent year available
    if "YEAR" in df.columns:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        latest_year = df["YEAR"].max()
        df = df[df["YEAR"] == latest_year].copy()
        logger.info("Using CMS year: %s", latest_year)

    # Known column names from API inspection
    # FIPS = BENE_GEO_CD, Name = BENE_GEO_DESC, Spending = TOT_MDCR_STDZD_PYMT_PC
    result = pd.DataFrame()
    result["fips"] = df["BENE_GEO_CD"].astype(str).str.zfill(5)
    result["county_name"] = df["BENE_GEO_DESC"].astype(str) if "BENE_GEO_DESC" in df.columns else "Unknown"
    result["state"] = ""
    result["medicare_spending_per_beneficiary"] = pd.to_numeric(
        df.get("TOT_MDCR_STDZD_PYMT_PC", df.get("TOT_MDCR_PYMT_PC", pd.Series())),
        errors="coerce"
    )

    result = result[result["fips"].str.len() == 5]
    result = result.dropna(subset=["medicare_spending_per_beneficiary"])
    result = result[result["medicare_spending_per_beneficiary"] > 0]

    logger.info("CMS spending: %d counties", len(result))
    return result


def _fetch_fallback() -> list:
    """Try alternate CMS dataset identifiers."""
    fallback_ids = ["3tdk-5bfr", "b5th-rtmm", "nm9v-yq7d"]
    for dataset_id in fallback_ids[1:]:
        url = f"https://data.cms.gov/resource/{dataset_id}.json"
        try:
            resp = requests.get(url, params={"$limit": 5000, "$where": "bene_geo_lvl='County'"}, timeout=30)
            if resp.ok:
                data = resp.json()
                if data:
                    logger.info("Fallback CMS dataset %s returned %d records", dataset_id, len(data))
                    return data
        except Exception:
            continue
    return []


def store_cms_spending(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ["fips", "county_name", "state", "medicare_spending_per_beneficiary"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    with get_connection() as conn:
        conn.execute("DELETE FROM cms_spending")
        df[cols].to_sql("cms_spending", conn, if_exists="append", index=False)
    logger.info("Stored %d rows in cms_spending", len(df))
