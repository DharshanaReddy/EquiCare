"""Export clean Tableau-ready CSV and supporting CSVs."""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

TABLEAU_COLUMNS = {
    "county_name":                      "County_Name",
    "state":                            "State",
    "fips":                             "FIPS",
    "median_income":                    "Median_Income",
    "poverty_rate":                     "Poverty_Rate_Pct",
    "uninsured_rate":                   "Uninsured_Rate_Pct",
    "pct_black":                        "Pct_Black",
    "pct_hispanic":                     "Pct_Hispanic",
    "diabetes_rate":                    "Diabetes_Rate",
    "hypertension_rate":                "Hypertension_Rate",
    "mental_health_poor_days":          "Mental_Health_Poor_Days",
    "preventive_screening_rate":        "Preventive_Screening_Rate",
    "medicare_spending_per_beneficiary":"Medicare_Spending_Per_Beneficiary",
    "equity_gap_score":                 "Equity_Gap_Score",
    "equity_quartile":                  "Equity_Quartile",
    "primary_driver":                   "Primary_Driver",
}


def export_tableau_csv(df: pd.DataFrame) -> int:
    """
    Export Tableau-ready CSV. Drops rows missing key columns.
    Returns count of exported rows.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Round percentage columns
    pct_cols = ["poverty_rate", "uninsured_rate", "pct_black", "pct_hispanic",
                "diabetes_rate", "hypertension_rate", "preventive_screening_rate"]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].round(2)

    if "equity_gap_score" in df.columns:
        df["equity_gap_score"] = df["equity_gap_score"].round(4)

    # Drop rows missing critical columns
    required = ["medicare_spending_per_beneficiary", "equity_gap_score"]
    clean = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    # Rename to Tableau-friendly column names
    available_cols = {k: v for k, v in TABLEAU_COLUMNS.items() if k in clean.columns}
    tableau_df = clean[list(available_cols.keys())].rename(columns=available_cols)

    path = OUTPUT_DIR / "equicare_tableau.csv"
    tableau_df.to_csv(path, index=False)
    logger.info("Tableau CSV exported: %d rows -> %s", len(tableau_df), path)
    return len(tableau_df)


def export_supporting_csvs(df: pd.DataFrame, state_summary: pd.DataFrame) -> None:
    """Export worst_counties.csv and state_summary.csv."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Top 20 worst counties
    worst = (
        df.dropna(subset=["equity_gap_score"])
        .nlargest(20, "equity_gap_score")[
            ["county_name", "state", "fips", "equity_gap_score",
             "medicare_spending_per_beneficiary", "poverty_rate",
             "diabetes_rate", "primary_driver"]
        ]
        .round(3)
    )
    worst.to_csv(OUTPUT_DIR / "worst_counties.csv", index=False)
    logger.info("worst_counties.csv exported (%d rows)", len(worst))

    if not state_summary.empty:
        state_summary.to_csv(OUTPUT_DIR / "state_summary.csv", index=False)
        logger.info("state_summary.csv exported (%d rows)", len(state_summary))

    # Full scored dataset
    df.round(3).to_csv(OUTPUT_DIR / "all_counties.csv", index=False)
    logger.info("all_counties.csv exported (%d rows)", len(df))
