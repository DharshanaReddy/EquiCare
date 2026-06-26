"""
Composite Health Equity Gap Score calculation and quartile segmentation.

Score = diabetes(0.30) + hypertension(0.25) + poverty(0.25) + uninsured(0.20)
All components MinMax-normalized to [0, 1] before weighting.
Higher score = worse health equity.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.database import get_connection

logger = logging.getLogger(__name__)

SCORE_COMPONENTS = {
    "diabetes_rate":    0.30,
    "hypertension_rate": 0.25,
    "poverty_rate":     0.25,
    "uninsured_rate":   0.20,
}


def compute_equity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add equity_gap_score, equity_quartile, and primary_driver columns to df.
    Returns the augmented DataFrame.
    """
    df = df.copy()
    components = list(SCORE_COMPONENTS.keys())
    available = [c for c in components if c in df.columns]

    if len(available) < 2:
        logger.error("Not enough component columns to compute equity score: %s", available)
        return df

    # Drop rows missing any component
    score_df = df.dropna(subset=available).copy()

    # MinMax normalize each component
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(score_df[available])
    norm_df = pd.DataFrame(normalized, columns=[f"{c}_norm" for c in available],
                           index=score_df.index)

    # Weighted sum
    weights = [SCORE_COMPONENTS[c] for c in available]
    # Renormalize weights if not all components present
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    score_df["equity_gap_score"] = sum(
        norm_df[f"{c}_norm"] * w for c, w in zip(available, weights)
    ).round(4)

    # Quartile segmentation (Q1=best equity, Q4=worst)
    score_df["equity_quartile"] = pd.qcut(
        score_df["equity_gap_score"], q=4, labels=[1, 2, 3, 4]
    ).astype(int)

    # Primary driver: which component is furthest above its national mean (normalized)
    nat_means = norm_df.mean()
    def _primary_driver(row_idx):
        excesses = {c: norm_df.loc[row_idx, f"{c}_norm"] - nat_means[f"{c}_norm"]
                    for c in available}
        return max(excesses, key=excesses.get).replace("_rate", "").replace("_", " ").title()

    score_df["primary_driver"] = [_primary_driver(i) for i in score_df.index]

    # Merge back into original df
    df = df.drop(columns=["equity_gap_score", "equity_quartile", "primary_driver"],
                 errors="ignore")
    df = df.merge(
        score_df[["fips", "equity_gap_score", "equity_quartile", "primary_driver"]],
        on="fips", how="left"
    )

    logger.info("Equity scores computed for %d counties", score_df["equity_gap_score"].notna().sum())
    return df


def quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean statistics per equity quartile."""
    cols = ["equity_quartile", "medicare_spending_per_beneficiary",
            "poverty_rate", "diabetes_rate", "uninsured_rate"]
    available = [c for c in cols if c in df.columns]
    summary = (
        df.dropna(subset=["equity_quartile"])[available]
        .groupby("equity_quartile")
        .agg(
            count=("equity_quartile", "count"),
            mean_spending=("medicare_spending_per_beneficiary", "mean"),
            median_spending=("medicare_spending_per_beneficiary", "median"),
            mean_poverty=("poverty_rate", "mean"),
            mean_diabetes=("diabetes_rate", "mean"),
        )
        .round(2)
        .reset_index()
    )

    print("\n=== QUARTILE SUMMARY ===")
    print(f"{'Quartile':<12} {'Count':>8} {'Mean Spending':>16} {'Median Spending':>18} {'Mean Poverty%':>15} {'Mean Diabetes%':>16}")
    print("-" * 90)
    for _, row in summary.iterrows():
        label = f"Q{int(row['equity_quartile'])} ({'best' if row['equity_quartile']==1 else 'worst' if row['equity_quartile']==4 else ''})"
        print(f"{label:<12} {int(row['count']):>8} ${row['mean_spending']:>14,.2f} ${row['median_spending']:>16,.2f} {row['mean_poverty']:>14.2f}% {row['mean_diabetes']:>15.2f}%")

    return summary


def state_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equity gap by state, count Q4 counties per state."""
    if "state" not in df.columns:
        return pd.DataFrame()

    q4_counts = (
        df[df["equity_quartile"] == 4]
        .groupby("state")
        .size()
        .rename("q4_county_count")
        .reset_index()
    )
    state_agg = (
        df.groupby("state")
        .agg(
            mean_equity_gap=("equity_gap_score", "mean"),
            mean_spending=("medicare_spending_per_beneficiary", "mean"),
            county_count=("fips", "count"),
        )
        .round(3)
        .reset_index()
    )
    state_summary = state_agg.merge(q4_counts, on="state", how="left")
    state_summary["q4_county_count"] = state_summary["q4_county_count"].fillna(0).astype(int)
    state_summary = state_summary.sort_values("q4_county_count", ascending=False)

    print("\n=== TOP 10 STATES BY Q4 COUNTY CONCENTRATION ===")
    print(f"{'State':<25} {'Q4 Counties':>12} {'Total Counties':>15} {'Mean Gap':>10} {'Mean Spending':>15}")
    print("-" * 80)
    for _, row in state_summary.head(10).iterrows():
        print(f"{row['state']:<25} {row['q4_county_count']:>12} {row['county_count']:>15} "
              f"{row['mean_equity_gap']:>10.3f} ${row['mean_spending']:>13,.2f}")

    return state_summary


def persist_master_county(df: pd.DataFrame) -> None:
    """Write the fully enriched master_county table to SQLite."""
    cols = [
        "fips", "county_name", "state", "median_income", "poverty_rate",
        "uninsured_rate", "pct_black", "pct_hispanic", "diabetes_rate",
        "hypertension_rate", "mental_health_poor_days", "preventive_screening_rate",
        "medicare_spending_per_beneficiary", "total_population",
        "equity_gap_score", "equity_quartile", "primary_driver",
    ]
    export = df[[c for c in cols if c in df.columns]].copy()
    with get_connection() as conn:
        conn.execute("DELETE FROM master_county")
        export.to_sql("master_county", conn, if_exists="append", index=False)
    logger.info("master_county table updated: %d rows", len(export))
