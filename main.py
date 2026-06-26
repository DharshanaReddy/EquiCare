"""
EquiCare — Health Equity Gap Analyzer
Entry point for the full data pipeline.

Usage:
    python main.py [--step collect|analyze|visualize|report|all]

Steps:
    collect   — Pull CDC PLACES, CMS Medicare, Census ACS data into SQLite
    analyze   — Run statistical analysis and compute equity scores
    visualize — Generate 4 matplotlib/seaborn charts
    report    — Export Tableau CSV and generate findings.md
    all       — Run every step in sequence (default)
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("equicare.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("equicare")


def step_collect() -> None:
    from src.database import initialize_database
    from src.data_collection.cdc_places import fetch_cdc_places, store_cdc_places
    from src.data_collection.cms_spending import fetch_cms_spending, store_cms_spending
    from src.data_collection.census_acs import fetch_census_acs, store_census_demographics

    logger.info("=== Step 1: Data Collection ===")
    initialize_database()

    cdc_df = fetch_cdc_places()
    store_cdc_places(cdc_df)

    cms_df = fetch_cms_spending()
    store_cms_spending(cms_df)

    census_api_key = os.getenv("CENSUS_API_KEY", "")
    census_df = fetch_census_acs(census_api_key)
    store_census_demographics(census_df)


def step_analyze() -> dict:
    from src.database import get_connection
    from src.analysis.statistics import pearson_correlation_matrix, ols_regression, ttest_q4_vs_q1
    from src.analysis.equity_score import (compute_equity_scores, quartile_summary,
                                            state_level_summary, persist_master_county)

    logger.info("=== Step 2: Statistical Analysis ===")

    # Build master DataFrame by joining all three tables on FIPS
    with get_connection() as conn:
        cdc = pd.read_sql("SELECT * FROM cdc_health_outcomes", conn)
        cms = pd.read_sql("SELECT * FROM cms_spending", conn)
        census = pd.read_sql("SELECT * FROM census_demographics", conn)

    if cdc.empty:
        logger.error("CDC data missing — run collect step first.")
        return {}
    if cms.empty:
        logger.warning("CMS spending data missing — analysis will proceed without spending columns.")
    if census.empty:
        logger.warning("Census data missing (no API key?) — analysis will proceed without demographics.")

    # Join on FIPS
    df = cdc.copy()
    if not cms.empty:
        df = df.merge(
            cms[["fips", "medicare_spending_per_beneficiary"]],
            on="fips", how="left"
        )
    if not census.empty:
        demo_cols = ["fips", "median_income", "poverty_rate", "uninsured_rate",
                     "pct_black", "pct_hispanic", "total_population"]
        available_demo = [c for c in demo_cols if c in census.columns]
        df = df.merge(census[available_demo], on="fips", how="left")

    merged_count = df.dropna(subset=["fips"]).shape[0]
    full_merge = df.dropna(subset=[c for c in ["medicare_spending_per_beneficiary",
                                                "poverty_rate", "diabetes_rate"]
                                   if c in df.columns]).shape[0]
    print(f"\n=== MERGE SUMMARY ===")
    print(f"Total counties in dataset:           {merged_count:,}")
    print(f"Counties with complete data (3 sources): {full_merge:,}")

    # Equity scoring
    df = compute_equity_scores(df)
    q_summary = quartile_summary(df)
    state_summary = state_level_summary(df)

    # Statistical tests
    corr_results = pearson_correlation_matrix(df)
    ols_results = ols_regression(df)
    ttest_results = ttest_q4_vs_q1(df)

    # Persist enriched master table
    persist_master_county(df)

    return {
        "df": df,
        "corr_results": corr_results,
        "ols_results": ols_results,
        "ttest_results": ttest_results,
        "q_summary": q_summary,
        "state_summary": state_summary,
    }


def step_visualize(df: pd.DataFrame) -> None:
    from src.visualization.charts import generate_all_charts
    logger.info("=== Step 3: Visualizations ===")
    if df.empty:
        logger.warning("No data to visualize.")
        return
    generate_all_charts(df)


def step_report(results: dict) -> None:
    from src.analysis.report import generate_findings_report
    from src.analysis.export import export_tableau_csv, export_supporting_csvs

    logger.info("=== Step 4: Export & Report ===")
    df = results.get("df", pd.DataFrame())
    if df.empty:
        logger.warning("No data for report.")
        return

    tableau_count = export_tableau_csv(df)
    export_supporting_csvs(df, results.get("state_summary", pd.DataFrame()))

    generate_findings_report(
        df=df,
        ols_results=results.get("ols_results", {}),
        ttest_results=results.get("ttest_results", {}),
        state_summary=results.get("state_summary", pd.DataFrame()),
        corr_results=results.get("corr_results", pd.DataFrame()),
    )

    # Final summary
    ols = results.get("ols_results", {})
    ttest = results.get("ttest_results", {})
    df_clean = df.copy()
    total = len(df_clean)
    q4_n = int((df_clean.get("equity_quartile", pd.Series()) == 4).sum())

    print("\n" + "=" * 60)
    print("EQUICARE PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"Counties analyzed:          {total:,}")
    print(f"Tableau CSV rows exported:  {tableau_count:,}")
    if ols:
        print(f"OLS regression R²:          {ols.get('r_squared', 'N/A')}")
    if ttest:
        print(f"T-test p-value (Q4 vs Q1):  {ttest.get('p_value', 'N/A')}")
        diff = ttest.get('mean_difference', 0)
        q4_pop = df_clean[df_clean.get("equity_quartile", pd.Series()) == 4].get("total_population", pd.Series()).sum()
        beneficiaries = q4_pop * 0.185 if q4_pop > 0 else 0
        cost_b = round(diff * beneficiaries / 1e9, 1)
        print(f"Estimated cost of inequity: ${cost_b:.1f} billion/year")
    print("=" * 60)
    print(f"\nOutputs in output/:")
    print("  equicare_tableau.csv  <- import this into Tableau Public")
    print("  findings.md           ← executive summary report")
    print("  01_correlation_heatmap.png")
    print("  02_poverty_vs_spending.png")
    print("  03_spending_by_quartile.png")
    print("  04_worst_counties.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="EquiCare Pipeline")
    parser.add_argument(
        "--step",
        choices=["collect", "analyze", "visualize", "report", "all"],
        default="all",
    )
    args = parser.parse_args()

    results = {}

    if args.step in ("collect", "all"):
        step_collect()

    if args.step in ("analyze", "all"):
        results = step_analyze()

    if args.step in ("visualize", "all"):
        df = results.get("df", pd.DataFrame())
        if df.empty:
            from src.database import get_connection
            with get_connection() as conn:
                df = pd.read_sql("SELECT * FROM master_county", conn)
        step_visualize(df)
        results["df"] = df

    if args.step in ("report", "all"):
        step_report(results)


if __name__ == "__main__":
    main()
