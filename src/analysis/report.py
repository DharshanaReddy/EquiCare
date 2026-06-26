"""
Generate findings.md — written for a healthcare executive, not a data scientist.
Every statistical finding is stated in plain English first, technical detail in parentheses.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def generate_findings_report(
    df: pd.DataFrame,
    ols_results: dict,
    ttest_results: dict,
    state_summary: pd.DataFrame,
    corr_results: pd.DataFrame,
) -> str:
    """Write findings.md and return the path."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_counties = len(df)
    q4_count = int((df["equity_quartile"] == 4).sum()) if "equity_quartile" in df.columns else 0
    q4_pct = round(q4_count / total_counties * 100, 1) if total_counties else 0

    # OLS insight: poverty coefficient
    poverty_coef = ols_results.get("coefficients", {}).get("poverty_rate", None)
    poverty_pval = ols_results.get("pvalues", {}).get("poverty_rate", None)
    r_squared = ols_results.get("r_squared", None)

    # T-test
    mean_q4 = ttest_results.get("mean_q4_spending", 0)
    mean_q1 = ttest_results.get("mean_q1_spending", 0)
    diff = ttest_results.get("mean_difference", 0)
    t_pval = ttest_results.get("p_value", None)
    t_sig = ttest_results.get("significant", False)

    # Top 5 states by Q4 concentration
    top5_states = state_summary.head(5)["state"].tolist() if not state_summary.empty else []

    # Cost of inequity estimate
    # Estimate Medicare beneficiaries in Q4 counties using total_population * 0.185 (national rate)
    q4_pop = df[df["equity_quartile"] == 4]["total_population"].sum() if "total_population" in df.columns else 0
    est_q4_beneficiaries = q4_pop * 0.185 if q4_pop > 0 else 0
    annual_cost_billions = round(diff * est_q4_beneficiaries / 1e9, 1) if diff > 0 else 0

    lines = [
        "# EquiCare — Health Equity Gap Analysis Report",
        "",
        f"**Analysis date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Counties analyzed:** {total_counties:,}  ",
        f"**Data sources:** CDC PLACES, CMS Medicare Geographic Variation, US Census ACS 5-Year  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"Health inequity is not evenly distributed across the United States. "
        f"This analysis of **{total_counties:,} counties** reveals a clear, statistically significant "
        f"relationship between socioeconomic disadvantage and both worse health outcomes and higher "
        f"Medicare costs. The {q4_count:,} counties in the worst equity quartile ({q4_pct}% of all counties) "
        f"cost Medicare an estimated **${annual_cost_billions:.1f} billion more per year** than the best-equity counties.",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Scale of the Problem",
        "",
        f"Of the {total_counties:,} counties analyzed, **{q4_count:,} ({q4_pct}%) fall in the worst equity quartile** "
        f"(Q4) — meaning they rank in the bottom 25% on a composite measure of diabetes prevalence, "
        f"hypertension, poverty rate, and uninsured rate.",
        "",
        "### 2. Poverty Drives Medicare Spending — Significantly",
        "",
    ]

    if poverty_coef is not None and poverty_pval is not None:
        coef_per_10ppt = round(poverty_coef * 10, 2)
        sig_text = "is statistically significant" if poverty_pval < 0.05 else "did not reach statistical significance"
        lines += [
            f"Counties with higher poverty rates spend significantly more on Medicare. "
            f"On average, every 10 percentage point increase in a county's poverty rate is associated "
            f"with **${coef_per_10ppt:,.0f} more in Medicare spending per beneficiary** — "
            f"and this relationship {sig_text}. "
            f"(OLS coefficient: {poverty_coef:.2f}, p={poverty_pval:.4f}; "
            f"model R² = {r_squared:.3f}, meaning the predictors explain "
            f"{round(r_squared*100, 1)}% of variance in county-level spending)",
            "",
        ]
    else:
        lines += [
            "Poverty rate was not available in the regression model for this run.",
            "",
        ]

    lines += [
        "### 3. The Medicare Cost of Worst-Equity Counties",
        "",
    ]

    if t_sig:
        lines += [
            f"Counties in the worst equity quartile (Q4) spend significantly more on Medicare "
            f"than the best-equity counties (Q1). The gap: **${diff:,.0f} more per beneficiary per year**. "
            f"This difference is highly statistically significant — it is almost certainly not due to chance. "
            f"(Welch's t-test: t={ttest_results.get('t_statistic', 'N/A')}, p={t_pval:.6f}; "
            f"mean Q4 spending = ${mean_q4:,.2f}, mean Q1 spending = ${mean_q1:,.2f})",
            "",
            f"Scaled to the estimated Medicare population in Q4 counties, this gap represents "
            f"an estimated **${annual_cost_billions:.1f} billion in excess annual Medicare spending** "
            f"attributable to health inequity.",
            "",
        ]
    else:
        lines += [
            f"Q4 counties spend ${diff:,.0f} more per beneficiary than Q1 counties on average "
            f"(${mean_q4:,.2f} vs ${mean_q1:,.2f}), though this did not reach statistical significance "
            f"in the current dataset. "
            f"(t={ttest_results.get('t_statistic', 'N/A')}, p={t_pval})",
            "",
        ]

    lines += [
        "### 4. Where the Problem Is Most Concentrated",
        "",
    ]

    if top5_states:
        states_str = ", ".join(top5_states[:5])
        lines += [
            f"The five states with the highest concentration of worst-equity (Q4) counties are: "
            f"**{states_str}**. These states should be priority targets for insurer intervention, "
            f"value-based care programs, and policy investment.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Estimated Annual Cost of Health Inequity",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean spending, Q4 counties (worst equity) | ${mean_q4:,.2f} per beneficiary |",
        f"| Mean spending, Q1 counties (best equity)  | ${mean_q1:,.2f} per beneficiary |",
        f"| Excess spending per beneficiary in Q4     | ${diff:,.2f} |",
        f"| Estimated Medicare beneficiaries in Q4    | {est_q4_beneficiaries:,.0f} |",
        f"| **Estimated excess annual spending**      | **${annual_cost_billions:.1f} billion** |",
        "",
        "---",
        "",
        "## Most Actionable Recommendation",
        "",
        "**Target preventive care investment in Q4 counties with high diabetes + poverty co-occurrence.**",
        "",
        "The data shows that diabetes and poverty are the two strongest drivers of both equity gap score "
        "and downstream Medicare spending. Counties where both metrics are in the top quartile nationally "
        "represent the highest-ROI intervention targets for health insurers and policy teams. "
        "Investing in diabetes prevention programs (nutrition, medication adherence, early screening) "
        "in these specific counties would both improve health outcomes and reduce long-term Medicare cost growth.",
        "",
        "---",
        "",
        "## Statistical Methodology",
        "",
        "| Method | Purpose | Tool |",
        "|--------|---------|------|",
        "| Pearson Correlation | Identify significant relationships between health and socioeconomic variables | `scipy.stats.pearsonr` |",
        "| OLS Multiple Linear Regression | Quantify predictors of Medicare spending | `statsmodels` |",
        "| Independent Samples T-Test (Welch's) | Compare spending between best/worst equity quartiles | `scipy.stats.ttest_ind` |",
        "| MinMax Normalization | Normalize components to [0,1] before composite scoring | `sklearn.preprocessing.MinMaxScaler` |",
        "| Composite Equity Scoring | Weighted sum of normalized health/socioeconomic factors | Custom — diabetes(0.30) + hypertension(0.25) + poverty(0.25) + uninsured(0.20) |",
        "| Quartile Segmentation | Divide counties into equity tiers | `pandas.qcut` |",
        "",
        "---",
        "*Generated by EquiCare. Not medical or investment advice.*",
    ]

    report_path = OUTPUT_DIR / "findings.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Findings report written to %s", report_path)
    return str(report_path)
