"""
Core statistical analysis for EquiCare.

Functions:
  - pearson_correlation_matrix: all health × socioeconomic variable pairs
  - ols_regression: Medicare spending predicted by socioeconomic/health factors
  - ttest_q4_vs_q1: spending difference between worst and best equity quartiles
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)

HEALTH_VARS = ["diabetes_rate", "hypertension_rate", "mental_health_poor_days",
               "preventive_screening_rate"]
SOCIO_VARS  = ["median_income", "poverty_rate", "uninsured_rate",
               "pct_black", "pct_hispanic"]
ALL_NUMERIC = HEALTH_VARS + SOCIO_VARS + ["medicare_spending_per_beneficiary"]


def pearson_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r and p-value for every pair of health vs socioeconomic variables.
    Returns a DataFrame with columns: var1, var2, r, p_value, significant.
    """
    rows = []
    available_health = [v for v in HEALTH_VARS if v in df.columns]
    available_socio  = [v for v in SOCIO_VARS + ["medicare_spending_per_beneficiary"]
                        if v in df.columns]

    for h in available_health:
        for s in available_socio:
            clean = df[[h, s]].dropna()
            if len(clean) < 10:
                continue
            r, p = stats.pearsonr(clean[h], clean[s])
            rows.append({
                "var1": h, "var2": s,
                "r": round(r, 4),
                "p_value": round(p, 6),
                "significant": p < 0.05,
                "n": len(clean),
            })

    result = pd.DataFrame(rows).sort_values("r", key=abs, ascending=False)

    sig = result[result["significant"]]
    logger.info("Pearson correlation: %d significant pairs (p<0.05) out of %d total",
                len(sig), len(result))

    print("\n=== TOP 10 SIGNIFICANT CORRELATIONS ===")
    print(f"{'Variable 1':<35} {'Variable 2':<35} {'r':>8} {'p-value':>12}")
    print("-" * 95)
    for _, row in result[result["significant"]].head(10).iterrows():
        print(f"{row['var1']:<35} {row['var2']:<35} {row['r']:>8.4f} {row['p_value']:>12.6f}")

    return result


def ols_regression(df: pd.DataFrame) -> dict:
    """
    OLS regression: medicare_spending_per_beneficiary ~
        poverty_rate + uninsured_rate + diabetes_rate + hypertension_rate + median_income

    Returns dict with model, summary string, and key statistics.
    """
    required = ["medicare_spending_per_beneficiary", "poverty_rate", "uninsured_rate",
                "diabetes_rate", "hypertension_rate", "median_income"]
    available = [c for c in required if c in df.columns]
    clean = df[available].dropna()

    if len(clean) < 50:
        logger.warning("Insufficient data for OLS regression (%d rows)", len(clean))
        return {}

    X_cols = [c for c in available if c != "medicare_spending_per_beneficiary"]
    X = sm.add_constant(clean[X_cols])
    y = clean["medicare_spending_per_beneficiary"]

    model = sm.OLS(y, X).fit()

    print("\n=== OLS REGRESSION: Medicare Spending ~ Socioeconomic + Health Factors ===")
    print(model.summary())

    # Extract significant predictors
    sig_predictors = model.pvalues[model.pvalues < 0.05].drop("const", errors="ignore")
    logger.info("Significant OLS predictors (p<0.05): %s", list(sig_predictors.index))

    return {
        "model": model,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "n_obs": int(model.nobs),
        "f_pvalue": round(model.f_pvalue, 6),
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "significant_predictors": list(sig_predictors.index),
        "summary": model.summary().as_text(),
    }


def ttest_q4_vs_q1(df: pd.DataFrame) -> dict:
    """
    Independent samples t-test: Medicare spending in Q4 (worst equity)
    vs Q1 (best equity) counties.
    """
    if "equity_quartile" not in df.columns:
        logger.warning("equity_quartile column missing — run equity scoring first")
        return {}

    q4 = df[df["equity_quartile"] == 4]["medicare_spending_per_beneficiary"].dropna()
    q1 = df[df["equity_quartile"] == 1]["medicare_spending_per_beneficiary"].dropna()

    if len(q4) < 10 or len(q1) < 10:
        logger.warning("Too few counties in Q4 (%d) or Q1 (%d) for t-test", len(q4), len(q1))
        return {}

    t_stat, p_value = stats.ttest_ind(q4, q1, equal_var=False)  # Welch's t-test

    mean_q4 = q4.mean()
    mean_q1 = q1.mean()
    diff = mean_q4 - mean_q1

    result = {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "mean_q4_spending": round(mean_q4, 2),
        "mean_q1_spending": round(mean_q1, 2),
        "mean_difference": round(diff, 2),
        "significant": p_value < 0.05,
        "n_q4": len(q4),
        "n_q1": len(q1),
    }

    print("\n=== T-TEST: Q4 vs Q1 Medicare Spending ===")
    print(f"  Q4 (worst equity) mean spending:  ${mean_q4:,.2f}  (n={len(q4)})")
    print(f"  Q1 (best equity)  mean spending:  ${mean_q1:,.2f}  (n={len(q1)})")
    print(f"  Mean difference:                  ${diff:,.2f}")
    print(f"  t-statistic: {t_stat:.4f}   p-value: {p_value:.6f}")
    print(f"  Statistically significant: {'YES (p < 0.05)' if p_value < 0.05 else 'NO'}")

    return result
