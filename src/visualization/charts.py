"""
Four matplotlib/seaborn charts for EquiCare GitHub README.
All saved to output/ as PNG files.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

PALETTE = {
    "bg": "#F8F9FA",
    "grid": "#E9ECEF",
    "q1": "#2E7D32",
    "q2": "#F9A825",
    "q3": "#E65100",
    "q4": "#B71C1C",
}
QUARTILE_COLORS = [PALETTE["q1"], PALETTE["q2"], PALETTE["q3"], PALETTE["q4"]]


def _save(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Seaborn heatmap of correlation matrix — health vs socioeconomic variables."""
    vars_of_interest = [
        "diabetes_rate", "hypertension_rate", "mental_health_poor_days",
        "preventive_screening_rate", "poverty_rate", "uninsured_rate",
        "pct_black", "pct_hispanic", "median_income",
        "medicare_spending_per_beneficiary",
    ]
    available = [v for v in vars_of_interest if v in df.columns]
    corr_df = df[available].dropna().corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    labels = {
        "diabetes_rate": "Diabetes Rate",
        "hypertension_rate": "Hypertension Rate",
        "mental_health_poor_days": "Poor Mental Health Days",
        "preventive_screening_rate": "Preventive Screening",
        "poverty_rate": "Poverty Rate",
        "uninsured_rate": "Uninsured Rate",
        "pct_black": "% Black Population",
        "pct_hispanic": "% Hispanic Population",
        "median_income": "Median Income",
        "medicare_spending_per_beneficiary": "Medicare Spending",
    }
    corr_df = corr_df.rename(index=labels, columns=labels)
    mask_renamed = mask[:len(corr_df), :len(corr_df)]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    sns.heatmap(
        corr_df, mask=mask_renamed, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 8}, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix: Health Outcomes vs Socioeconomic Factors",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    _save(fig, "01_correlation_heatmap.png")


def plot_poverty_vs_spending(df: pd.DataFrame) -> None:
    """Scatter: poverty rate vs Medicare spending, colored by equity quartile."""
    data = df.dropna(subset=["poverty_rate", "medicare_spending_per_beneficiary",
                              "equity_quartile"]).copy()
    if data.empty:
        logger.warning("No data for poverty vs spending scatter")
        return

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    for q, color in zip([1, 2, 3, 4], QUARTILE_COLORS):
        sub = data[data["equity_quartile"] == q]
        ax.scatter(sub["poverty_rate"], sub["medicare_spending_per_beneficiary"],
                   c=color, alpha=0.5, s=25, label=f"Q{q}", edgecolors="none")

    # OLS trend line
    clean = data.dropna(subset=["poverty_rate", "medicare_spending_per_beneficiary"])
    z = np.polyfit(clean["poverty_rate"], clean["medicare_spending_per_beneficiary"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean["poverty_rate"].min(), clean["poverty_rate"].max(), 200)
    ax.plot(x_line, p(x_line), "k--", linewidth=2, label="OLS trend", alpha=0.8)

    ax.set_xlabel("Poverty Rate (%)", fontsize=12)
    ax.set_ylabel("Medicare Spending per Beneficiary ($)", fontsize=12)
    ax.set_title("Poverty Rate vs Medicare Spending per Beneficiary\nColored by Health Equity Quartile",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Equity Quartile", fontsize=10)
    ax.grid(color=PALETTE["grid"], linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    _save(fig, "02_poverty_vs_spending.png")


def plot_spending_by_quartile(df: pd.DataFrame) -> None:
    """Bar chart: mean Medicare spending by equity quartile with error bars."""
    data = df.dropna(subset=["equity_quartile", "medicare_spending_per_beneficiary"])
    if data.empty:
        return

    summary = data.groupby("equity_quartile")["medicare_spending_per_beneficiary"].agg(
        ["mean", "std", "count"]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.bar(
        [f"Q{int(q)}" for q in summary["equity_quartile"]],
        summary["mean"],
        yerr=summary["std"],
        color=QUARTILE_COLORS[:len(summary)],
        edgecolor="white", linewidth=0.5,
        capsize=5, error_kw={"linewidth": 1.5, "color": "#333"},
        width=0.6,
    )

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row["std"] + 50,
                f"${row['mean']:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Health Equity Quartile (Q1 = Best, Q4 = Worst)", fontsize=12)
    ax.set_ylabel("Mean Medicare Spending per Beneficiary ($)", fontsize=12)
    ax.set_title("Average Medicare Spending by Health Equity Quartile\n(Error bars = ±1 SD)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    legend_patches = [
        mpatches.Patch(color=QUARTILE_COLORS[i], label=f"Q{i+1}")
        for i in range(len(summary))
    ]
    ax.legend(handles=legend_patches, title="Equity Quartile", fontsize=10)
    fig.tight_layout()
    _save(fig, "03_spending_by_quartile.png")


def plot_worst_counties(df: pd.DataFrame) -> None:
    """Horizontal bar: top 20 worst counties by equity gap score."""
    data = df.dropna(subset=["equity_gap_score"]).nlargest(20, "equity_gap_score").copy()
    if data.empty:
        return

    data = data.sort_values("equity_gap_score")
    data["label"] = data["county_name"] + ", " + data["state"].fillna("")

    driver_colors = {
        "Diabetes": "#C62828", "Hypertension": "#E65100",
        "Poverty": "#4527A0", "Uninsured": "#1565C0",
    }
    colors = [driver_colors.get(d.split()[0], "#546E7A")
              for d in data["primary_driver"].fillna("Other")]

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(data["label"], data["equity_gap_score"],
                   color=colors, edgecolor="white", linewidth=0.3, height=0.7)

    for bar, val in zip(bars, data["equity_gap_score"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#333")

    ax.set_xlabel("Health Equity Gap Score (0–1)", fontsize=12)
    ax.set_title("Top 20 Counties by Health Equity Gap Score\nColored by Primary Driver",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)

    legend_patches = [mpatches.Patch(color=c, label=d) for d, c in driver_colors.items()]
    ax.legend(handles=legend_patches, title="Primary Driver", loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "04_worst_counties.png")


def generate_all_charts(df: pd.DataFrame) -> None:
    plot_correlation_heatmap(df)
    plot_poverty_vs_spending(df)
    plot_spending_by_quartile(df)
    plot_worst_counties(df)
    logger.info("All 4 charts generated.")
