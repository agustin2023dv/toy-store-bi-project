"""
eda.py
------
Exploratory Data Analysis helpers that produce seaborn figures.

Each public function accepts the *master* DataFrame (from data_loader) and
an optional *ax* / *axes* argument so the figures can be embedded in
notebooks or saved to disk.

Public functions
~~~~~~~~~~~~~~~~
plot_monthly_revenue(master, ax=None)
plot_revenue_by_category(master, ax=None)
plot_top_products(master, n=10, ax=None)
plot_order_status_distribution(master, ax=None)
plot_age_distribution(master, ax=None)
plot_revenue_heatmap(master, ax=None)
plot_customer_state_map(master, ax=None)  -- bar chart per state
plot_correlation_matrix(master, ax=None)
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend – safe in scripts & notebooks

sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Public plots
# ---------------------------------------------------------------------------

def plot_monthly_revenue(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Line chart of total revenue aggregated by month."""
    df = (
        master[master["is_cancelled"] == False]
        .groupby(["order_year", "order_month"])["revenue"]
        .sum()
        .reset_index()
    )
    df["period"] = pd.to_datetime(
        df["order_year"].astype(str) + "-" + df["order_month"].astype(str).str.zfill(2)
    )
    df = df.sort_values("period")

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(data=df, x="period", y="revenue", ax=ax, marker="o", linewidth=2)
    ax.set_title("Monthly Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (USD)")
    ax.tick_params(axis="x", rotation=45)

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_revenue_by_category(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Horizontal bar chart of total revenue by product category."""
    df = (
        master[master["is_cancelled"] == False]
        .groupby("category")["revenue"]
        .sum()
        .sort_values()
        .reset_index()
    )

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(data=df, y="category", x="revenue", ax=ax, orient="h")
    ax.set_title("Revenue by Product Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("Category")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_top_products(
    master: pd.DataFrame,
    n: int = 10,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Bar chart of the top-N products by total revenue."""
    df = (
        master[master["is_cancelled"] == False]
        .groupby("product_name")["revenue"]
        .sum()
        .nlargest(n)
        .reset_index()
    )

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(data=df, x="revenue", y="product_name", ax=ax, orient="h")
    ax.set_title(f"Top {n} Products by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("Product")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_order_status_distribution(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Count-plot of order statuses."""
    df = master.drop_duplicates(subset=["order_id"])

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(7, 4))

    order = df["status"].value_counts().index.tolist()
    sns.countplot(data=df, x="status", order=order, ax=ax)
    ax.set_title("Order Status Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of Orders")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_age_distribution(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Histogram + KDE of customer age."""
    customers = master.drop_duplicates(subset=["customer_id"])

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(data=customers, x="age", kde=True, bins=20, ax=ax)
    ax.set_title("Customer Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_revenue_heatmap(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Heatmap of revenue by month (rows) and year (columns)."""
    df = (
        master[master["is_cancelled"] == False]
        .groupby(["order_month", "order_year"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        df,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Revenue (USD)"},
    )
    ax.set_title("Revenue Heatmap (Month × Year)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_customer_state_revenue(
    master: pd.DataFrame,
    n: int = 15,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Horizontal bar chart of revenue by US state (top-N)."""
    df = (
        master[master["is_cancelled"] == False]
        .groupby("state")["revenue"]
        .sum()
        .nlargest(n)
        .reset_index()
    )

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(data=df, x="revenue", y="state", ax=ax, orient="h")
    ax.set_title(f"Top {n} States by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (USD)")
    ax.set_ylabel("State")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax


def plot_correlation_matrix(
    master: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Heatmap of Pearson correlations for numeric columns."""
    numeric_cols = ["unit_price", "quantity", "revenue", "age"]
    df = master[numeric_cols].dropna()
    corr = df.corr()

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")

    if fig_created:
        plt.tight_layout()
        _savefig_or_show(ax.get_figure(), save_path)
    return ax
