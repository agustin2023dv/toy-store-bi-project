"""
volatility_utils.py – Reusable functions for volatility and variability
analysis in the toy-store BI project.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt


# ---------------------------------------------------------------------------
# Coefficient of Variation
# ---------------------------------------------------------------------------

def compute_cv(df: pd.DataFrame, column: str, period_label: str = "Full") -> dict:
    """Compute the Coefficient of Variation for a single metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric column.
    column : str
        Column name to compute CV for.
    period_label : str
        Label describing the period (e.g. "Full", "2012", "2014").

    Returns
    -------
    dict
        Keys: period, column, mean, std, cv_pct.
    """
    series = df[column].dropna()
    mean = series.mean()
    std = series.std()
    cv_pct = (std / mean) * 100 if mean != 0 else np.nan
    return {
        "period": period_label,
        "metric": column,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv_pct": round(cv_pct, 2),
    }


def compute_cv_table(df: pd.DataFrame, columns: list[str],
                     period_label: str = "Full") -> pd.DataFrame:
    """Compute CV for multiple metrics and return as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric columns.
    columns : list[str]
        List of column names.
    period_label : str
        Label describing the period.

    Returns
    -------
    pd.DataFrame
        One row per metric with CV statistics.
    """
    rows = [compute_cv(df, col, period_label) for col in columns]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling Standard Deviation
# ---------------------------------------------------------------------------

def compute_rolling_std(df: pd.DataFrame, column: str,
                        windows: list[int] | None = None) -> pd.DataFrame:
    """Compute rolling standard deviation for one or more window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column and the target *column*.
    column : str
        Metric column name.
    windows : list[int], optional
        Window sizes in days. Defaults to [7, 30].

    Returns
    -------
    pd.DataFrame
        Original date and metric columns plus rolling std columns named
        ``{column}_std_{w}d``.
    """
    if windows is None:
        windows = [7, 30]
    result = df[["date", column]].copy()
    for w in windows:
        result[f"{column}_std_{w}d"] = (
            df[column].rolling(window=w, min_periods=max(3, w // 2)).std()
        )
    return result


def plot_rolling_std(df: pd.DataFrame, column: str,
                     windows: list[int] | None = None,
                     title: str | None = None,
                     save_path: str | None = None) -> plt.Figure:
    """Plot the original metric with overlaid rolling standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and the target *column*.
    column : str
        Metric column name.
    windows : list[int], optional
        Window sizes. Defaults to [7, 30].
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if windows is None:
        windows = [7, 30]
    rolling_df = compute_rolling_std(df, column, windows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel: original metric
    axes[0].plot(df["date"], df[column], linewidth=0.7, alpha=0.6, label=column)
    axes[0].set_ylabel(column)
    axes[0].set_title(title or f"{column} – Value & Rolling Std Dev")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: rolling std
    colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]
    for i, w in enumerate(windows):
        col_name = f"{column}_std_{w}d"
        c = colors[i % len(colors)]
        axes[1].plot(rolling_df["date"], rolling_df[col_name],
                     linewidth=1.2, label=f"{w}-day rolling std", color=c)
    axes[1].set_ylabel("Standard Deviation")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Period Comparison / Range Analysis
# ---------------------------------------------------------------------------

def compare_periods(df: pd.DataFrame, column: str,
                    split_date: str = "2014-01-01") -> dict:
    """Compare descriptive stats and variance for a metric across two periods.

    Also performs an F-test and Levene test for equality of variances.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and the target *column*.
    column : str
        Metric column name.
    split_date : str
        ISO date string to split the data.

    Returns
    -------
    dict
        Keys: metric, early/late stats, f_stat, f_pvalue,
        levene_stat, levene_pvalue.
    """
    early = df.loc[df["date"] < split_date, column].dropna()
    late = df.loc[df["date"] >= split_date, column].dropna()

    def _stats(s, label):
        return {
            f"{label}_n": len(s),
            f"{label}_mean": round(s.mean(), 4),
            f"{label}_std": round(s.std(), 4),
            f"{label}_min": round(s.min(), 4),
            f"{label}_max": round(s.max(), 4),
            f"{label}_range": round(s.max() - s.min(), 4),
            f"{label}_cv": round((s.std() / s.mean()) * 100, 2) if s.mean() != 0 else np.nan,
        }

    result = {"metric": column}
    result.update(_stats(early, "early"))
    result.update(_stats(late, "late"))

    # F-test (ratio of variances)
    var_early = early.var(ddof=1)
    var_late = late.var(ddof=1)
    if var_early > 0 and var_late > 0:
        f_stat = var_late / var_early
        dfn = len(late) - 1
        dfd = len(early) - 1
        f_pvalue = 2 * min(
            sp_stats.f.cdf(f_stat, dfn, dfd),
            1 - sp_stats.f.cdf(f_stat, dfn, dfd),
        )
    else:
        f_stat, f_pvalue = np.nan, np.nan
    result["f_stat"] = round(f_stat, 4)
    result["f_pvalue"] = f_pvalue

    # Levene test
    lev_stat, lev_pvalue = sp_stats.levene(early, late)
    result["levene_stat"] = round(lev_stat, 4)
    result["levene_pvalue"] = lev_pvalue

    return result


def build_period_comparison_table(df: pd.DataFrame, columns: list[str],
                                  split_date: str = "2014-01-01") -> pd.DataFrame:
    """Build a comparison table for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Daily DataFrame.
    columns : list[str]
        Metric column names.
    split_date : str
        ISO date string to split.

    Returns
    -------
    pd.DataFrame
    """
    rows = [compare_periods(df, col, split_date) for col in columns]
    return pd.DataFrame(rows)


def plot_period_boxplots(df: pd.DataFrame, columns: list[str],
                         split_date: str = "2014-01-01",
                         save_path: str | None = None) -> plt.Figure:
    """Side-by-side boxplots comparing early vs late period for each metric.

    Parameters
    ----------
    df : pd.DataFrame
        Daily DataFrame with ``date`` column.
    columns : list[str]
        Metric column names.
    split_date : str
        Date to split periods.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.copy()
    split = pd.Timestamp(split_date)
    df["period"] = df["date"].apply(
        lambda d: f"Before {split_date[:4]}" if d < split else f"{split_date[:4]}+"
    )

    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data_early = df.loc[df["date"] < split, col].dropna()
        data_late = df.loc[df["date"] >= split, col].dropna()
        bp = ax.boxplot(
            [data_early, data_late],
            labels=[f"Before {split_date[:4]}", f"{split_date[:4]}+"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Period Comparison – Before vs After {split_date}", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Change Point Detection
# ---------------------------------------------------------------------------

def detect_change_point_manual(df: pd.DataFrame, column: str,
                               window: int = 30,
                               threshold_factor: float = 2.0) -> pd.DataFrame:
    """Detect change points by finding where rolling std exceeds a threshold.

    The threshold is defined as ``threshold_factor * median(rolling_std)``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and the target *column*.
    column : str
        Metric column name.
    window : int
        Rolling window size for std calculation.
    threshold_factor : float
        Multiplier for the median rolling std to set threshold.

    Returns
    -------
    pd.DataFrame
        Rows where the rolling std first exceeds the threshold (transition
        points from below to above).
    """
    rolling_std = df[column].rolling(window=window, min_periods=window // 2).std()
    median_std = rolling_std.median()
    threshold = threshold_factor * median_std

    above = rolling_std > threshold
    # Find transitions: False → True
    transitions = above & ~above.shift(1, fill_value=False)
    change_dates = df.loc[transitions, ["date"]].copy()
    change_dates["rolling_std"] = rolling_std.loc[transitions].values
    change_dates["threshold"] = threshold
    change_dates["metric"] = column
    return change_dates


def detect_change_point_ruptures(df: pd.DataFrame, column: str,
                                  n_bkps: int = 3,
                                  model: str = "rbf",
                                  min_size: int = 30) -> list[dict]:
    """Detect change points using the PELT algorithm from the ruptures library.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and the target *column*.
    column : str
        Metric column name.
    n_bkps : int
        Maximum number of breakpoints to detect.
    model : str
        Cost model for ruptures (e.g. "rbf", "l2", "normal").
    min_size : int
        Minimum segment length.

    Returns
    -------
    list[dict]
        Each dict has keys: metric, bkp_index, bkp_date.
    """
    signal = df[column].dropna().values
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=10)

    # Remove the last breakpoint (always == len(signal))
    breakpoints = [b for b in breakpoints if b < len(signal)]

    results = []
    for b in breakpoints[:n_bkps]:
        results.append({
            "metric": column,
            "bkp_index": b,
            "bkp_date": df["date"].iloc[b],
        })
    return results


def plot_change_points(df: pd.DataFrame, column: str,
                       change_dates: list,
                       window: int = 30,
                       title: str | None = None,
                       save_path: str | None = None) -> plt.Figure:
    """Plot rolling std with detected change point dates marked.

    Parameters
    ----------
    df : pd.DataFrame
        Daily DataFrame.
    column : str
        Metric column name.
    change_dates : list
        List of datetime-like change point dates to mark.
    window : int
        Rolling window for std.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rolling_std = df[column].rolling(window=window, min_periods=window // 2).std()
    median_std = rolling_std.median()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], rolling_std, linewidth=1, label=f"{window}-day rolling std",
            color="#2980b9")
    ax.axhline(y=2 * median_std, color="#e67e22", linestyle="--", linewidth=0.8,
               label=f"2× median threshold ({2 * median_std:.4f})")

    for d in change_dates:
        ax.axvline(x=pd.Timestamp(d), color="#e74c3c", linestyle="-",
                   linewidth=1.2, alpha=0.7)
    # Add a single legend entry for change points
    if change_dates:
        ax.axvline(x=pd.Timestamp(change_dates[0]), color="#e74c3c",
                   linestyle="-", linewidth=1.2, alpha=0.7,
                   label="Change point")

    ax.set_title(title or f"Change Point Detection – {column}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
