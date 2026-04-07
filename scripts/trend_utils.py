"""
trend_utils.py – Reusable functions for trend and correlation analysis
in the toy-store BI project.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------

def fit_linear_trend(df: pd.DataFrame, metric: str):
    """Fit an OLS linear regression of *metric* against a numeric time index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column and the target *metric* column.
    metric : str
        Column name of the metric to regress.

    Returns
    -------
    dict
        Keys: slope, intercept, r_value, r_squared, p_value, std_err,
        fitted (np.ndarray of predicted values), time_idx (numeric index).
    """
    y = np.asarray(df[metric].values, dtype=float)
    x = np.arange(len(y))
    _s, _i, _r, _p, _se = sp_stats.linregress(x, y)
    slope: float = float(_s)  # type: ignore[arg-type]
    intercept: float = float(_i)  # type: ignore[arg-type]
    r_value: float = float(_r)  # type: ignore[arg-type]
    p_value: float = float(_p)  # type: ignore[arg-type]
    std_err: float = float(_se)  # type: ignore[arg-type]
    fitted = intercept + slope * x
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "fitted": fitted,
        "time_idx": x,
    }


def fit_polynomial_trend(df: pd.DataFrame, metric: str, degree: int = 2):
    """Fit a polynomial regression of *metric* against a numeric time index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column and the target *metric* column.
    metric : str
        Column name of the metric to regress.
    degree : int, optional
        Polynomial degree (default 2 = quadratic).

    Returns
    -------
    dict
        Keys: coefficients, r_squared, fitted, time_idx.
    """
    y = np.asarray(df[metric].values, dtype=float)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    fitted = poly(x)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {
        "coefficients": coeffs,
        "r_squared": r_squared,
        "fitted": fitted,
        "time_idx": x,
    }


def compute_moving_averages(df: pd.DataFrame, metric: str,
                            windows: list[int] | None = None) -> pd.DataFrame:
    """Compute rolling means for *metric*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the target *metric* column.
    metric : str
        Column to smooth.
    windows : list[int], optional
        Window sizes in days (default [7, 30, 90]).

    Returns
    -------
    pd.DataFrame
        Original dataframe with added MA columns (e.g., ``MA_7``).
    """
    if windows is None:
        windows = [7, 30, 90]
    result = df.copy()
    for w in windows:
        result[f"MA_{w}"] = result[metric].rolling(window=w, min_periods=1).mean()
    return result


def mann_kendall_test(series: pd.Series):
    """Run the Mann-Kendall trend test on *series*.

    Uses the ``pymannkendall`` package.  Falls back to scipy.stats.kendalltau
    with a numeric index if pymannkendall is unavailable.

    Parameters
    ----------
    series : pd.Series
        Time-ordered numeric values.

    Returns
    -------
    dict
        Keys: trend, p_value, statistic, method.
    """
    try:
        import pymannkendall as mk
        result = mk.original_test(series.dropna())
        return {
            "trend": result.trend,
            "p_value": result.p,
            "statistic": result.s,
            "method": "pymannkendall",
        }
    except ImportError:
        x = np.arange(len(series))
        _tau, _pval = sp_stats.kendalltau(x, series.values)
        tau: float = float(_tau)  # type: ignore[arg-type]
        p_value: float = float(_pval)  # type: ignore[arg-type]
        trend = "increasing" if tau > 0 else ("decreasing" if tau < 0 else "no trend")
        return {
            "trend": trend,
            "p_value": p_value,
            "statistic": tau,
            "method": "scipy.kendalltau",
        }


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame, metric: str):
    """Compute Pearson and Spearman correlations between a numeric time index
    and *metric*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the target *metric* column.
    metric : str
        Column name.

    Returns
    -------
    dict
        Keys: pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    y = df[metric].dropna().values
    x = np.arange(len(y))
    pearson_r, pearson_p = sp_stats.pearsonr(x, y)
    spearman_rho, spearman_p = sp_stats.spearmanr(x, y)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def compute_cross_correlation(s1: pd.Series, s2: pd.Series,
                              max_lag: int = 30) -> pd.DataFrame:
    """Compute cross-correlation between *s1* and *s2* at multiple lags.

    Parameters
    ----------
    s1, s2 : pd.Series
        Equal-length time series.
    max_lag : int, optional
        Maximum number of lags to compute (default 30).

    Returns
    -------
    pd.DataFrame
        Columns: lag, cross_corr.
    """
    s1_norm = (s1 - s1.mean()) / s1.std()
    s2_norm = (s2 - s2.mean()) / s2.std()
    n = len(s1_norm)
    lags = range(-max_lag, max_lag + 1)
    cc = []
    for lag in lags:
        if lag >= 0:
            corr = np.corrcoef(s1_norm[lag:], s2_norm[:n - lag])[0, 1] if n - lag > 0 else np.nan
        else:
            corr = np.corrcoef(s1_norm[:n + lag], s2_norm[-lag:])[0, 1] if n + lag > 0 else np.nan
        cc.append(corr)
    return pd.DataFrame({"lag": list(lags), "cross_corr": cc})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trend_regression(df: pd.DataFrame, metric: str, metric_label: str,
                         save_path: str | None = None):
    """Plot the raw metric with linear and polynomial regression overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and *metric* columns.
    metric : str
        Column to plot.
    metric_label : str
        Human-readable name for axis labels.
    save_path : str, optional
        If given, save the figure to this path.
    """
    lin = fit_linear_trend(df, metric)
    poly = fit_polynomial_trend(df, metric)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df[metric], alpha=0.35, linewidth=0.8, label="Daily")
    ax.plot(df["date"], lin["fitted"], color="red", linewidth=2,
            label=f"Linear (R²={lin['r_squared']:.4f})")
    ax.plot(df["date"], poly["fitted"], color="green", linewidth=2,
            linestyle="--", label=f"Poly deg-2 (R²={poly['r_squared']:.4f})")
    ax.set_title(f"Trend Regression — {metric_label}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_moving_averages(df: pd.DataFrame, metric: str, metric_label: str,
                         windows: list[int] | None = None,
                         save_path: str | None = None):
    """Plot raw metric with multiple moving-average overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and *metric* columns.
    metric : str
        Column to plot.
    metric_label : str
        Human-readable name for axis labels.
    windows : list[int], optional
        Window sizes (default [7, 30, 90]).
    save_path : str, optional
        If given, save the figure to this path.
    """
    if windows is None:
        windows = [7, 30, 90]
    ma_df = compute_moving_averages(df, metric, windows)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ma_df["date"], ma_df[metric], alpha=0.25, linewidth=0.7,
            label="Daily", color="gray")
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for w, c in zip(windows, colours):
        ax.plot(ma_df["date"], ma_df[f"MA_{w}"], linewidth=2, label=f"{w}-day MA",
                color=c)
    ax.set_title(f"Moving Averages — {metric_label}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_acf_pacf(series: pd.Series, metric_name: str, lags: int = 30,
                  save_path: str | None = None):
    """Plot ACF and PACF side by side.

    Parameters
    ----------
    series : pd.Series
        Time-ordered numeric values (NaN-free preferred).
    metric_name : str
        Human-readable name for the title.
    lags : int, optional
        Number of lags (default 30).
    save_path : str, optional
        If given, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f"ACF — {metric_name}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title=f"PACF — {metric_name}",
              method="ywm")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cross_correlation(cc_df: pd.DataFrame, label: str,
                           save_path: str | None = None):
    """Plot cross-correlation between two series.

    Parameters
    ----------
    cc_df : pd.DataFrame
        Output of ``compute_cross_correlation``.
    label : str
        Title label.
    save_path : str, optional
        If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(cc_df["lag"], cc_df["cross_corr"], width=0.8, color="#1f77b4",
           edgecolor="white")
    n = len(cc_df)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Cross-Correlation — {label}", fontsize=14)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_trend_summary(df: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    """Build a summary table of trend statistics for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Daily aggregated dataframe with ``date`` and metric columns.
    metrics : dict[str, str]
        Mapping of column name → human-readable label.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per metric.
    """
    rows = []
    for col, label in metrics.items():
        lin = fit_linear_trend(df, col)
        poly = fit_polynomial_trend(df, col)
        mk = mann_kendall_test(df[col])
        rows.append({
            "Metric": label,
            "Linear Slope": lin["slope"],
            "Linear R²": lin["r_squared"],
            "Linear p-value": lin["p_value"],
            "Poly R²": poly["r_squared"],
            "MK Trend": mk["trend"],
            "MK p-value": mk["p_value"],
        })
    return pd.DataFrame(rows)


def build_correlation_summary(df: pd.DataFrame,
                              metrics: dict[str, str]) -> pd.DataFrame:
    """Build a summary table of correlation statistics for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Daily aggregated dataframe.
    metrics : dict[str, str]
        Mapping of column name → human-readable label.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per metric.
    """
    rows = []
    for col, label in metrics.items():
        corr = compute_correlations(df, col)
        rows.append({
            "Metric": label,
            "Pearson r": corr["pearson_r"],
            "Pearson p-value": corr["pearson_p"],
            "Spearman ρ": corr["spearman_rho"],
            "Spearman p-value": corr["spearman_p"],
        })
    return pd.DataFrame(rows)
