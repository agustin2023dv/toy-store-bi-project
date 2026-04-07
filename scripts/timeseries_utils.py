"""
Utility functions for time-series decomposition & seasonality analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import os


# ── Decomposition ──────────────────────────────────────────────────────

def run_additive_decomposition(series, period=7):
    """Classical additive decomposition."""
    return seasonal_decompose(series, model="additive", period=period, extrapolate_trend=period)


def run_multiplicative_decomposition(series, period=7):
    """Classical multiplicative decomposition (series must be > 0)."""
    return seasonal_decompose(series, model="multiplicative", period=period, extrapolate_trend=period)


def run_stl_decomposition(series, period=7, seasonal=7, robust=True):
    """STL decomposition."""
    stl = STL(series, period=period, seasonal=seasonal, robust=robust)
    return stl.fit()


def plot_decomposition(result, title, save_path=None, model_type="additive"):
    """Plot decomposition result (observed, trend, seasonal, residual)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        ("Observed", result.observed),
        ("Trend", result.trend),
        ("Seasonal", result.seasonal),
        ("Residual", result.resid),
    ]
    for ax, (label, data) in zip(axes, components):
        ax.plot(data, linewidth=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{title} ({model_type})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def residual_variance(result):
    """Return variance of residuals (ignoring NaN)."""
    return np.nanvar(result.resid)


# ── Seasonality ────────────────────────────────────────────────────────

def seasonal_subseries_plot(daily, metric, period_col, period_order, title, save_path=None):
    """
    Seasonal subseries plot: box-plot of metric grouped by period_col.
    period_col example: 'weekday' (0-6) or 'month' (1-12).
    period_order: list defining x-axis order.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=daily, x=period_col, y=metric, order=period_order, ax=ax,
                palette="Set2", linewidth=0.8, fliersize=2)
    # overlay means
    means = daily.groupby(period_col)[metric].mean().reindex(period_order)
    ax.plot(range(len(period_order)), means.values, "D-", color="red", markersize=5, label="Mean")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def compute_fourier_spectrum(series, sampling_rate=1.0):
    """
    Compute one-sided Fourier amplitude spectrum.
    Returns (frequencies, amplitudes) arrays.
    """
    N = len(series)
    vals = np.asarray(series.values, dtype=float) - np.nanmean(series.values)  # de-mean
    yf = fft(vals)
    xf = np.asarray(fftfreq(N, d=1.0 / sampling_rate))
    # one-sided
    mask = xf > 0
    freqs = xf[mask]
    amplitudes = 2.0 / N * np.abs(np.asarray(yf)[mask])
    return freqs, amplitudes


def plot_fourier_spectrum(freqs, amplitudes, title, top_n=10, save_path=None):
    """Plot Fourier amplitude spectrum with top-n peaks annotated."""
    fig, ax = plt.subplots(figsize=(14, 5))
    # convert freq to period (days)
    periods = 1.0 / freqs

    ax.plot(periods, amplitudes, linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(1, min(365, periods.max()))
    ax.grid(True, alpha=0.3)

    # annotate top peaks
    top_idx = np.argsort(amplitudes)[-top_n:]
    for idx in top_idx:
        ax.annotate(f"{periods[idx]:.1f}d",
                     xy=(periods[idx], amplitudes[idx]),
                     fontsize=8, color="red", ha="center",
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def get_dominant_periods(freqs, amplitudes, top_n=5):
    """Return DataFrame of top-n dominant periods sorted by amplitude."""
    periods = 1.0 / freqs
    top_idx = np.argsort(amplitudes)[-top_n:][::-1]
    return pd.DataFrame({
        "Period (days)": periods[top_idx],
        "Frequency": freqs[top_idx],
        "Amplitude": amplitudes[top_idx],
    })


def dummy_variable_regression(daily, metric, dummies_col):
    """
    OLS regression of metric on dummy variables (e.g. weekday or month dummies).
    Returns (model, summary_df) with coefficients, t-stats, p-values.
    """
    import statsmodels.api as sm
    dummies = pd.get_dummies(daily[dummies_col], drop_first=True, prefix=dummies_col, dtype=float)
    X = sm.add_constant(dummies)
    y = np.asarray(daily[metric].values, dtype=float)
    mask = ~np.isnan(y)
    model = sm.OLS(y[mask], X[mask]).fit()  # type: ignore[call-overload]
    summary_df = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-stat": model.tvalues,
        "p-value": model.pvalues,
    })
    return model, summary_df


# ── Comparative / Growth ───────────────────────────────────────────────

def compute_growth_rates(daily, metric, date_col="date"):
    """
    Compute Year-over-Year, Quarter-over-Quarter, Month-over-Month growth rates.
    Returns a DataFrame with period aggregates and growth rates.
    """
    df = daily[[date_col, metric]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Monthly
    monthly = df.set_index(date_col).resample("MS")[metric].mean().to_frame("mean")
    monthly["MoM_growth"] = monthly["mean"].pct_change() * 100

    # Quarterly
    quarterly = df.set_index(date_col).resample("QS")[metric].mean().to_frame("mean")
    quarterly["QoQ_growth"] = quarterly["mean"].pct_change() * 100

    # Yearly
    yearly = df.set_index(date_col).resample("YS")[metric].mean().to_frame("mean")
    yearly["YoY_growth"] = yearly["mean"].pct_change() * 100

    return {"monthly": monthly, "quarterly": quarterly, "yearly": yearly}


def plot_growth_bar(growth_df, period_label, metric_label, save_path=None):
    """Bar chart of growth rates over time."""
    col = [c for c in growth_df.columns if "growth" in c][0]
    data = growth_df.dropna(subset=[col])
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if v >= 0 else "red" for v in data[col]]
    ax.bar(data.index.astype(str), data[col], color=colors, edgecolor="grey", linewidth=0.3)
    ax.set_title(f"{metric_label} – {period_label} Growth Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth (%)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def cusum_test(series):
    """
    Compute CUSUM (cumulative sum of deviations from overall mean)
    and confidence interval (±σ√n threshold lines).
    Returns (cusum, mean, std, upper_ci, lower_ci).
    """
    mean = np.nanmean(series)
    std = np.nanstd(series)
    deviations = series - mean
    cusum = np.nancumsum(deviations)
    n = np.arange(1, len(series) + 1)
    upper_ci = std * np.sqrt(n)
    lower_ci = -std * np.sqrt(n)
    return cusum, mean, std, upper_ci, lower_ci


def plot_cusum(series, title, save_path=None):
    """Plot CUSUM chart with confidence bands."""
    cusum, mean, std, upper_ci, lower_ci = cusum_test(series.values)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, cusum, linewidth=0.8, label="CUSUM")
    ax.plot(series.index, upper_ci, "r--", linewidth=0.7, label="Upper CI (σ√n)")
    ax.plot(series.index, lower_ci, "r--", linewidth=0.7, label="Lower CI (−σ√n)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("CUSUM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ── Statistical Tests ──────────────────────────────────────────────────

def anova_by_group(daily, metric, group_col):
    """One-way ANOVA of metric across groups. Returns (F-stat, p-value)."""
    groups = [g[metric].dropna().values for _, g in daily.groupby(group_col)]
    f_stat, p_val = sp_stats.f_oneway(*groups)
    return f_stat, p_val


def ttest_two_periods(daily, metric, split_date):
    """Independent t-test comparing metric before/after split_date."""
    before = daily.loc[daily["date"] < split_date, metric].dropna()
    after = daily.loc[daily["date"] >= split_date, metric].dropna()
    t_stat, p_val = sp_stats.ttest_ind(before, after, equal_var=False)
    return t_stat, p_val, before.mean(), after.mean()
