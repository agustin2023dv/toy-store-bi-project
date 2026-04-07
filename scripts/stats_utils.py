"""
stats_utils.py – Reusable descriptive statistics functions for the toy-store BI project.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def compute_descriptive_stats(df: pd.DataFrame, column: str) -> pd.Series:
    """Compute a full set of descriptive statistics for a continuous variable.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    column : str
        Name of the numeric column to analyse.

    Returns
    -------
    pd.Series
        Named series containing mean, median, mode, std, variance, min, max,
        range, Q1, Q2, Q3, IQR, skewness, and kurtosis.
    """
    s = df[column].dropna()
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
    mode_val = s.mode()
    mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan

    return pd.Series(
        {
            "Mean": s.mean(),
            "Median": s.median(),
            "Mode": mode_val,
            "Std Dev": s.std(),
            "Variance": s.var(),
            "Min": s.min(),
            "Max": s.max(),
            "Range": s.max() - s.min(),
            "Q1 (25%)": q1,
            "Q2 (50%)": q2,
            "Q3 (75%)": q3,
            "IQR": q3 - q1,
            "Skewness": s.skew(),
            "Kurtosis": s.kurtosis(),
        },
        name=column,
    )


def compute_binary_stats(df: pd.DataFrame, column: str) -> pd.Series:
    """Compute proportion statistics for a binary (0/1) variable.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    column : str
        Name of the binary column.

    Returns
    -------
    pd.Series
        Named series with count, proportion (mean), and theoretical variance p*(1-p).
    """
    s = df[column].dropna()
    p = s.mean()

    return pd.Series(
        {
            "Count": int(s.count()),
            "Sum (1s)": int(s.sum()),
            "Proportion (p)": p,
            "Variance p*(1-p)": p * (1 - p),
        },
        name=column,
    )


def build_stats_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build a combined descriptive-statistics table for multiple continuous columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    columns : list[str]
        Continuous numeric columns to analyse.

    Returns
    -------
    pd.DataFrame
        Each column of the result corresponds to one input variable.
    """
    return pd.DataFrame({col: compute_descriptive_stats(df, col) for col in columns})
