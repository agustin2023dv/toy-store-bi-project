"""
forecast_utils.py
=================
Utility functions for stationarity testing, forecasting models,
and anomaly detection on daily time-series data.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Stationarity Tests
# ---------------------------------------------------------------------------

def adf_test(series, name="series"):
    """Run the Augmented Dickey-Fuller test.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    name : str
        Label printed in the output.

    Returns
    -------
    dict with test_statistic, p_value, used_lag, n_obs, critical_values, stationary (bool).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stationary = result[1] < 0.05
    return {
        "test": "ADF",
        "metric": name,
        "test_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "stationary": stationary,
        "conclusion": "Stationary" if stationary else "Non-stationary",
    }


def kpss_test(series, name="series", regression="c"):
    """Run the KPSS test (null: series IS stationary).

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    name : str
        Label printed in the output.
    regression : str
        'c' for constant (level), 'ct' for constant + trend.

    Returns
    -------
    dict with test_statistic, p_value, used_lag, critical_values, stationary (bool).
    """
    stat, p_value, used_lag, critical_values = kpss(series.dropna(), regression=regression, nlags="auto")
    stationary = p_value > 0.05  # fail to reject null → stationary
    return {
        "test": "KPSS",
        "metric": name,
        "test_statistic": stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "critical_values": critical_values,
        "stationary": stationary,
        "conclusion": "Stationary" if stationary else "Non-stationary",
    }


def stationarity_tests(series, name="series"):
    """Run ADF and KPSS tests and return a summary DataFrame.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    name : str
        Metric label.

    Returns
    -------
    pd.DataFrame with one row per test.
    """
    adf = adf_test(series, name)
    kp = kpss_test(series, name)
    rows = []
    for r in [adf, kp]:
        rows.append({
            "Metric": r["metric"],
            "Test": r["test"],
            "Statistic": round(r["test_statistic"], 4),
            "p-value": round(r["p_value"], 4),
            "Conclusion": r["conclusion"],
        })
    return pd.DataFrame(rows)


def plot_stationarity(series, diff1, diff2, name, save_dir="graphs"):
    """Plot original, 1st-diff, and 2nd-diff series with ACF/PACF.

    Parameters
    ----------
    series : pd.Series
        Original time series.
    diff1 : pd.Series
        First-order differenced series.
    diff2 : pd.Series
        Second-order differenced series.
    name : str
        Metric label.
    save_dir : str
        Directory for saving plots.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Stationarity Analysis — {name}", fontsize=16, fontweight="bold")

    datasets = [
        (series, f"Original {name}"),
        (diff1.dropna(), f"1st Diff {name}"),
        (diff2.dropna(), f"2nd Diff {name}"),
    ]
    for i, (data, title) in enumerate(datasets):
        axes[i, 0].plot(data, linewidth=0.7)
        axes[i, 0].set_title(title)
        axes[i, 0].set_xlabel("Date")
        plot_acf(data, ax=axes[i, 1], lags=40, title=f"ACF — {title}")
        plot_pacf(data, ax=axes[i, 2], lags=40, title=f"PACF — {title}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_dir}/stationarity_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Forecasting Models
# ---------------------------------------------------------------------------

def compute_metrics(actual, predicted):
    """Compute RMSE, MAE, and MAPE.

    Parameters
    ----------
    actual : array-like
        True values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    dict with rmse, mae, mape.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {"RMSE": round(rmse, 6), "MAE": round(mae, 6), "MAPE": round(mape, 2)}


def fit_sarima(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """Fit a SARIMA model, forecast on the test set, and return metrics.

    Parameters
    ----------
    train : pd.Series
        Training data (datetime index).
    test : pd.Series
        Test data (datetime index).
    order : tuple
        ARIMA (p, d, q) order.
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order.

    Returns
    -------
    dict with model, forecast (pd.Series), metrics (dict).
    """
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    metrics = compute_metrics(test, forecast)
    return {"model": fitted, "forecast": forecast, "metrics": metrics}


def fit_exponential_smoothing(train, test, seasonal_periods=7,
                              trend="add", seasonal="add"):
    """Fit Holt-Winters Exponential Smoothing model.

    Parameters
    ----------
    train : pd.Series
        Training data.
    test : pd.Series
        Test data.
    seasonal_periods : int
        Length of seasonal cycle.
    trend : str
        'add' or 'mul'.
    seasonal : str
        'add' or 'mul'.

    Returns
    -------
    dict with model, forecast, metrics.
    """
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                seasonal_periods=seasonal_periods)
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    metrics = compute_metrics(test, forecast)
    return {"model": fitted, "forecast": forecast, "metrics": metrics}


def fit_prophet(train, test):
    """Fit a Facebook Prophet model.

    Parameters
    ----------
    train : pd.Series
        Training data with datetime index.
    test : pd.Series
        Test data with datetime index.

    Returns
    -------
    dict with model, forecast, metrics.
    """
    from prophet import Prophet

    df_train = train.reset_index()
    df_train.columns = ["ds", "y"]

    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                yearly_seasonality=True)
    m.fit(df_train)

    future = pd.DataFrame({"ds": test.index})
    pred = m.predict(future)
    forecast = pd.Series(pred["yhat"].values, index=test.index)
    metrics = compute_metrics(test, forecast)
    return {"model": m, "forecast": forecast, "metrics": metrics}


def fit_ml_model(train, test, model_type="xgboost"):
    """Fit a machine-learning model using time-based features.

    Parameters
    ----------
    train : pd.Series
        Training data with datetime index.
    test : pd.Series
        Test data with datetime index.
    model_type : str
        'xgboost' or 'random_forest'.

    Returns
    -------
    dict with model, forecast, metrics.
    """
    from sklearn.ensemble import RandomForestRegressor

    def build_features(s):
        df = pd.DataFrame({"y": s})
        df["dayofweek"] = s.index.dayofweek
        df["dayofmonth"] = s.index.day
        df["month"] = s.index.month
        df["quarter"] = s.index.quarter
        df["dayofyear"] = s.index.dayofyear
        for lag in [1, 7, 14, 30]:
            df[f"lag_{lag}"] = s.shift(lag)
        for w in [7, 14, 30]:
            df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
            df[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
        return df.dropna()

    full = pd.concat([train, test])
    feat = build_features(full)
    feat_train = feat.loc[feat.index.isin(train.index)]
    feat_test = feat.loc[feat.index.isin(test.index)]

    X_train = feat_train.drop("y", axis=1)
    y_train = feat_train["y"]
    X_test = feat_test.drop("y", axis=1)
    y_test = feat_test["y"]

    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                 learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=10,
                                      random_state=42)

    model.fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=y_test.index)
    metrics = compute_metrics(y_test, preds)
    return {"model": model, "forecast": preds, "metrics": metrics}


def plot_forecast(train, test, forecast, model_name, metric_label,
                  save_dir="graphs"):
    """Plot train, test, and forecast on a single axis.

    Parameters
    ----------
    train : pd.Series
        Training data.
    test : pd.Series
        Actual test data.
    forecast : pd.Series
        Model predictions for the test period.
    model_name : str
        Model label for plot title.
    metric_label : str
        Y-axis label.
    save_dir : str
        Directory for saving.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train, label="Train", linewidth=0.8, alpha=0.7)
    ax.plot(test.index, test, label="Actual (Test)", linewidth=1.2, color="black")
    ax.plot(forecast.index, forecast, label=f"{model_name} Forecast",
            linewidth=1.2, linestyle="--", color="red")
    ax.set_title(f"{model_name} Forecast — {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{save_dir}/{model_name.lower().replace(' ', '_')}_forecast_{metric_label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(series, threshold=3):
    """Detect anomalies using z-score method.

    Parameters
    ----------
    series : pd.Series
        Time series.
    threshold : float
        Number of standard deviations for flagging.

    Returns
    -------
    pd.DataFrame with date, value, z_score for flagged points.
    """
    z = stats.zscore(series.dropna())
    z_series = pd.Series(z, index=series.dropna().index)
    mask = z_series.abs() > threshold
    anomalies = pd.DataFrame({
        "date": series.dropna().index[mask],
        "value": series.dropna().values[mask],
        "z_score": z_series[mask].values,
        "method": "Z-score",
    })
    return anomalies


def detect_anomalies_iqr(series):
    """Detect anomalies using the IQR (Tukey's fences) method.

    Parameters
    ----------
    series : pd.Series
        Time series.

    Returns
    -------
    pd.DataFrame with date, value, method for flagged points.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    anomalies = pd.DataFrame({
        "date": series.index[mask],
        "value": series.values[mask],
        "lower_bound": lower,
        "upper_bound": upper,
        "method": "IQR",
    })
    return anomalies


def detect_anomalies_isolation_forest(series, contamination=0.05):
    """Detect anomalies using Isolation Forest.

    Parameters
    ----------
    series : pd.Series
        Time series.
    contamination : float
        Expected proportion of outliers.

    Returns
    -------
    pd.DataFrame with date, value, score, method for flagged points.
    """
    X = series.dropna().values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=42)
    labels = clf.fit_predict(X)
    scores = clf.decision_function(X)
    mask = labels == -1
    anomalies = pd.DataFrame({
        "date": series.dropna().index[mask],
        "value": series.dropna().values[mask],
        "anomaly_score": scores[mask],
        "method": "Isolation Forest",
    })
    return anomalies


def plot_anomalies(series, anomaly_df, method_name, metric_label,
                   save_dir="graphs"):
    """Plot time series with detected anomalies highlighted in red.

    Parameters
    ----------
    series : pd.Series
        Full time series.
    anomaly_df : pd.DataFrame
        Must contain 'date' and 'value' columns.
    method_name : str
        Detection method name for the title.
    metric_label : str
        Y-axis label.
    save_dir : str
        Directory for saving.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series, linewidth=0.7, alpha=0.7, label=metric_label)
    if len(anomaly_df) > 0:
        ax.scatter(anomaly_df["date"], anomaly_df["value"],
                   color="red", s=30, zorder=5, label=f"Anomaly ({method_name})")
    ax.set_title(f"Anomaly Detection — {method_name} — {metric_label}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{save_dir}/anomaly_{method_name.lower().replace(' ', '_')}_{metric_label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
