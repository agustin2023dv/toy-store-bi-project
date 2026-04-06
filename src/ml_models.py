"""
ml_models.py
------------
Machine-learning models for the Toy Store BI project.

Three tasks are covered:

1. **Customer Segmentation** (unsupervised)
   K-Means clustering on RFM features (Recency, Frequency, Monetary).
   Class: ``CustomerSegmentation``

2. **Revenue Forecasting** (regression)
   RandomForestRegressor predicts monthly revenue from date features and
   category-level aggregates.
   Class: ``RevenueForecaster``

3. **Cancellation Prediction** (binary classification)
   LogisticRegression predicts whether an order will be cancelled based
   on customer and product features.
   Class: ``CancellationPredictor``
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 1. Customer Segmentation – RFM + K-Means
# ---------------------------------------------------------------------------

def build_rfm(master: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute RFM features per customer from the *master* DataFrame.

    Returns a DataFrame indexed by ``customer_id`` with columns
    ``recency``, ``frequency``, ``monetary``.
    """
    df = master[master["is_cancelled"] == False].copy()
    if reference_date is None:
        reference_date = df["order_date"].max() + pd.Timedelta(days=1)

    orders_per_customer = df.drop_duplicates(subset=["order_id"])
    rfm = (
        orders_per_customer.groupby("customer_id")
        .agg(
            recency=("order_date", lambda x: (reference_date - x.max()).days),
            frequency=("order_id", "nunique"),
        )
        .reset_index()
    )
    monetary = df.groupby("customer_id")["revenue"].sum().reset_index(name="monetary")
    rfm = rfm.merge(monetary, on="customer_id", how="left")
    return rfm.set_index("customer_id")


class CustomerSegmentation:
    """Segment customers using K-Means on scaled RFM features.

    Parameters
    ----------
    n_clusters : int
        Number of customer segments (default 4).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.rfm_: Optional[pd.DataFrame] = None
        self.labels_: Optional[np.ndarray] = None
        self.silhouette_score_: Optional[float] = None

    def fit(self, master: pd.DataFrame) -> "CustomerSegmentation":
        """Fit K-Means on RFM features derived from *master*."""
        rfm = build_rfm(master)
        self.rfm_ = rfm

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)),
            ]
        )
        self.labels_ = self.pipeline.fit_predict(rfm[["recency", "frequency", "monetary"]])
        self.rfm_["segment"] = self.labels_

        if len(rfm) > self.n_clusters:
            scaled = self.pipeline.named_steps["scaler"].transform(
                rfm[["recency", "frequency", "monetary"]]
            )
            self.silhouette_score_ = silhouette_score(scaled, self.labels_)
        return self

    def predict(self, master: pd.DataFrame) -> pd.Series:
        """Return cluster labels for customers in *master*."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        rfm = build_rfm(master)
        labels = self.pipeline.predict(rfm[["recency", "frequency", "monetary"]])
        return pd.Series(labels, index=rfm.index, name="segment")

    def segment_summary(self) -> pd.DataFrame:
        """Return mean RFM values per segment."""
        if self.rfm_ is None:
            raise RuntimeError("Call fit() first.")
        return self.rfm_.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(2)

    def plot_segments(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Scatter plot of Frequency vs. Monetary coloured by segment."""
        if self.rfm_ is None:
            raise RuntimeError("Call fit() first.")
        fig_created = ax is None
        if fig_created:
            fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=self.rfm_.reset_index(),
            x="frequency",
            y="monetary",
            hue="segment",
            palette="tab10",
            alpha=0.6,
            ax=ax,
        )
        ax.set_title("Customer Segments (Frequency vs Monetary)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Order Frequency")
        ax.set_ylabel("Total Spend (USD)")
        if fig_created:
            plt.tight_layout()
        return ax


# ---------------------------------------------------------------------------
# 2. Revenue Forecasting – RandomForestRegressor
# ---------------------------------------------------------------------------

def _build_forecast_features(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly level and engineer lag / rolling features."""
    df = master[master["is_cancelled"] == False].copy()
    monthly = (
        df.groupby(["order_year", "order_month"])["revenue"]
        .sum()
        .reset_index()
        .sort_values(["order_year", "order_month"])
        .reset_index(drop=True)
    )
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["order_month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["order_month"] / 12)
    monthly["lag_1"] = monthly["revenue"].shift(1)
    monthly["lag_2"] = monthly["revenue"].shift(2)
    monthly["lag_3"] = monthly["revenue"].shift(3)
    monthly["rolling_3"] = monthly["revenue"].shift(1).rolling(3).mean()
    monthly = monthly.dropna().reset_index(drop=True)
    return monthly


class RevenueForecaster:
    """Forecast monthly revenue with a RandomForestRegressor.

    After ``fit()``, call ``evaluate()`` for MAE/R² on the test split, or
    ``plot_forecast()`` to visualise actual vs predicted.
    """

    FEATURES = ["order_year", "order_month", "month_sin", "month_cos", "lag_1", "lag_2", "lag_3", "rolling_3"]

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self._monthly: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None
        self._y_pred: Optional[np.ndarray] = None

    def fit(self, master: pd.DataFrame, test_size: float = 0.2) -> "RevenueForecaster":
        monthly = _build_forecast_features(master)
        self._monthly = monthly

        X = monthly[self.FEATURES]
        y = monthly["revenue"]
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self._X_test = X_test
        self._y_test = y_test
        self._y_pred = self.model.predict(X_test)
        return self

    def evaluate(self) -> dict:
        """Return MAE and R² on the held-out test split."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return {
            "mae": float(mean_absolute_error(self._y_test, self._y_pred)),
            "r2": float(r2_score(self._y_test, self._y_pred)),
        }

    def feature_importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.FEATURES,
            name="importance",
        ).sort_values(ascending=False)

    def plot_forecast(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Line chart comparing actual vs. predicted test-set revenue."""
        if self._y_pred is None:
            raise RuntimeError("Call fit() first.")

        fig_created = ax is None
        if fig_created:
            fig, ax = plt.subplots(figsize=(10, 5))

        test_df = self._monthly.iloc[self._X_test.index].copy()
        test_df["predicted"] = self._y_pred

        period = pd.to_datetime(
            test_df["order_year"].astype(str)
            + "-"
            + test_df["order_month"].astype(str).str.zfill(2)
        )
        ax.plot(period, test_df["revenue"], label="Actual", marker="o")
        ax.plot(period, test_df["predicted"], label="Predicted", marker="s", linestyle="--")
        ax.set_title("Revenue Forecast – Actual vs Predicted", fontsize=13, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue (USD)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        if fig_created:
            plt.tight_layout()
        return ax


# ---------------------------------------------------------------------------
# 3. Cancellation Prediction – LogisticRegression
# ---------------------------------------------------------------------------

def _build_cancel_features(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and binary target for cancellation prediction."""
    df = master.drop_duplicates(subset=["order_id"]).copy()
    df["target"] = (df["status"] == "Cancelled").astype(int)

    # Encode category
    df["category_encoded"] = df["category"].astype("category").cat.codes
    features = ["unit_price", "quantity", "age", "order_month", "category_encoded"]
    df = df.dropna(subset=features + ["target"])
    return df[features], df["target"]


class CancellationPredictor:
    """Predict order cancellation using Logistic Regression.

    After ``fit()``, ``evaluate()`` returns a classification report dict and
    ``plot_coefficients()`` shows feature importance.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self._feature_names: Optional[list] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None

    def fit(self, master: pd.DataFrame, test_size: float = 0.2) -> "CancellationPredictor":
        X, y = _build_cancel_features(master)
        self._feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=self.C, random_state=self.random_state, max_iter=500)),
            ]
        )
        self.pipeline.fit(X_train, y_train)
        self._X_test = X_test
        self._y_test = y_test
        return self

    def evaluate(self) -> dict:
        """Return classification_report as a dict."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() first.")
        y_pred = self.pipeline.predict(self._X_test)
        return classification_report(self._y_test, y_pred, output_dict=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Call fit() first.")
        return self.pipeline.predict_proba(X)

    def plot_coefficients(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Bar chart of logistic regression coefficients."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() first.")
        coef = self.pipeline.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"feature": self._feature_names, "coefficient": coef})
        coef_df = coef_df.sort_values("coefficient")

        fig_created = ax is None
        if fig_created:
            fig, ax = plt.subplots(figsize=(8, 5))

        coef_df["color"] = coef_df["coefficient"].apply(lambda c: "positive" if c > 0 else "negative")
        sns.barplot(
            data=coef_df,
            x="coefficient",
            y="feature",
            hue="color",
            palette={"positive": "#d62728", "negative": "#1f77b4"},
            legend=False,
            ax=ax,
            orient="h",
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Cancellation Predictor – Feature Coefficients", fontsize=13, fontweight="bold")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Feature")
        if fig_created:
            plt.tight_layout()
        return ax
