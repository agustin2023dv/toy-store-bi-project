"""
tests/test_ml_models.py
-----------------------
Unit tests for src/ml_models.py
"""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from src.data_generator import generate_dataset
from src.data_loader import build_master
from src.ml_models import (
    CancellationPredictor,
    CustomerSegmentation,
    RevenueForecaster,
    build_rfm,
)


@pytest.fixture(scope="module")
def master():
    tables = generate_dataset(seed=42)
    return build_master(tables)


# ── RFM ──────────────────────────────────────────────────────────────────────

def test_build_rfm_columns(master):
    rfm = build_rfm(master)
    assert set(rfm.columns) >= {"recency", "frequency", "monetary"}


def test_build_rfm_non_negative(master):
    rfm = build_rfm(master)
    assert (rfm["recency"] >= 0).all()
    assert (rfm["frequency"] >= 1).all()
    assert (rfm["monetary"] > 0).all()


# ── CustomerSegmentation ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_segmentation(master):
    seg = CustomerSegmentation(n_clusters=4, random_state=0)
    seg.fit(master)
    return seg


def test_segmentation_labels_count(fitted_segmentation, master):
    rfm = build_rfm(master)
    assert len(fitted_segmentation.labels_) == len(rfm)


def test_segmentation_labels_in_range(fitted_segmentation):
    labels = fitted_segmentation.labels_
    assert set(labels) <= set(range(4))


def test_segmentation_silhouette_positive(fitted_segmentation):
    assert fitted_segmentation.silhouette_score_ is not None
    assert fitted_segmentation.silhouette_score_ > -1


def test_segment_summary_shape(fitted_segmentation):
    summary = fitted_segmentation.segment_summary()
    assert summary.shape == (4, 3)


def test_segmentation_predict(fitted_segmentation, master):
    preds = fitted_segmentation.predict(master)
    assert len(preds) > 0
    assert set(preds.unique()) <= set(range(4))


def test_segmentation_plot(fitted_segmentation):
    ax = fitted_segmentation.plot_segments()
    assert ax is not None
    plt.close("all")


# ── RevenueForecaster ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_forecaster(master):
    rf = RevenueForecaster(n_estimators=50, random_state=0)
    rf.fit(master, test_size=0.25)
    return rf


def test_forecaster_metrics(fitted_forecaster):
    metrics = fitted_forecaster.evaluate()
    assert "mae" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0


def test_forecaster_feature_importances(fitted_forecaster):
    fi = fitted_forecaster.feature_importances()
    assert len(fi) == len(RevenueForecaster.FEATURES)
    assert abs(fi.sum() - 1.0) < 1e-6


def test_forecaster_plot(fitted_forecaster):
    ax = fitted_forecaster.plot_forecast()
    assert ax is not None
    plt.close("all")


# ── CancellationPredictor ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_predictor(master):
    cp = CancellationPredictor(random_state=0)
    cp.fit(master, test_size=0.2)
    return cp


def test_predictor_evaluate_keys(fitted_predictor):
    report = fitted_predictor.evaluate()
    assert "0" in report or 0 in report  # binary class labels


def test_predictor_predict_proba(fitted_predictor, master):
    from src.ml_models import _build_cancel_features
    X, _ = _build_cancel_features(master)
    proba = fitted_predictor.predict_proba(X.head(10))
    assert proba.shape == (10, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predictor_plot_coefficients(fitted_predictor):
    ax = fitted_predictor.plot_coefficients()
    assert ax is not None
    plt.close("all")


def test_predictor_raises_before_fit():
    cp = CancellationPredictor()
    with pytest.raises(RuntimeError):
        cp.evaluate()
