"""
tests/test_eda.py
-----------------
Smoke tests for src/eda.py – verify each plot function runs without error
and returns a matplotlib Axes.
"""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from src.data_generator import generate_dataset
from src.data_loader import build_master
from src import eda


@pytest.fixture(scope="module")
def master():
    tables = generate_dataset(seed=42)
    return build_master(tables)


def _fresh_ax():
    fig, ax = plt.subplots()
    return ax


def test_plot_monthly_revenue(master):
    ax = eda.plot_monthly_revenue(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_revenue_by_category(master):
    ax = eda.plot_revenue_by_category(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_top_products(master):
    ax = eda.plot_top_products(master, n=5, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_order_status_distribution(master):
    ax = eda.plot_order_status_distribution(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_age_distribution(master):
    ax = eda.plot_age_distribution(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_revenue_heatmap(master):
    ax = eda.plot_revenue_heatmap(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_customer_state_revenue(master):
    ax = eda.plot_customer_state_revenue(master, n=10, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_plot_correlation_matrix(master):
    ax = eda.plot_correlation_matrix(master, ax=_fresh_ax())
    assert ax is not None
    plt.close("all")


def test_save_path(master, tmp_path):
    out = str(tmp_path / "test_plot.png")
    eda.plot_monthly_revenue(master, save_path=out)
    import os
    assert os.path.exists(out)
