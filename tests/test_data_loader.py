"""
tests/test_data_loader.py
--------------------------
Unit tests for src/data_loader.py
"""

import pandas as pd
import pytest

from src.data_generator import save_dataset
from src.data_loader import build_master, load_or_generate, load_tables


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    save_dataset(d, seed=42)
    return d


@pytest.fixture(scope="module")
def tables(data_dir):
    return load_tables(data_dir)


@pytest.fixture(scope="module")
def master(tables):
    return build_master(tables)


def test_load_tables_returns_four_keys(tables):
    assert set(tables.keys()) == {"customers", "products", "orders", "order_items"}


def test_load_tables_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tables(tmp_path)


def test_master_has_revenue(master):
    assert "revenue" in master.columns
    assert (master["revenue"] >= 0).all()


def test_master_has_date_parts(master):
    for col in ("order_year", "order_month", "order_quarter"):
        assert col in master.columns


def test_master_is_cancelled_flag(master):
    assert master["is_cancelled"].dtype == bool


def test_load_or_generate_creates_files(tmp_path):
    tables = load_or_generate(tmp_path, seed=5)
    assert set(tables.keys()) == {"customers", "products", "orders", "order_items"}


def test_no_null_revenue(master):
    assert master["revenue"].notna().all()
