"""
tests/test_data_generator.py
-----------------------------
Unit tests for src/data_generator.py
"""

import pandas as pd
import pytest

from src.data_generator import generate_dataset, save_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(seed=0)


def test_dataset_keys(dataset):
    assert set(dataset.keys()) == {"customers", "products", "orders", "order_items"}


def test_customers_schema(dataset):
    df = dataset["customers"]
    assert set(df.columns) >= {"customer_id", "age", "gender", "state", "signup_date"}
    assert len(df) > 0


def test_products_schema(dataset):
    df = dataset["products"]
    assert set(df.columns) >= {"product_id", "product_name", "category", "unit_price"}
    assert (df["unit_price"] > 0).all()


def test_orders_schema(dataset):
    df = dataset["orders"]
    assert set(df.columns) >= {"order_id", "customer_id", "order_date", "status"}
    assert len(df) > 0


def test_order_items_schema(dataset):
    df = dataset["order_items"]
    assert set(df.columns) >= {"item_id", "order_id", "product_id", "quantity"}
    assert (df["quantity"] >= 1).all()


def test_no_duplicate_customer_ids(dataset):
    ids = dataset["customers"]["customer_id"]
    assert ids.nunique() == len(ids)


def test_no_duplicate_order_ids(dataset):
    ids = dataset["orders"]["order_id"]
    assert ids.nunique() == len(ids)


def test_reproducibility():
    d1 = generate_dataset(seed=7)
    d2 = generate_dataset(seed=7)
    pd.testing.assert_frame_equal(d1["orders"], d2["orders"])


def test_save_dataset(tmp_path):
    save_dataset(tmp_path, seed=1)
    for name in ("customers", "products", "orders", "order_items"):
        assert (tmp_path / f"{name}.csv").exists()
