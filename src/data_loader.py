"""
data_loader.py
--------------
Loads the toy store dataset from CSV files (or generates it on-the-fly if the
files are absent) and returns a single enriched DataFrame called the *master*
table used by both the EDA and ML modules.

Public functions
~~~~~~~~~~~~~~~~
load_tables(data_dir)  – returns dict of raw DataFrames
build_master(tables)   – joins & engineers features; returns master DataFrame
load_or_generate(data_dir, seed) – convenience wrapper
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd


def load_tables(data_dir: str | os.PathLike = "data") -> Dict[str, pd.DataFrame]:
    """Load the four CSV tables from *data_dir*. Raises FileNotFoundError if
    any table is missing."""
    data_dir = Path(data_dir)
    tables = {}
    for name in ("customers", "products", "orders", "order_items"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected {path} – run data_generator.save_dataset() first.")
        tables[name] = pd.read_csv(path, parse_dates=["order_date"] if name == "orders" else False)
    # Parse signup_date separately to avoid touching other tables
    if "signup_date" in tables["customers"].columns:
        tables["customers"]["signup_date"] = pd.to_datetime(tables["customers"]["signup_date"])
    return tables


def build_master(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join all tables and engineer features.

    Returns one row per *order_item* enriched with order, customer, and
    product information plus derived columns:
        revenue        – unit_price × quantity
        order_year     – year of order_date
        order_month    – month of order_date
        order_quarter  – quarter of order_date
        is_cancelled   – bool flag
    """
    oi = tables["order_items"].copy()
    orders = tables["orders"].copy()
    products = tables["products"].copy()
    customers = tables["customers"].copy()

    master = (
        oi
        .merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    master["revenue"] = master["unit_price"] * master["quantity"]
    master["order_date"] = pd.to_datetime(master["order_date"])
    master["order_year"] = master["order_date"].dt.year
    master["order_month"] = master["order_date"].dt.month
    master["order_quarter"] = master["order_date"].dt.quarter
    master["is_cancelled"] = master["status"] == "Cancelled"

    return master


def load_or_generate(
    data_dir: str | os.PathLike = "data",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Return tables dict, generating synthetic CSVs if they don't exist yet."""
    data_dir = Path(data_dir)
    required = [data_dir / f"{n}.csv" for n in ("customers", "products", "orders", "order_items")]
    if not all(p.exists() for p in required):
        from src.data_generator import save_dataset
        save_dataset(data_dir, seed=seed)
    return load_tables(data_dir)
