"""
data_generator.py
-----------------
Generates a synthetic toy store e-commerce dataset that mirrors the schema of
the Maven Analytics "Toy Store E-Commerce Database":

    customers   – customer demographics
    products    – product catalogue with category and price
    orders      – one row per order (customer + date + status)
    order_items – one row per line item (order × product × quantity)

Call `generate_dataset(seed=42)` to get a dict of DataFrames, or
`save_dataset(path, seed=42)` to persist the CSVs to *path*.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CATEGORIES = ["Action Figures", "Board Games", "Dolls", "Puzzles", "Vehicles", "Arts & Crafts"]

_PRODUCTS = [
    ("Action Figure – Hero", "Action Figures", 14.99),
    ("Action Figure – Villain", "Action Figures", 12.99),
    ("Building Blocks Set", "Board Games", 24.99),
    ("Classic Chess", "Board Games", 34.99),
    ("Scrabble Junior", "Board Games", 19.99),
    ("Baby Doll", "Dolls", 22.99),
    ("Fashion Doll", "Dolls", 29.99),
    ("Wooden Doll House", "Dolls", 79.99),
    ("500-Piece Puzzle", "Puzzles", 17.99),
    ("1000-Piece Puzzle", "Puzzles", 22.99),
    ("Mini Race Cars Set", "Vehicles", 18.99),
    ("Remote Control Truck", "Vehicles", 49.99),
    ("Train Set", "Vehicles", 59.99),
    ("Watercolor Kit", "Arts & Crafts", 12.99),
    ("Modeling Clay Pack", "Arts & Crafts", 9.99),
    ("DIY Bracelet Kit", "Arts & Crafts", 8.99),
]

_STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "CO",
]

_ORDER_STATUSES = ["Delivered", "Delivered", "Delivered", "Shipped", "Processing", "Cancelled"]


def _make_customers(rng: np.random.Generator, n: int = 2_000) -> pd.DataFrame:
    customer_ids = [f"CUST{i:05d}" for i in range(1, n + 1)]
    ages = rng.integers(18, 70, size=n)
    genders = rng.choice(["Male", "Female", "Other"], size=n, p=[0.47, 0.47, 0.06])
    states = rng.choice(_STATES, size=n)
    since = pd.Timestamp("2020-01-01")
    days_offset = rng.integers(0, 365 * 3, size=n)
    signup_dates = [since + pd.Timedelta(days=int(d)) for d in days_offset]
    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": genders,
            "state": states,
            "signup_date": signup_dates,
        }
    )


def _make_products() -> pd.DataFrame:
    rows = []
    for i, (name, category, price) in enumerate(_PRODUCTS, start=1):
        rows.append(
            {
                "product_id": f"PROD{i:03d}",
                "product_name": name,
                "category": category,
                "unit_price": price,
            }
        )
    return pd.DataFrame(rows)


def _make_orders(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    n_orders: int = 15_000,
) -> pd.DataFrame:
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2023-12-31")
    span_days = (end - start).days

    # Heavy customers place more orders (Pareto-ish distribution)
    n_customers = len(customers)
    weights = rng.exponential(1.0, size=n_customers)
    weights /= weights.sum()

    order_ids = [f"ORD{i:06d}" for i in range(1, n_orders + 1)]
    customer_ids = rng.choice(customers["customer_id"], size=n_orders, p=weights)
    days_offset = rng.integers(0, span_days, size=n_orders)
    order_dates = [start + pd.Timedelta(days=int(d)) for d in days_offset]
    statuses = rng.choice(_ORDER_STATUSES, size=n_orders)

    return pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_date": order_dates,
            "status": statuses,
        }
    )


def _make_order_items(
    rng: np.random.Generator,
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    product_ids = products["product_id"].values
    # Slight bias toward cheaper products
    price_weights = 1.0 / products["unit_price"].values
    price_weights /= price_weights.sum()

    item_counter = 1
    for _, order in orders.iterrows():
        n_items = int(rng.integers(1, 5))
        chosen = rng.choice(product_ids, size=n_items, replace=False, p=price_weights)
        for prod_id in chosen:
            qty = int(rng.integers(1, 4))
            rows.append(
                {
                    "item_id": f"ITEM{item_counter:07d}",
                    "order_id": order["order_id"],
                    "product_id": prod_id,
                    "quantity": qty,
                }
            )
            item_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Return a dict with keys ``customers``, ``products``, ``orders``,
    ``order_items`` as synthetic DataFrames."""
    rng = np.random.default_rng(seed)
    customers = _make_customers(rng)
    products = _make_products()
    orders = _make_orders(rng, customers)
    order_items = _make_order_items(rng, orders, products)
    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
    }


def save_dataset(path: str | os.PathLike = "data", seed: int = 42) -> None:
    """Generate the dataset and save each table as a CSV under *path*."""
    dataset = generate_dataset(seed)
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in dataset.items():
        out = dest / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"Saved {len(df):,} rows → {out}")
