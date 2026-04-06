# Toy Store E-Commerce – End-to-End BI Project

End-to-end Business Intelligence project using a toy store e-commerce dataset inspired by the
[Maven Analytics Toy Store E-Commerce Database](https://mavenanalytics.io/data-playground/toy-store-e-commerce-database).

The project covers the full BI workflow: synthetic data generation, data loading and feature engineering,
exploratory data analysis with **seaborn**, and **machine-learning models** (scikit-learn) for customer
segmentation, revenue forecasting, and cancellation prediction.

---

## Project Structure

```
toy-store-bi-project/
├── data/                        # Generated CSV files (auto-created on first run)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset overview, KPIs, schema inspection
│   ├── 02_eda_seaborn.ipynb        # EDA visualisations (seaborn)
│   └── 03_ml_models.ipynb          # Customer segmentation, forecasting, cancellation
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # Synthetic dataset generator
│   ├── data_loader.py           # CSV loader + feature engineering
│   ├── eda.py                   # Seaborn plot helpers
│   └── ml_models.py             # ML classes (K-Means, RandomForest, LogisticRegression)
├── tests/
│   ├── test_data_generator.py
│   ├── test_data_loader.py
│   ├── test_eda.py
│   └── test_ml_models.py
├── requirements.txt
└── README.md
```

---

## Dataset Schema

The project uses a four-table relational schema:

| Table | Description |
|-------|-------------|
| `customers` | Customer demographics (ID, age, gender, US state, signup date) |
| `products` | Product catalogue (ID, name, category, unit price) |
| `orders` | One row per order (order ID, customer ID, date, status) |
| `order_items` | Line items (item ID, order ID, product ID, quantity) |

A **master** table is built by joining all four tables and adding engineered features:
`revenue`, `order_year`, `order_month`, `order_quarter`, `is_cancelled`.

> **Note:** If you have the real Maven Analytics CSV files, place them in the `data/` directory
> with the matching names. Otherwise the project auto-generates a synthetic dataset on first run.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the notebooks

```bash
cd notebooks
jupyter notebook
```

Open the notebooks in order:
1. `01_data_exploration.ipynb` – dataset overview and KPIs
2. `02_eda_seaborn.ipynb` – exploratory data analysis
3. `03_ml_models.ipynb` – machine learning models

### 3. Run the tests

```bash
pytest tests/ -v
```

---

## Machine Learning Models

### Customer Segmentation (`CustomerSegmentation`)
Uses **K-Means clustering** on RFM (Recency, Frequency, Monetary) features to group customers
into behavioural segments such as Champions, At-Risk, and Hibernating customers.

```python
from src.data_loader import load_or_generate, build_master
from src.ml_models import CustomerSegmentation

tables = load_or_generate('data')
master = build_master(tables)

seg = CustomerSegmentation(n_clusters=4)
seg.fit(master)
print(seg.silhouette_score_)
print(seg.segment_summary())
seg.plot_segments()
```

### Revenue Forecasting (`RevenueForecaster`)
A **RandomForestRegressor** trained on monthly aggregates with lag and rolling-mean features
to forecast future revenue.

```python
from src.ml_models import RevenueForecaster

forecaster = RevenueForecaster(n_estimators=200)
forecaster.fit(master)
print(forecaster.evaluate())   # {'mae': ..., 'r2': ...}
forecaster.plot_forecast()
```

### Cancellation Prediction (`CancellationPredictor`)
A **Logistic Regression** model that predicts whether an order will be cancelled based on
product price, quantity, customer age, order month, and product category.

```python
from src.ml_models import CancellationPredictor

predictor = CancellationPredictor()
predictor.fit(master)
print(predictor.evaluate())   # classification report dict
predictor.plot_coefficients()
```

---

## EDA Visualisations

All plot functions are in `src/eda.py` and accept an optional `ax` parameter for embedding
in multi-panel figures, and an optional `save_path` to persist the figure as a PNG file.

| Function | Description |
|----------|-------------|
| `plot_monthly_revenue` | Line chart of monthly total revenue |
| `plot_revenue_by_category` | Horizontal bar chart by product category |
| `plot_top_products` | Top-N products by revenue |
| `plot_order_status_distribution` | Count-plot of order statuses |
| `plot_age_distribution` | Histogram + KDE of customer ages |
| `plot_revenue_heatmap` | Revenue heatmap (month × year) |
| `plot_customer_state_revenue` | Revenue by US state (top-N) |
| `plot_correlation_matrix` | Pearson correlation heatmap |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data wrangling | pandas, numpy |
| Visualisation | seaborn, matplotlib |
| Machine Learning | scikit-learn (KMeans, RandomForestRegressor, LogisticRegression) |
| Notebooks | Jupyter |
| Testing | pytest |

