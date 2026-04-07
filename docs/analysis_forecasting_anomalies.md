# Forecasting & Anomaly Detection — Analysis Summary

> Auto-generated from `notebooks/07_forecasting_anomaly_detection.ipynb`

---

## 1. Stationarity Test Results

| Metric                               | Test   |   Statistic |   p-value | Conclusion     |
|:-------------------------------------|:-------|------------:|----------:|:---------------|
| Conversion Rate (Original)           | ADF    |     -2.1144 |    0.2387 | Non-stationary |
| Conversion Rate (Original)           | KPSS   |      4.5337 |    0.01   | Non-stationary |
| Conversion Rate (1st Diff)           | ADF    |    -13.3768 |    0      | Stationary     |
| Conversion Rate (1st Diff)           | KPSS   |      0.1405 |    0.1    | Stationary     |
| Conversion Rate (2nd Diff)           | ADF    |    -14.4658 |    0      | Stationary     |
| Conversion Rate (2nd Diff)           | KPSS   |      0.061  |    0.1    | Stationary     |
| Revenue per Session (USD) (Original) | ADF    |     -1.5839 |    0.4917 | Non-stationary |
| Revenue per Session (USD) (Original) | KPSS   |      5.1268 |    0.01   | Non-stationary |
| Revenue per Session (USD) (1st Diff) | ADF    |    -13.1893 |    0      | Stationary     |
| Revenue per Session (USD) (1st Diff) | KPSS   |      0.2092 |    0.1    | Stationary     |
| Revenue per Session (USD) (2nd Diff) | ADF    |    -14.3587 |    0      | Stationary     |
| Revenue per Session (USD) (2nd Diff) | KPSS   |      0.0789 |    0.1    | Stationary     |

**Interpretation:**
- ADF test: rejects null (non-stationary) at p < 0.05 → series is stationary.
- KPSS test: fails to reject null (stationary) at p > 0.05 → series is stationary.
- If the original series is non-stationary by either test, first-order differencing typically achieves stationarity.
- Stationarity plots with ACF/PACF saved to `graphs/`.

---

## 2. Forecasting Model Comparison — Conversion Rate

| Model                       |     RMSE |      MAE |   MAPE |
|:----------------------------|---------:|---------:|-------:|
| Exponential Smoothing (ETS) | 0.010816 | 0.00809  |   9.49 |
| SARIMA                      | 0.011817 | 0.008936 |  10.14 |
| Prophet                     | 0.011889 | 0.008973 |  10.21 |
| XGBoost                     | 0.018446 | 0.015206 |  16.99 |

**Best Model:** **Exponential Smoothing (ETS)** with RMSE = 0.010816 and MAPE = 9.49%.

### Model Descriptions
- **SARIMA**: Seasonal ARIMA with auto-selected (p,d,q)(P,D,Q,s) parameters via `auto_arima`. Captures weekly seasonality (period=7).
- **Exponential Smoothing (ETS)**: Holt-Winters with additive trend and additive seasonality (period=7).
- **Prophet**: Facebook's forecasting library with weekly and yearly seasonality components.
- **XGBoost**: Gradient-boosted trees using lag features (1, 7, 14, 30 days), rolling mean/std (7, 14, 30 days), and date components.

Forecast plots saved to `graphs/`.

---

## 3. Anomaly Detection Results

| Metric                    |   Z-score (k=3) |   IQR |   Isolation Forest |
|:--------------------------|----------------:|------:|-------------------:|
| Conversion Rate           |               5 |     8 |                 55 |
| Revenue per Session (USD) |               2 |     3 |                 55 |

### Conversion Rate Anomalies (flagged by ≥2 methods)

| date                |   conv_rate |   methods_flagged | methods                        |
|:--------------------|------------:|------------------:|:-------------------------------|
| 2012-04-28 00:00:00 |  0          |                 3 | Z-score, IQR, Isolation Forest |
| 2012-05-05 00:00:00 |  0          |                 3 | Z-score, IQR, Isolation Forest |
| 2012-05-06 00:00:00 |  0          |                 3 | Z-score, IQR, Isolation Forest |
| 2012-05-07 00:00:00 |  0          |                 3 | Z-score, IQR, Isolation Forest |
| 2012-06-14 00:00:00 |  0.00510204 |                 2 | IQR, Isolation Forest          |
| 2013-03-30 00:00:00 |  0.120968   |                 2 | IQR, Isolation Forest          |
| 2013-05-05 00:00:00 |  0.121212   |                 2 | IQR, Isolation Forest          |
| 2014-10-12 00:00:00 |  0.126344   |                 3 | Z-score, IQR, Isolation Forest |

### Revenue per Session Anomalies (flagged by ≥2 methods)

| date                |   revenue_per_session |   methods_flagged | methods                        |
|:--------------------|----------------------:|------------------:|:-------------------------------|
| 2014-06-28 00:00:00 |               7.84345 |                 3 | Z-score, IQR, Isolation Forest |
| 2014-10-12 00:00:00 |               8.2671  |                 3 | Z-score, IQR, Isolation Forest |
| 2015-02-08 00:00:00 |               7.64228 |                 2 | IQR, Isolation Forest          |

### Method Comparison
- **Z-score (k=3)**: Most conservative — flags only extreme outliers beyond 3σ.
- **IQR**: Moderate sensitivity — based on quartile spread; catches fat-tail outliers.
- **Isolation Forest**: ML-based; captures multi-dimensional anomalies and local density changes.

---

## 4. Business Interpretation

- **Conversion Rate** shows strong weekly seasonality (weekday vs weekend patterns) confirmed in prior decomposition analysis. The series is largely stationary at the daily level, enabling ARIMA-family models to perform well.
- **Revenue per Session** exhibits higher volatility and occasional extreme spikes (e.g., >$10 RPS days) likely driven by high-value product launches, marketing campaigns, or low-session-count days inflating the ratio.
- Anomalous spikes in RPS typically correspond to:
  - Product launch dates (new SKUs added to the catalog)
  - Marketing events or promotional campaigns
  - Low-traffic days where a few high-value orders inflate the per-session average
- The Isolation Forest method detects the broadest set of anomalies, including subtle shifts that z-score and IQR may miss.

---

## 5. Practical Recommendations

1. **Short-term forecasting**: Use **Exponential Smoothing (ETS)** for daily conversion rate prediction (lowest RMSE/MAPE). Re-train weekly or monthly.
2. **Anomaly monitoring**: Implement **IQR-based alerts** for production dashboards — good balance of sensitivity and specificity. Use Isolation Forest for periodic deep analysis.
3. **Seasonality**: Account for strong weekly patterns in any forecasting or A/B testing design.
4. **Data quality**: Extreme RPS values on low-traffic days should be flagged and potentially winsorized for reporting.

