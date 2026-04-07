# Autocorrelation Insights

## What ACF/PACF Tell Us

- **ACF (Autocorrelation Function):** Shows how today's metric value correlates with values at previous lags. Slowly decaying ACF indicates a trend or long-memory process.
- **PACF (Partial Autocorrelation Function):** Isolates the direct effect of each lag after removing intermediate effects. Sharp cutoff at lag *p* suggests an AR(p) model.

## Implications for Forecasting
- Significant ACF at many lags means the series has trend/seasonality that must be differenced before modelling.
- PACF cutoff lag suggests the AR order for ARIMA models.
- If PACF is significant at lag 1 only, it indicates an AR(1) process (today depends mostly on yesterday).
- If PACF is significant at lags 1 and 7, it indicates a weekly seasonality component.
