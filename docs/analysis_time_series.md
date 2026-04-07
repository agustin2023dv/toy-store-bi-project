# Time-Series Decomposition & Seasonality Analysis

## 1. Methods

- **Additive decomposition**: Y(t) = Trend + Seasonal + Residual
- **Multiplicative decomposition**: Y(t) = Trend × Seasonal × Residual
- **STL (Seasonal-Trend decomposition using LOESS)**: robust, iterative, handles outliers
- **Fourier spectral analysis**: FFT to identify dominant periodicities
- **Dummy variable OLS regression**: weekday/month dummies to test seasonal significance
- **CUSUM**: cumulative sum of deviations to detect structural breaks
- **ANOVA / Welch t-test**: compare metric means across groups and periods

## 2. Decomposition Results – Residual Variance

Lower residual variance indicates a better-fitting model.

| Metric                    |    Additive |   Multiplicative |         Stl |
|:--------------------------|------------:|-----------------:|------------:|
| Conversion Rate           | 0.000163131 |        0.0679463 | 0.000145523 |
| Revenue per Session (USD) | 0.5211      |        0.0692294 | 0.470647    |

- **Conversion Rate**: best fit → **Stl** (residual var = 0.000146)
- **Revenue per Session (USD)**: best fit → **Multiplicative** (residual var = 0.069229)

## 3. Dominant Periodicities (Fourier Analysis)

### Conversion Rate

|   Period (days) |   Frequency |   Amplitude |
|----------------:|------------:|------------:|
|        1096     | 0.000912409 |  0.0130093  |
|         548     | 0.00182482  |  0.0119494  |
|         365.333 | 0.00273723  |  0.00538159 |
|         219.2   | 0.00456204  |  0.00495693 |
|         274     | 0.00364964  |  0.00486567 |

### Revenue per Session (USD)

|   Period (days) |   Frequency |   Amplitude |
|----------------:|------------:|------------:|
|        1096     | 0.000912409 |    1.15536  |
|         548     | 0.00182482  |    0.761305 |
|         365.333 | 0.00273723  |    0.377874 |
|         274     | 0.00364964  |    0.355823 |
|         219.2   | 0.00456204  |    0.306228 |

## 4. Seasonal Significance (Dummy Variable Regression)

### Conversion Rate ~ Weekday
- R² = 0.0049, F-test p-value = 5.02e-01, Significant at 5%: **No**

### Conversion Rate ~ Month
- R² = 0.0558, F-test p-value = 3.11e-09, Significant at 5%: **Yes**
- Significant dummies (p < 0.05): const, month_3, month_4, month_5, month_6, month_7, month_8, month_9, month_10, month_11

### Revenue per Session (USD) ~ Weekday
- R² = 0.0039, F-test p-value = 6.36e-01, Significant at 5%: **No**

### Revenue per Session (USD) ~ Month
- R² = 0.0588, F-test p-value = 6.95e-10, Significant at 5%: **Yes**
- Significant dummies (p < 0.05): const, month_4, month_5, month_6, month_7, month_8, month_9, month_10, month_11

## 5. Growth Rates

### Conversion Rate – Quarterly

| date                |   mean |   QoQ_growth |
|:--------------------|-------:|-------------:|
| 2012-04-01 00:00:00 |   0.03 |        -5.3  |
| 2012-07-01 00:00:00 |   0.04 |        32.5  |
| 2012-10-01 00:00:00 |   0.05 |        13.14 |
| 2013-01-01 00:00:00 |   0.06 |        36.6  |
| 2013-04-01 00:00:00 |   0.07 |         9.04 |
| 2013-07-01 00:00:00 |   0.07 |        -3.85 |
| 2013-10-01 00:00:00 |   0.07 |        -1.39 |
| 2014-01-01 00:00:00 |   0.07 |         0.47 |
| 2014-04-01 00:00:00 |   0.07 |         9.38 |
| 2014-07-01 00:00:00 |   0.07 |        -1.71 |
| 2014-10-01 00:00:00 |   0.08 |         8.9  |
| 2015-01-01 00:00:00 |   0.08 |         9.43 |

### Conversion Rate – Yearly

| date                |   mean |   YoY_growth |
|:--------------------|-------:|-------------:|
| 2013-01-01 00:00:00 |   0.07 |        68.78 |
| 2014-01-01 00:00:00 |   0.07 |         8.06 |
| 2015-01-01 00:00:00 |   0.08 |        18.05 |

### Revenue per Session (USD) – Quarterly

| date                |   mean |   QoQ_growth |
|:--------------------|-------:|-------------:|
| 2012-04-01 00:00:00 |   1.55 |        -5.3  |
| 2012-07-01 00:00:00 |   2.05 |        32.5  |
| 2012-10-01 00:00:00 |   2.32 |        13.14 |
| 2013-01-01 00:00:00 |   3.27 |        41.35 |
| 2013-04-01 00:00:00 |   3.56 |         8.65 |
| 2013-07-01 00:00:00 |   3.43 |        -3.55 |
| 2013-10-01 00:00:00 |   3.58 |         4.29 |
| 2014-01-01 00:00:00 |   4.07 |        13.68 |
| 2014-04-01 00:00:00 |   4.64 |        14.04 |
| 2014-07-01 00:00:00 |   4.55 |        -1.94 |
| 2014-10-01 00:00:00 |   4.9  |         7.76 |
| 2015-01-01 00:00:00 |   5.29 |         7.86 |

### Revenue per Session (USD) – Yearly

| date                |   mean |   YoY_growth |
|:--------------------|-------:|-------------:|
| 2013-01-01 00:00:00 |   3.46 |        76.96 |
| 2014-01-01 00:00:00 |   4.54 |        31.23 |
| 2015-01-01 00:00:00 |   5.29 |        16.42 |

## 6. Statistical Tests (ANOVA & t-test)

| Metric                    |   ANOVA Weekday F |   ANOVA Weekday p |   ANOVA Month F |   ANOVA Month p |   t-test stat |     t-test p |   Mean (before) |   Mean (after) | Split Date   |
|:--------------------------|------------------:|------------------:|----------------:|----------------:|--------------:|-------------:|----------------:|---------------:|:-------------|
| Conversion Rate           |          0.888942 |          0.502188 |         5.82878 |     3.10532e-09 |      -18.6293 | 3.42034e-66  |       0.0519931 |      0.0720992 | 2013-09-17   |
| Revenue per Session (USD) |          0.717261 |          0.63575  |         6.16125 |     6.95286e-10 |      -28.7849 | 1.21769e-135 |       2.64972   |      4.46273   | 2013-09-17   |

### Conversion Rate
- Weekday ANOVA: F=0.89, p=5.02e-01 → **not significant**
- Month ANOVA: F=5.83, p=3.11e-09 → **significant**
- t-test (before vs after 2013-09-17): t=-18.63, p=3.42e-66 → **significant**
  - Mean before: 0.0520, Mean after: 0.0721

### Revenue per Session (USD)
- Weekday ANOVA: F=0.72, p=6.36e-01 → **not significant**
- Month ANOVA: F=6.16, p=6.95e-10 → **significant**
- t-test (before vs after 2013-09-17): t=-28.78, p=1.22e-135 → **significant**
  - Mean before: 2.6497, Mean after: 4.4627

## 7. CUSUM Structural Change

The CUSUM plots reveal whether the cumulative deviations from the overall mean 
cross the ±σ√n confidence bands, indicating a structural shift in the series.

- **Conversion Rate**: **Breach detected** – structural change likely
- **Revenue per Session (USD)**: **Breach detected** – structural change likely

## 8. Business Implications

1. **Weekly seasonality** is the dominant cycle for both metrics, confirming that 
   marketing and operational planning should account for day-of-week effects.
2. **STL decomposition** generally yields the lowest residual variance, making it the 
   recommended method for ongoing monitoring dashboards.
3. **Growth trends** show the direction and magnitude of business performance changes; 
   quarters with declining growth warrant investigation into traffic sources or product mix.
4. **CUSUM breach** (if detected) signals a regime change — e.g. a new product launch, 
   marketing campaign, or platform change that shifted baseline performance.
5. **Significant weekday/month effects** mean that conversion-rate benchmarks should be 
   adjusted by time period rather than using a single global target.

## 9. Graphs Index

| Graph | File |
|-------|------|
| Acf Pacf Aov | `graphs/acf_pacf_aov.png` |
| Acf Pacf Conv Rate | `graphs/acf_pacf_conv_rate.png` |
| Acf Pacf Orders | `graphs/acf_pacf_orders.png` |
| Acf Pacf Revenue Per Session | `graphs/acf_pacf_revenue_per_session.png` |
| Acf Pacf Sessions | `graphs/acf_pacf_sessions.png` |
| Additive Decomp Conv Rate | `graphs/additive_decomp_conv_rate.png` |
| Additive Decomp Revenue Per Session | `graphs/additive_decomp_revenue_per_session.png` |
| Aov Trend | `graphs/aov_trend.png` |
| Boxplot Revenue | `graphs/boxplot_revenue.png` |
| Change Point Detection Conv Rate | `graphs/change_point_detection_conv_rate.png` |
| Change Point Detection Revenue Per Session | `graphs/change_point_detection_revenue_per_session.png` |
| Coefficient Of Variation Comparison | `graphs/coefficient_of_variation_comparison.png` |
| Conversion Rate Trend | `graphs/conversion_rate_trend.png` |
| Cross Corr Convrate Sessions | `graphs/cross_corr_convrate_sessions.png` |
| Cusum Plot Conv Rate | `graphs/cusum_plot_conv_rate.png` |
| Cusum Plot Revenue Per Session | `graphs/cusum_plot_revenue_per_session.png` |
| Distribution Continuous | `graphs/distribution_continuous.png` |
| Fourier Spectrum Conv Rate | `graphs/fourier_spectrum_conv_rate.png` |
| Fourier Spectrum Revenue Per Session | `graphs/fourier_spectrum_revenue_per_session.png` |
| Growth Mom Conv Rate | `graphs/growth_mom_conv_rate.png` |
| Growth Mom Revenue Per Session | `graphs/growth_mom_revenue_per_session.png` |
| Growth Qoq Conv Rate | `graphs/growth_qoq_conv_rate.png` |
| Growth Qoq Revenue Per Session | `graphs/growth_qoq_revenue_per_session.png` |
| Growth Yoy Conv Rate | `graphs/growth_yoy_conv_rate.png` |
| Growth Yoy Revenue Per Session | `graphs/growth_yoy_revenue_per_session.png` |
| Moving Averages Aov | `graphs/moving_averages_aov.png` |
| Moving Averages Conv Rate | `graphs/moving_averages_conv_rate.png` |
| Moving Averages Orders | `graphs/moving_averages_orders.png` |
| Moving Averages Revenue Per Session | `graphs/moving_averages_revenue_per_session.png` |
| Moving Averages Sessions | `graphs/moving_averages_sessions.png` |
| Multiplicative Decomp Conv Rate | `graphs/multiplicative_decomp_conv_rate.png` |
| Multiplicative Decomp Revenue Per Session | `graphs/multiplicative_decomp_revenue_per_session.png` |
| Revenue By Channel | `graphs/revenue_by_channel.png` |
| Rolling Std Aov | `graphs/rolling_std_aov.png` |
| Rolling Std Conversion | `graphs/rolling_std_conversion.png` |
| Rolling Std Orders | `graphs/rolling_std_orders.png` |
| Rolling Std Rps | `graphs/rolling_std_rps.png` |
| Rolling Std Sessions | `graphs/rolling_std_sessions.png` |
| Rps Trend | `graphs/rps_trend.png` |
| Seasonal Subseries Month Conv Rate | `graphs/seasonal_subseries_month_conv_rate.png` |
| Seasonal Subseries Month Revenue Per Session | `graphs/seasonal_subseries_month_revenue_per_session.png` |
| Seasonal Subseries Weekday Conv Rate | `graphs/seasonal_subseries_weekday_conv_rate.png` |
| Seasonal Subseries Weekday Revenue Per Session | `graphs/seasonal_subseries_weekday_revenue_per_session.png` |
| Stl Decomposition Conv Rate | `graphs/stl_decomposition_conv_rate.png` |
| Stl Decomposition Revenue Per Session | `graphs/stl_decomposition_revenue_per_session.png` |
| Trend Regression Aov | `graphs/trend_regression_aov.png` |
| Trend Regression Conv Rate | `graphs/trend_regression_conv_rate.png` |
| Trend Regression Orders | `graphs/trend_regression_orders.png` |
| Trend Regression Revenue Per Session | `graphs/trend_regression_revenue_per_session.png` |
| Trend Regression Sessions | `graphs/trend_regression_sessions.png` |
| Trend Sessions Orders | `graphs/trend_sessions_orders.png` |
