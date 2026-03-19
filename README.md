# BDEW 2D4 Temperature Profile — Gas Demand Forecasting

A Jupyter notebook that implements the **BDEW 2D4 sigmoid profile formula** for
daily gas demand estimation from temperature, extended with **machine-learning
models** for d+3 and d+7 forward forecasting.

---

## Overview

Gas demand is strongly correlated with temperature. The German energy industry
standard **BDEW 2D4 formula** captures this relationship as a sigmoid:

```
h(T) = A / (1 + (B / (T − T₀))^C) + D
```

| Parameter | Value | Role |
|---|---|---|
| A | 1.0443538 | Amplitude |
| B | −35.0333754 | Shape / inflection |
| C | 6.2240634 | Sigmoid steepness |
| D | 0.0502917 | Floor factor |
| T₀ | 40.0 °C | Reference temperature cap |

Daily demand is then:

```
Demand (MWh/d) = h(T) × Kundenwert
```

where **Kundenwert = 778.5 MWh/d** is the contracted annual daily average.

---

## Contents

| File | Description |
|---|---|
| `temp_profile_2d4.ipynb` | Main notebook — 2D4 formula + ML models |
| `temp.xlsx` | Historical daily temperature data (DWD Chemnitz, 1990–2025) |
| `temp_profile_results.csv` | Exported daily results (date, temp, factor, demand) |
| `image.png` | 2D4 formula reference diagram |

---

## Notebook Structure

### Sections 1–8 — BDEW 2D4 Formula

| Section | Content |
|---|---|
| 1 | Imports & parameters |
| 2 | `profile_2d4(T)` formula definition |
| 3 | Load historical temperature data |
| 4 | Apply formula & normalise |
| 5 | Summary statistics |
| 6 | Charts (curve, scatter, time series, seasonal) |
| 7 | Monthly volume summary |
| 8 | Export to CSV |

### Section 9 — Machine Learning Extension

| Section | Content |
|---|---|
| 9.1 | Feature engineering (temp polynomials, rolling means, cyclical calendar) |
| 9.2 | Train / test split (hold-out: 2024) |
| 9.3 | Train 5 ML models on same-day prediction |
| 9.4 | Performance comparison table |
| 9.5 | Feature importance (Random Forest & Gradient Boosting) |
| 9.6 | Predictions vs 2D4 formula — time series |
| 9.7 | Learned h(T) curves vs 2D4 sigmoid |
| **9.8** | **Lag forecast: d+3 and d+7 — all 5 models trained per lag** |
| 9.8a | Metrics table: MAE, RMSE, R² — all models × both lags |
| 9.8b | Commentary: which model is best, limitations |
| **9.8c** | **d+3 vs d+7: quantitative improvement comparison** |
| 9.9 | Cross-validation (5-fold, time-ordered) |

---

## ML Models

Five models are trained for **each forecast horizon** (d+3 and d+7):

| Model | Type |
|---|---|
| Linear Regression | Linear baseline |
| Ridge Regression | Regularised linear |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) — typically best |
| Neural Network (MLP) | 3-layer ReLU network |

### Features

For each origin day **d**, predicting target day **d+lag**:

| Feature | Source |
|---|---|
| `temp_c`, `temp_c²`, `temp_c³` | Temperature at **d** |
| `roll3`, `roll7`, `roll14` | Rolling mean temp ending at **d** |
| `month_sin/cos`, `doy_sin/cos` | Cyclical encoding of **target date** |
| `is_winter` | Binary flag for Nov–Mar at target date |

---

## Key Findings

- **Same-day prediction**: all ML models achieve R² > 0.99 — they learn the
  deterministic 2D4 sigmoid near-perfectly from temperature alone.
- **d+3 forecast**: temperature autocorrelation at lag-3 is ~0.6–0.8, meaning
  today's temperature is still a meaningful predictor. Models capture both the
  temperature signal and seasonal shape reliably.
- **d+7 forecast**: autocorrelation drops to ~0.3–0.5. Models shift reliance
  toward seasonal calendar features; the temperature nudge adds limited value.
- **Best model for both horizons**: Gradient Boosting — handles non-linearity,
  corrects residuals iteratively, and degrades least at longer horizons.
- **Worst months**: shoulder seasons (Mar–Apr, Oct–Nov) where the sigmoid is
  steepest and temperature uncertainty propagates most to demand error.

## d+3 vs d+7: How Much Better?

Shorter horizon consistently wins across all models and all metrics. The
improvement comes from stronger temperature persistence at 3 days vs 7 days.

| What improves | Why |
|---|---|
| **Lower MAE / RMSE** | Temperature at d is a better proxy for d+3 than d+7 |
| **Higher R²** | More variance explained when temp signal is still strong |
| **Lower bias** | Less seasonal drift to correct over a shorter window |

Key patterns to look for in the metrics (section 9.8c):

- **Tree models (RF, GB)** tend to gain the most from d+3 — they can exploit
  the non-linear temperature signal that is strong at lag-3 but weak at lag-7.
- **Linear models** gain less — they already rely mostly on the polynomial
  temperature terms and seasonal features, which don't change as dramatically
  between the two horizons.
- **Neural Network** gain depends on training convergence — it can match tree
  models at d+3 if the temperature signal is clean enough to learn.
- **In winter** (steep sigmoid): d+3 advantage is largest because a ±2°C
  error at d+3 vs ±5°C at d+7 translates directly to a large demand difference.
- **In summer** (flat sigmoid): both horizons perform similarly — temperature
  barely affects the profile factor, so the horizon gap narrows.

---

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scikit-learn
openpyxl        # for reading temp.xlsx
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

---

## Usage

```bash
git clone https://github.com/dmitrygoryunov2000/weather.git
cd weather
jupyter notebook temp_profile_2d4.ipynb
```

Run all cells in order. The notebook reads `temp.xlsx` from the same directory.

---

## Data

Daily effective temperature data from **DWD (Deutscher Wetterdienst)**,
station Chemnitz-10577, covering **1990-04-01 to 2025-02-25** (12,745 records).
