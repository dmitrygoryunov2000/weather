# BDEW 2D4 Temperature Profile — Gas Demand Forecasting

A Jupyter notebook that implements the **BDEW 2D4 sigmoid profile formula** for
daily gas demand estimation from temperature, extended with **machine-learning
models** for d+7 and d+14 forward forecasting.

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
| **9.8** | **Lag forecast: d+7 and d+14 — all 5 models trained per lag** |
| 9.8a | Metrics table: MAE, RMSE, R² — all models × both lags |
| 9.8b | Commentary: which model is best, limitations |
| 9.9 | Cross-validation (5-fold, time-ordered) |

---

## ML Models

Five models are trained for **each forecast horizon** (d+7 and d+14):

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
- **d+7 forecast**: R² drops significantly as temperature persistence over
  7 days is weak (~0.3–0.5 autocorrelation). Models rely on seasonal calendar
  features more than the temperature signal.
- **d+14 forecast**: temperature persistence is largely noise. Performance
  approaches a climatological baseline.
- **Best model for d+7**: Gradient Boosting — handles non-linearity, corrects
  residuals, and degrades least at longer horizons.
- **Worst months**: shoulder seasons (Mar–Apr, Oct–Nov) where the sigmoid is
  steepest and temperature uncertainty propagates most to demand error.

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
