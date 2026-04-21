# Stock Insights — Technical Documentation

> **Stack:** Python 3 · Flask · yfinance · `ta` · scikit-learn · Prophet · Chart.js · Bootstrap 5  
> **Entry point:** `app.py` — self-contained Flask backend  
> **Frontend:** Jinja2 templates (`chart.html`, `dashboard.html`, `compare.html`, `index.html`)  
> **Author:** N3x · © 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Technical Indicators](#4-technical-indicators)
   - 4.1 [Simple Moving Average — SMA](#41-simple-moving-average--sma)
   - 4.2 [Exponential Moving Average — EMA](#42-exponential-moving-average--ema)
   - 4.3 [Relative Strength Index — RSI](#43-relative-strength-index--rsi)
   - 4.4 [Bollinger Bands](#44-bollinger-bands)
5. [Feature Engineering](#5-feature-engineering)
   - 5.1 [Simple Return](#51-simple-return)
   - 5.2 [Logarithmic Return](#52-logarithmic-return)
   - 5.3 [Rolling Volatility](#53-rolling-volatility)
   - 5.4 [Momentum](#54-momentum)
   - 5.5 [Lag Features](#55-lag-features)
6. [Risk Metrics](#6-risk-metrics)
   - 6.1 [Maximum Drawdown](#61-maximum-drawdown)
   - 6.2 [Sharpe Ratio](#62-sharpe-ratio)
7. [Forecast Evaluation](#7-forecast-evaluation)
   - 7.1 [Mean Absolute Error — MAE](#71-mean-absolute-error--mae)
   - 7.2 [Mean Absolute Percentage Error — MAPE](#72-mean-absolute-percentage-error--mape)
8. [Hybrid Forecasting Model](#8-hybrid-forecasting-model)
   - 8.1 [Prophet Component](#81-prophet-component)
   - 8.2 [ML Component — Ridge Regression](#82-ml-component--ridge-regression)
   - 8.3 [ML Component — Random Forest](#83-ml-component--random-forest)
   - 8.4 [Autoregressive Rollout](#84-autoregressive-rollout)
   - 8.5 [Ensemble Blending Strategy](#85-ensemble-blending-strategy)
9. [Dashboard Metrics](#9-dashboard-metrics)
10. [Comparison View — Frontend Analytics](#10-comparison-view--frontend-analytics)
    - 10.1 [Normalisation Modes](#101-normalisation-modes)
    - 10.2 [Custom Moving Average](#102-custom-moving-average)
    - 10.3 [Pearson Correlation Coefficient](#103-pearson-correlation-coefficient)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Frontend Views](#12-frontend-views)
13. [Configuration & Environment](#13-configuration--environment)
14. [Dependency Map](#14-dependency-map)

---

## 1. Project Overview

**Stock Insights** is an educational, end-to-end stock analytics and forecasting platform. It downloads historical OHLCV data for any Yahoo Finance-listed ticker, computes a suite of technical and statistical indicators, trains a hybrid forecasting model (Prophet + Ridge Regression + Random Forest), and serves all results through a lightweight Flask JSON API consumed by interactive Chart.js dashboards.

**Design principles:**

- *Graceful degradation:* every heavy dependency (`yfinance`, `ta`, `sklearn`, `prophet`) is wrapped in a guarded import. The application starts and serves partial results even when libraries are missing.
- *In-memory caching:* `@lru_cache(maxsize=128)` on the data-fetching layer prevents redundant network calls within a process lifetime.
- *Single-file backend:* the entire server logic lives in `app.py`, making the codebase trivial to read, test, and extend.

---

## 2. Architecture

```
Browser
   │  HTTP GET
   ▼
┌────────────────────────────────────────────────────┐
│                  Flask Router                      │
│  /  ·  /data  ·  /predict  ·  /dashboard_data     │
│  /dashboard  ·  /compare  ·  /docs                │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│         Data Layer  —  fetch_history_cached        │
│   yfinance  ──►  @lru_cache(maxsize=128)           │
└────────────────────────┬───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│        Preparation Layer  —  prepare_df            │
│  TA indicators  +  Feature engineering             │
│  NaN-drop  +  index reset                         │
└──────────────────┬─────────────────┬───────────────┘
                   │                 │
          ┌────────▼──────┐  ┌───────▼──────────────┐
          │  Risk Metrics │  │  Hybrid Forecaster   │
          │  max_drawdown │  │  Prophet             │
          │  sharpe_ratio │  │  Ridge + RandomForest│
          └───────────────┘  └──────────────────────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                      JSON Response
                            │
                            ▼
                ┌───────────────────────┐
                │  Chart.js Dashboards  │
                │  chart · dashboard    │
                │  compare · index      │
                └───────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Fetching — `fetch_history_cached(ticker, period)`

Raw OHLCV data is retrieved from Yahoo Finance through `yfinance.Ticker.history(period=period)`. Results are memoised with Python's `functools.lru_cache` (128-entry capacity) using the `(ticker, period)` tuple as the cache key. The function returns `None` on any failure, leaving all downstream functions responsible for null-guarding.

**Valid period tokens:**

| Token | Description |
|-------|-------------|
| `1d` | Last trading day |
| `5d` | Last 5 days |
| `1mo` | Last calendar month |
| `3mo` / `6mo` | Quarter / half-year |
| `1y` / `2y` / `5y` / `10y` | Yearly windows |
| `ytd` | Year-to-date |
| `max` | Full available history |

### 3.2 Preparation — `prepare_df(ticker, period)`

After fetching, the pipeline performs the following transformations in order:

1. Assert that a `Close` column is present; return `None` otherwise.
2. Reset the DatetimeIndex to a typed `Date` column (`pd.Timestamp`).
3. Compute technical indicators (§4) — wrapped in a `try/except` so the pipeline continues even if the `ta` library is absent.
4. Compute engineered features (§5).
5. Drop all rows containing `NaN` (generated by rolling windows and `shift` operations).
6. Reset the integer index.

> **Window alignment.** The longest rolling window is 20 bars (SMA, EMA, Bollinger Bands). After the `dropna()` step, the returned frame therefore starts at bar 20 of the raw history at the earliest.

---

## 4. Technical Indicators

All indicators are computed on the **closing price** series $\{P_t\}_{t=1}^{T}$ using the `ta` library.

---

### 4.1 Simple Moving Average — SMA

**Column:** `SMA20` · **Window:** $n = 20$

$$
\text{SMA}_{20}(t) = \frac{1}{20} \sum_{i=0}^{19} P_{t-i}
$$

Equal-weight arithmetic mean over the 20-bar lookback. Serves as a **trend filter**: closing prices above $\text{SMA}_{20}$ indicate an uptrend, below indicate a downtrend. Displayed in the chart view as a dashed amber overlay.

---

### 4.2 Exponential Moving Average — EMA

**Column:** `EMA20` · **Window:** $n = 20$

The smoothing factor:

$$
k = \frac{2}{n + 1} = \frac{2}{21} \approx 0.0952
$$

The recursive update rule:

$$
\text{EMA}_{20}(t) = k \cdot P_t + (1 - k) \cdot \text{EMA}_{20}(t-1)
$$

with $\text{EMA}_{20}(1) = P_1$ as initialisation. The EMA down-weights older prices **geometrically**: the weight of observation $P_{t-j}$ decays as $k\,(1-k)^j$. Compared to SMA, it is more reactive to recent price changes. Displayed as a dashed aquamarine overlay.

---

### 4.3 Relative Strength Index — RSI

**Column:** `RSI14` · **Window:** $n = 14$

**Step 1 — Signed price differences:**

$$
\Delta_t = P_t - P_{t-1}
$$

**Step 2 — Windowed average gain and loss** (Wilder smoothing):

$$
\overline{G}(t) = \frac{1}{n} \sum_{i=0}^{n-1} \max(\Delta_{t-i},\; 0)
\qquad
\overline{L}(t) = \frac{1}{n} \sum_{i=0}^{n-1} \max(-\Delta_{t-i},\; 0)
$$

**Step 3 — Relative Strength:**

$$
RS(t) = \frac{\overline{G}(t)}{\overline{L}(t)}
$$

**Step 4 — RSI:**

$$
\text{RSI}_{14}(t) = 100 - \frac{100}{1 + RS(t)} \in [0,\; 100]
$$

The RSI is bounded and unitless. The chart view displays the latest RSI(14) value as an inline indicator chip.

| RSI zone | Conventional interpretation |
|----------|-----------------------------|
| $> 70$ | Overbought — watch for reversal |
| $[30,\; 70]$ | Neutral |
| $< 30$ | Oversold — watch for bounce |

---

### 4.4 Bollinger Bands

**Columns:** `BB_H`, `BB_L` · **Window:** $n = 20$ · **Width:** $k = 2$

Rolling standard deviation of closes over the 20-bar window:

$$
\sigma_{20}(t) = \sqrt{\frac{1}{20} \sum_{i=0}^{19} \bigl(P_{t-i} - \text{SMA}_{20}(t)\bigr)^2}
$$

Upper and lower bands:

$$
\text{BB}_{\text{upper}}(t) = \text{SMA}_{20}(t) + 2\,\sigma_{20}(t)
$$

$$
\text{BB}_{\text{lower}}(t) = \text{SMA}_{20}(t) - 2\,\sigma_{20}(t)
$$

Under the assumption of Gaussian returns, approximately **95%** of closing prices fall within the band. **Band width** $\text{BW}(t) = 4\,\sigma_{20}(t)$ is a direct proxy for realised short-term volatility: a squeeze (narrow band) typically precedes a breakout. Displayed as dashed red/blue envelope overlays in the chart view.

---

## 5. Feature Engineering

These features are computed in `prepare_df` and serve as inputs to the ML forecasting models.

---

### 5.1 Simple Return

**Column:** `Return`

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

Arithmetic rate of change between consecutive closes. Retained in the frame for downstream analysis.

---

### 5.2 Logarithmic Return

**Column:** `LogReturn`

$$
r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

Key mathematical properties:

- **Time-additivity:** the multi-period log return is $r_{[0,T]} = \sum_{t=1}^{T} r_t$.
- **Symmetry:** $+50\%$ and $-33\%$ arithmetic returns do not cancel; the equivalent log returns $+\ln 1.5$ and $-\ln 1.5$ do.
- **First-order approximation:** $r_t \approx R_t$ for small $|R_t|$.

`LogReturn` is the base input for rolling volatility (§5.3).

---

### 5.3 Rolling Volatility

**Column:** `Volatility` · **Window:** $w = 10$

$$
\sigma_{\text{roll}}(t) = \sqrt{\frac{1}{w - 1} \sum_{i=0}^{w-1} \bigl(r_{t-i} - \bar{r}_t\bigr)^2}
$$

where $\bar{r}_t = \frac{1}{w}\sum_{i=0}^{w-1} r_{t-i}$ is the in-window mean of log returns. This is the **sample standard deviation** (Bessel-corrected, denominator $w-1$) of log returns over 10 bars, fed directly into the ML regressors as a regime-state feature.

> **Annualisation:** $\sigma_{\text{annual}} = \sigma_{\text{roll}} \times \sqrt{252}$ converts daily rolling volatility to an annualised figure.

---

### 5.4 Momentum

**Column:** `Momentum10` · **Window:** $w = 10$

$$
M_{10}(t) = \frac{P_t}{P_{t-10}} - 1
$$

This is the **Rate of Change (ROC)** over 10 periods, expressed as a decimal fraction. It captures medium-term trend direction and strength. Positive values indicate a rising price over the window; negative values indicate a declining price.

---

### 5.5 Lag Features

**Columns:** `Lag1`, `Lag2`

$$
\text{Lag1}(t) = P_{t-1}, \qquad \text{Lag2}(t) = P_{t-2}
$$

Autoregressive features encoding **local price memory**. Together with `Momentum10` and `Volatility`, they form the feature vector $\mathbf{x}_t \in \mathbb{R}^4$ used by both Ridge regression and the Random Forest.

---

## 6. Risk Metrics

### 6.1 Maximum Drawdown

**Function:** `max_drawdown(series: pd.Series) → float`

Given a price series $\{P_t\}_{t=1}^{T}$, define the running maximum up to time $t$:

$$
M_t = \max_{s \leq t}\; P_s
$$

The drawdown at time $t$:

$$
DD(t) = \frac{M_t - P_t}{M_t} \in [0, 1]
$$

The **Maximum Drawdown**:

$$
\text{MDD} = \max_{t \in [1,T]}\; DD(t) = \max_{t \in [1,T]}\; \frac{M_t - P_t}{M_t}
$$

Implementation:
```python
cummax   = series.cummax()           # M_t for every t
drawdown = (cummax - series) / cummax
MDD      = float(drawdown.max())
```

MDD measures the **worst peak-to-trough decline** an investor would have experienced entering at the historical peak. A value of $0.25$ means the asset lost 25% from its highest point. The function returns `0.0` for an empty series.

---

### 6.2 Sharpe Ratio

**Function:** `sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) → float`

Assumes a **zero risk-free rate** (no benchmark is subtracted). Given the empirical mean $\mu_R$ and standard deviation $\sigma_R$ of the daily return series:

$$
\text{SR} = \sqrt{T_{\text{year}}} \cdot \frac{\mu_R}{\sigma_R}, \qquad T_{\text{year}} = 252
$$

The $\sqrt{252}$ factor **annualises** the ratio from daily frequency. The Sharpe ratio quantifies risk-adjusted return: how many units of return are earned per unit of total risk.

Implementation:
```python
SR = float(np.sqrt(252) * (ret.mean() / ret.std()))
```

| Sharpe Ratio | Interpretation |
|-------------|----------------|
| $< 0$ | Negative risk-adjusted return |
| $[0,\; 1)$ | Suboptimal |
| $[1,\; 2)$ | Acceptable |
| $[2,\; 3)$ | Good |
| $\geq 3$ | Exceptional |

> **Edge case:** returns `0.0` when $\sigma_R = 0$ (perfectly constant return series), avoiding division by zero.

---

## 7. Forecast Evaluation

**Function:** `evaluate_forecast(y_true, y_pred) → dict`

Both inputs are cast to `np.ndarray(dtype=float)` before computation.

---

### 7.1 Mean Absolute Error — MAE

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

where $y_i$ is the true price and $\hat{y}_i$ the predicted price. MAE is expressed in the **same monetary unit as the price** (e.g. USD), making it directly interpretable. It is robust to outliers compared to MSE because errors enter linearly, not quadratically. Computed via `sklearn.metrics.mean_absolute_error`.

---

### 7.2 Mean Absolute Percentage Error — MAPE

$$
\text{MAPE} = \frac{100}{n} \sum_{\substack{i=1 \\ y_i \neq 0}}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad (\%)
$$

Scale-independent, allowing direct comparison of forecast quality across tickers with vastly different price magnitudes. The implementation **excludes observations where $y_i = 0$** to avoid division by zero, and returns `None` if all actuals are zero.

| MAPE | Forecast quality |
|------|-----------------|
| $< 5\%$ | Highly accurate |
| $[5\%,\; 10\%)$ | Good |
| $[10\%,\; 20\%)$ | Reasonable |
| $\geq 20\%$ | Poor |

---

## 8. Hybrid Forecasting Model

**Function:** `hybrid_forecast(df: pd.DataFrame, days: int = 7) → List[dict]`

A two-component ensemble that attempts Prophet first and gracefully falls back to a pure ML predictor. Each returned record is a dict `{ds, yhat, yhat_lower, yhat_upper}`.

---

### 8.1 Prophet Component

Prophet decomposes the time series **additively**:

$$
P(t) = g(t) + s(t) + \varepsilon(t)
$$

| Term | Name | Description |
|------|------|-------------|
| $g(t)$ | Trend | Piecewise-linear growth with automatic changepoint detection |
| $s(t)$ | Seasonality | Fourier-series approximation of periodic patterns |
| $\varepsilon(t)$ | Error | Assumed i.i.d. Laplace noise |

**Fourier representation of seasonality:**

$$
s(t) = \sum_{n=1}^{N} \left[ a_n \cos\!\left(\frac{2\pi n\, t}{P}\right) + b_n \sin\!\left(\frac{2\pi n\, t}{P}\right) \right]
$$

where $P$ is the period (7 days for weekly seasonality) and $N$ the number of Fourier terms.

**Configuration applied in this application:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `daily_seasonality` | `False` | Daily OHLCV data — intraday pattern irrelevant |
| `weekly_seasonality` | `True` | Captures Mon–Fri market rhythm and weekend gap |
| `yearly_seasonality` | `False` | Disabled to avoid overfitting on short histories |

Prophet outputs a probabilistic forecast:

$$
\hat{P}(t), \quad \hat{P}_{\text{lower}}(t), \quad \hat{P}_{\text{upper}}(t)
$$

---

### 8.2 ML Component — Ridge Regression

**Feature vector** at time $t$:

$$
\mathbf{x}_t = \bigl[\text{Lag1}_t,\; \text{Lag2}_t,\; M_{10}(t),\; \sigma_{\text{roll}}(t)\bigr]^\top \in \mathbb{R}^4
$$

Features are **standardised** before fitting:

$$
\tilde{\mathbf{x}}_t = \frac{\mathbf{x}_t - \boldsymbol{\mu}_X}{\boldsymbol{s}_X}
$$

where $\boldsymbol{\mu}_X$ and $\boldsymbol{s}_X$ are the column-wise mean and standard deviation of the training set.

Ridge (L2-regularised) regression objective:

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \underset{\boldsymbol{\beta}}{\arg\min} \left\{ \sum_{t=1}^{T} \bigl(P_t - \boldsymbol{\beta}^\top \tilde{\mathbf{x}}_t\bigr)^2 + \alpha \|\boldsymbol{\beta}\|_2^2 \right\}, \quad \alpha = 1.0
$$

Closed-form solution:

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \bigl(\tilde{X}^\top \tilde{X} + \alpha I\bigr)^{-1} \tilde{X}^\top \mathbf{y}
$$

The L2 penalty shrinks coefficients toward zero, reducing variance at the cost of a small bias — appropriate given the high multicollinearity between `Lag1`, `Lag2`, and `Close`.

---

### 8.3 ML Component — Random Forest

$$
\hat{P}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}), \qquad B = 100 \text{ trees}, \quad \texttt{random\_state} = 42
$$

Each tree $T_b$ is trained on a **bootstrap sample** of the training set with random feature sub-sampling at each split. The ensemble average reduces variance relative to a single regression tree. `random_state=42` ensures reproducibility.

---

### 8.4 Autoregressive Rollout

The ML forecast is generated **iteratively** for $k = 1, \dots, d$ future steps:

**Per-step ML sub-blend:**

$$
\hat{P}_{T+k}^{\text{ML}} = 0.55 \cdot \hat{P}_{T+k}^{\text{Ridge}} + 0.45 \cdot \hat{P}_{T+k}^{\text{RF}}
$$

After each step, lag features are updated:

$$
\text{Lag2} \leftarrow \text{Lag1}, \qquad \text{Lag1} \leftarrow \hat{P}_{T+k}^{\text{ML}}
$$

`Momentum10` and `Volatility` are held fixed at their last observed values throughout the rollout.

> **Horizon limitation.** Because `Momentum10` and `Volatility` do not update during rollout, forecast uncertainty compounds with horizon. The ML component is most reliable for short horizons ($d \leq 7$ days).

---

### 8.5 Ensemble Blending Strategy

**Decision logic:**

```
Both Prophet AND ML succeeded?
    ├── YES  →  Weighted blend (see formula below)
    ├── Prophet only  →  Prophet output as-is
    ├── ML only  →  ML output as-is
    └── Neither  →  return []
```

**Final blending formula** (when both components are available):

$$
\hat{P}_{T+k}^{\text{hybrid}} = 0.65 \cdot \hat{P}_{T+k}^{\text{Prophet}} + 0.35 \cdot \hat{P}_{T+k}^{\text{ML}}
$$

Confidence intervals come exclusively from Prophet:

$$
\hat{P}_{T+k,\,\text{lower}}^{\text{hybrid}} = \hat{P}_{T+k,\,\text{lower}}^{\text{Prophet}},
\qquad
\hat{P}_{T+k,\,\text{upper}}^{\text{hybrid}} = \hat{P}_{T+k,\,\text{upper}}^{\text{Prophet}}
$$

**Weight rationale:**

| Component | Weight | Justification |
|-----------|--------|--------------|
| Prophet | 0.65 | Accounts for weekly seasonality; provides calibrated uncertainty intervals |
| ML ensemble | 0.35 | Captures recent momentum and autocorrelation through lag features |

The `model` field in the `/predict` JSON response reflects the active strategy: `"hybrid (prophet+ml)"` or `"ml_fallback"`.

---

## 9. Dashboard Metrics

Computed per ticker in `dashboard_data()` — powers the multi-ticker summary cards.

### 9.1 Period Price Variation

$$
\Delta\%_{\text{period}} = \frac{P_T - P_1}{P_1} \times 100
$$

where $P_1$ is the first closing price of the selected period and $P_T$ is the latest. Positive values are displayed with a blue gradient badge (`badge-up`); negative values with an orange gradient badge (`badge-down`).

**Alert flag** (computed in `/predict` only):

$$
\text{alert} = \mathbf{1}\!\left[|\Delta\%_{\text{period}}| \geq \theta\right], \qquad \theta = \texttt{ALERT\_THRESHOLD\_PCT} \;\;(\text{default: } 5\%)
$$

### 9.2 Price Statistics

$$
\bar{P} = \frac{1}{T} \sum_{t=1}^{T} P_t \qquad (\texttt{mean})
$$

$$
s_P = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (P_t - \bar{P})^2} \qquad (\texttt{volatility})
$$

> Note: `volatility` in the dashboard payload is the **price standard deviation** in USD, not the log-return rolling volatility of §5.3.

### 9.3 Baseline Linear Prediction

Simple **Ordinary Least Squares (OLS)** regression on the integer time index $\{1, 2, \dots, T\}$:

$$
\hat{P}(t) = \hat{\beta}_0 + \hat{\beta}_1 \cdot t
$$

Closed-form OLS estimates:

$$
\hat{\beta}_1 = \frac{\displaystyle\sum_{t=1}^{T}(t - \bar{t})(P_t - \bar{P})}{\displaystyle\sum_{t=1}^{T}(t - \bar{t})^2}, \qquad \hat{\beta}_0 = \bar{P} - \hat{\beta}_1\,\bar{t}
$$

Next-bar prediction:

$$
\hat{P}_{T+1} = \hat{\beta}_0 + \hat{\beta}_1 \cdot (T + 1)
$$

This trend-only extrapolation serves as a **sanity baseline**, not a production forecast.

### 9.4 Dashboard Summary Bar

Computed client-side across all $K$ successfully loaded tickers:

$$
\bar{P}_{\text{last}} = \frac{1}{K} \sum_{k=1}^{K} P_{T}^{(k)}, \qquad \bar{s}_{P} = \frac{1}{K} \sum_{k=1}^{K} s_{P}^{(k)}
$$

---

## 10. Comparison View — Frontend Analytics

The comparison view (`compare.html` / `/compare`) implements several analytics **entirely in the browser** via JavaScript, consuming the `/data` endpoint for both tickers.

### 10.1 Normalisation Modes

Three price normalisation modes are offered via radio buttons. Let $P_1$ be the first closing price after date alignment.

**Absolute** (raw prices, no transformation):

$$
\tilde{P}_t^{\text{abs}} = P_t
$$

**Indexed to 100** (base-100 rebasing):

$$
\tilde{P}_t^{\text{idx}} = \frac{P_t}{P_1} \times 100
$$

**Percentage change** from first observation:

$$
\tilde{P}_t^{\text{pct}} = \frac{P_t - P_1}{P_1} \times 100 \quad (\%)
$$

Both indexed and percentage modes normalise from the **first common date** after the two series are date-aligned by inner join on date strings via `alignByDate`.

---

### 10.2 Custom Moving Average

A user-configurable **simple moving average** of the normalised series, with window $w$ (default 5, range 2–∞):

$$
\text{MA}_w(t) = \frac{1}{w} \sum_{i=0}^{w-1} \tilde{P}_{t-i}
$$

For positions $t < w - 1$ (insufficient history), `null` is returned so Chart.js renders a gap rather than a misleading value. Displayed as dashed overlay series alongside the main price lines.

---

### 10.3 Pearson Correlation Coefficient

Computed client-side on the **date-aligned, normalised** price series of both tickers:

$$
r = \frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

where $\{x_i\}$ and $\{y_i\}$ are the normalised closing-price series, $\bar{x}$ and $\bar{y}$ their respective means, and $n$ the number of common dates. Displayed as a floating badge.

$r \in [-1, 1]$:

| $r$ | Interpretation |
|-----|----------------|
| $[0.8,\; 1]$ | Strong positive — assets move together |
| $[0.4,\; 0.8)$ | Moderate positive |
| $(-0.4,\; 0.4)$ | Weak / no linear relationship |
| $(-0.8,\; -0.4]$ | Moderate negative |
| $[-1,\; -0.8)$ | Strong negative — assets move inversely |

The denominator is guarded: if $\sqrt{\sum dx^2 \cdot \sum dy^2} = 0$ (constant series), it is floored to `1` to avoid `NaN`.

---

## 11. API Endpoints Reference

### `GET /`
Renders `chart.html` with `ticker` and `period` injected as Jinja2 variables.  
**Params:** `ticker` (default `AAPL`), `period` (default `1mo`)

---

### `GET /data`
Returns the last 365 rows of historical data with computed indicators.  
**Params:** `ticker`, `period`

```json
[
  {
    "Date":   "2025-01-02",
    "Close":  182.45,
    "SMA20":  180.12,
    "EMA20":  181.33,
    "RSI14":  62.4,
    "BB_H":   188.90,
    "BB_L":   171.34
  }
]
```

**Errors:** `400` invalid period · `404` data unavailable

---

### `GET /predict`
Runs the hybrid forecast and returns predictions + period risk summary.  
**Params:** `ticker`, `period` (default `6mo`), `days` (default `7`)

```json
{
  "ticker":          "AAPL",
  "model":           "hybrid (prophet+ml)",
  "history":         [{ "Date": "2025-01-02", "Close": 182.45 }],
  "forecast":        [{ "ds": "2025-04-22", "yhat": 183.10, "yhat_lower": 178.5, "yhat_upper": 187.7 }],
  "next_prediction": 183.10,
  "variation_pct":   4.32,
  "alert":           false
}
```

**Errors:** `400` · `404`

---

### `GET /dashboard`
Renders `dashboard.html`.  
**Params:** `tickers` (CSV, default `AAPL,MSFT,GOOG,TSLA`), `period`

---

### `GET /dashboard_data`
Returns per-ticker summary statistics for all requested tickers.  
**Params:** `tickers` (CSV), `period`

```json
[
  {
    "ticker":             "AAPL",
    "last":               182.45,
    "mean":               179.23,
    "volatility":         8.54,
    "variation_pct":      3.21,
    "predicted_baseline": 183.10,
    "recent":             [{ "Date": "2025-03-25", "Close": 180.10 }]
  }
]
```

---

### `GET /compare`
Renders `compare.html`.  
**Params:** `ticker1`, `ticker2`, `period`

---

### `GET /docs`
Returns a JSON catalogue of available endpoints.

---

## 12. Frontend Views

| Template | Route | Key features |
|----------|-------|--------------|
| `index.html` | *(landing)* | Ticker search form, shortcut links to dashboard / compare / docs |
| `chart.html` | `/` | Single-ticker candlestick-style chart, SMA/EMA/BB overlays, RSI chip, forecast overlay, CSV download |
| `dashboard.html` | `/dashboard` | Multi-ticker cards with sparklines (30-bar), variation badges, portfolio summary bar |
| `compare.html` | `/compare` | Dual-ticker chart, 3 normalisation modes, custom-window MA overlay, Pearson correlation badge, per-ticker forecast overlays |

**Shared UI stack:** Bootstrap 5.3.2 · Chart.js 4 · Inter (Google Fonts) · dark gradient theme (`#0f2027 → #203a43 → #2c5364`)

---

## 13. Configuration & Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | TCP port Flask binds to |
| `FLASK_DEBUG` / `DEBUG` | `"0"` | Enable debug mode (`"1"` or `"true"`) |
| `ALERT_THRESHOLD_PCT` | `"5.0"` | Minimum absolute variation (%) to set `alert: true` in `/predict` |

The server binds to `0.0.0.0` (all interfaces). All variables are read at startup via `os.environ.get`.

---

## 14. Dependency Map

| Package | Pinned version | Role in app |
|---------|---------------|-------------|
| `Flask` | 3.1.2 | HTTP server, routing, Jinja2 templating |
| `yfinance` | 1.3.0 | Yahoo Finance OHLCV data fetcher |
| `pandas` | 3.0.0 | DataFrame manipulation, rolling windows, `pct_change`, `shift` |
| `numpy` | 2.2.6 | Vectorised arithmetic (`log`, `sqrt`, `arange`, `ndarray`) |
| `ta` | 0.11.0 | Technical indicators: `SMAIndicator`, `EMAIndicator`, `RSIIndicator`, `BollingerBands` |
| `scikit-learn` | 1.8.0 | `Ridge`, `RandomForestRegressor`, `StandardScaler`, `LinearRegression`, `mean_absolute_error` |
| `prophet` | *(optional)* | Additive time-series decomposition and probabilistic forecasting |
| `Flask-SocketIO` | 5.3.6 | Installed — not yet used in `app.py` |
| `Flask-SQLAlchemy` | 3.1.1 | Installed — not yet used in `app.py` |
| `statsmodels` | 0.14.6 | Installed — available for extension |
| `torch` | 2.11.0 | Installed — available for deep-learning extension |

> Prophet is **fully optional**: if it fails to import, `PROPHET_AVAILABLE = False` and the system falls back to the Ridge + RF ML pipeline transparently.
