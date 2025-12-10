"""
Minimal, robust Flask app for the test folder. Rewritten to be clearer, safer, and
easier to extend. This file intentionally keeps the same public endpoints used by
the templates while improving validation, error handling, and readability.

Notes:
- This file is a self-contained refactor to make the endpoints predictable.
- It still depends on yfinance, pandas, numpy, ta and scikit-learn for modeling.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from datetime import timedelta
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

# Optional heavy imports guarded to avoid ImportError at import time
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    # TA (technical analysis) helpers
    from ta.trend import SMAIndicator, EMAIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
except Exception:
    SMAIndicator = EMAIndicator = RSIIndicator = BollingerBands = None

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
except Exception:
    LinearRegression = Ridge = RandomForestRegressor = StandardScaler = mean_absolute_error = None

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
ALERT_THRESHOLD_PCT = float(os.environ.get("ALERT_THRESHOLD_PCT", "5.0"))


def json_error(message: str, code: int = 400) -> Any:
    return jsonify({"error": message}), code


@lru_cache(maxsize=128)
def fetch_history_cached(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch historical data via yfinance and cache results in-memory.

    Returns None on failure or if yfinance is not available.
    """
    if yf is None:
        logging.error("yfinance is not installed or failed to import")
        return None
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df is None or df.empty:
            return None
        return df.copy()
    except Exception as e:
        logging.exception("fetch error")
        return None


def prepare_df(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Download and prepare a DataFrame with indicators and common features.

    The returned DataFrame always has a `Date` column of pandas.Timestamp and a
    `Close` column.
    """
    df = fetch_history_cached(ticker, period)
    if df is None or df.empty:
        return None

    # Ensure Close exists
    if "Close" not in df.columns:
        return None

    # Basic indicators only if TA lib available
    df2 = df.copy()
    df2 = df2.reset_index()
    if "Date" not in df2.columns:
        df2["Date"] = pd.to_datetime(df2.index)
    df2["Date"] = pd.to_datetime(df2["Date"])

    try:
        if SMAIndicator is not None:
            df2["SMA20"] = SMAIndicator(df2["Close"], window=20).sma_indicator()
            df2["EMA20"] = EMAIndicator(df2["Close"], window=20).ema_indicator()
            df2["RSI14"] = RSIIndicator(df2["Close"], window=14).rsi()
            bb = BollingerBands(df2["Close"], window=20)
            df2["BB_H"] = bb.bollinger_hband()
            df2["BB_L"] = bb.bollinger_lband()
    except Exception:
        logging.exception("Indicator computation failed; continuing without them")

    # Feature engineering (robust)
    df2["Return"] = df2["Close"].pct_change()
    df2["LogReturn"] = np.log(df2["Close"] / df2["Close"].shift(1))
    df2["Volatility"] = df2["LogReturn"].rolling(window=10).std()
    df2["Momentum10"] = df2["Close"] / df2["Close"].shift(10) - 1
    df2["Lag1"] = df2["Close"].shift(1)
    df2["Lag2"] = df2["Close"].shift(2)
    df2 = df2.dropna().reset_index(drop=True)
    return df2


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cummax = series.cummax()
    drawdown = (cummax - series) / cummax
    return float(drawdown.max())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ret = returns.dropna()
    if ret.empty or float(ret.std()) == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * (ret.mean() / ret.std()))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    if y_true_arr.size == 0:
        return {"MAE": None, "MAPE_pct": None}
    mae = None
    try:
        if mean_absolute_error is not None:
            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    except Exception:
        logging.exception("MAE calculation failed")
        mae = None
    # MAPE safe
    non_zero = y_true_arr != 0
    if non_zero.sum() == 0:
        mape = None
    else:
        mape = float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100)
    return {"MAE": mae, "MAPE_pct": mape}


def hybrid_forecast(df: pd.DataFrame, days: int = 7) -> List[Dict[str, Optional[float]]]:
    """A pragmatic hybrid forecast: try Prophet then fall back to a simple ML baseline.

    The function returns a list of days (dicts with ds,yhat, yhat_lower, yhat_upper).
    """
    if df is None or df.empty:
        return []

    # Try Prophet first (if available)
    prophet_list = None
    if PROPHET_AVAILABLE and Prophet is not None:
        try:
            dfm = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
            dfm["ds"] = pd.to_datetime(dfm["ds"])
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(dfm)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
            pred["ds"] = pred["ds"].dt.strftime("%Y-%m-%d")
            prophet_list = pred.to_dict(orient="records")
        except Exception:
            logging.exception("Prophet failed; falling back to ML")
            prophet_list = None

    # ML fallback: linear trend or simple Ridge+RF if features available
    ml_list = None
    try:
        if {"Lag1", "Lag2", "Momentum10", "Volatility"}.issubset(df.columns) and Ridge is not None:
            X = df[["Lag1", "Lag2", "Momentum10", "Volatility"]].values
            y = df["Close"].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            ridge = Ridge(alpha=1.0).fit(Xs, y)
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xs, y)
            last = df.iloc[-1:].copy()
            preds = []
            for _ in range(days):
                Xf = scaler.transform(last[["Lag1", "Lag2", "Momentum10", "Volatility"]].values)
                p = 0.55 * ridge.predict(Xf)[0] + 0.45 * rf.predict(Xf)[0]
                preds.append(float(p))
                # shift lags
                last["Lag2"] = last["Lag1"]
                last["Lag1"] = p
            base_date = pd.to_datetime(df["Date"].iloc[-1])
            ml_list = [{"ds": (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"), "yhat": preds[i], "yhat_lower": None, "yhat_upper": None} for i in range(days)]
    except Exception:
        logging.exception("ML fallback failed")
        ml_list = None

    # Combine: prefer Prophet if available; otherwise ML; otherwise empty
    if prophet_list and ml_list:
        # simple blend
        out = []
        for p, m in zip(prophet_list, ml_list):
            yhat = 0.65 * p.get("yhat", 0) + 0.35 * m.get("yhat", 0)
            lower = p.get("yhat_lower")
            upper = p.get("yhat_upper")
            out.append({"ds": p.get("ds"), "yhat": float(yhat), "yhat_lower": float(lower) if lower is not None else None, "yhat_upper": float(upper) if upper is not None else None})
        return out
    if prophet_list:
        return prophet_list
    if ml_list:
        return ml_list
    return []


@app.route("/")
def home() -> Any:
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    period = request.args.get("period", "1mo").strip()
    return render_template("chart.html", ticker=ticker, period=period)


@app.route("/data")
def data() -> Any:
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    period = request.args.get("period", "1mo").strip()
    if period not in VALID_PERIODS:
        return json_error("Invalid period", 400)
    df = prepare_df(ticker, period)
    if df is None or df.empty:
        return json_error("Data not available", 404)
    out = df[["Date", "Close", "SMA20", "EMA20", "RSI14", "BB_H", "BB_L"]].tail(365).copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return jsonify(out.to_dict(orient="records"))


@app.route("/predict")
def predict_route() -> Any:
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    period = request.args.get("period", "6mo").strip()
    days = int(request.args.get("days", 7))
    if period not in VALID_PERIODS:
        return json_error("Invalid period", 400)
    df = prepare_df(ticker, period)
    if df is None or df.empty:
        return json_error("Data not available", 404)
    forecast = hybrid_forecast(df, days=days)
    next_pred = forecast[-1]["yhat"] if forecast else None
    # variation on period safely
    try:
        recent = float(df["Close"].iloc[-1])
        past = float(df["Close"].iloc[0])
        variation_pct = None
        alert = False
        if past != 0:
            variation_pct = ((recent - past) / past) * 100
            alert = abs(variation_pct) >= ALERT_THRESHOLD_PCT
    except Exception:
        variation_pct = None
        alert = False

    return jsonify({
        "ticker": ticker,
        "model": "hybrid (prophet+ml)" if PROPHET_AVAILABLE else "ml_fallback",
        "history": df[["Date", "Close"]].tail(90).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "forecast": forecast,
        "next_prediction": round(float(next_pred), 2) if next_pred is not None else None,
        "variation_pct": round(float(variation_pct), 2) if variation_pct is not None else None,
        "alert": bool(alert),
    })


@app.route("/dashboard")
def dashboard_page() -> Any:
    tickers = request.args.get("tickers", "AAPL,MSFT,GOOG,TSLA")
    period = request.args.get("period", "1mo")
    return render_template("dashboard.html", tickers=tickers, period=period)


@app.route("/dashboard_data")
def dashboard_data() -> Any:
    tickers_param = request.args.get("tickers", "AAPL,MSFT,GOOG,TSLA")
    period = request.args.get("period", "1mo")
    tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    results: List[Dict[str, Any]] = []
    for t in tickers:
        df = prepare_df(t, period)
        if df is None or df.empty:
            continue
        last_close = float(df["Close"].iloc[-1])
        mean_price = float(df["Close"].mean())
        std_price = float(df["Close"].std())
        variation = None
        try:
            pv = float(df["Close"].iloc[0])
            if pv != 0:
                variation = ((df["Close"].iloc[-1] - pv) / pv) * 100
        except Exception:
            variation = None

        # baseline linear prediction
        next_pred = None
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df["Close"].values
            if LinearRegression is not None:
                lr = LinearRegression().fit(X, y)
                next_pred = float(lr.predict([[len(df) + 1]])[0])
        except Exception:
            next_pred = None

        results.append({
            "ticker": t,
            "last": round(last_close, 2),
            "mean": round(mean_price, 2),
            "volatility": round(std_price, 2),
            "variation_pct": round(float(variation), 2) if variation is not None else None,
            "predicted_baseline": round(next_pred, 2) if next_pred else None,
            "recent": df[["Date", "Close"]].tail(30).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        })
    return jsonify(results)


@app.route("/compare")
def compare_page() -> Any:
    ticker1 = request.args.get("ticker1", "AAPL").upper().strip()
    ticker2 = request.args.get("ticker2", "MSFT").upper().strip()
    period = request.args.get("period", "1mo")
    return render_template("compare.html", ticker1=ticker1, ticker2=ticker2, period=period)


@app.route("/docs")
def docs() -> Any:
    doc = {
        "endpoints": {
            "/": "Main UI (chart)",
            "/data?ticker=..&period=..": "Historical data + indicators",
            "/predict?ticker=..&period=..&days=..": "Hybrid forecast (prophet+ml)",
            "/backtest?ticker=..&period=..&window=..": "Rolling backtest",
            "/dashboard_data?tickers=AAPL,MSFT&period=..": "Multi-ticker summary",
        },
        "notes": "Prophet is optional; if missing the code falls back to a ML-based predictor.",
    }
    return jsonify(doc)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_flag = os.environ.get("FLASK_DEBUG", os.environ.get("DEBUG", "0"))
    debug = str(debug_flag).lower() in ("1", "true")
    logging.info(f"Starting test app on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
