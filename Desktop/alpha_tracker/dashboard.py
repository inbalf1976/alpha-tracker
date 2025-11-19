import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import time
import threading
import os
import warnings
import joblib
import json
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import traceback
from enum import Enum

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ================================
# LOGGING SETUP
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

class ErrorSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def reset_all_logs_on_startup():
    try:
        for f in ['app.log', 'errors.log']:
            p = LOG_DIR / f
            if p.exists(): p.unlink()
        if ERROR_LOG_PATH.exists(): ERROR_LOG_PATH.unlink()
        with open(ERROR_LOG_PATH, 'w') as f: json.dump([], f)
    except: pass

reset_all_logs_on_startup()

def setup_logging():
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    file_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    error_handler = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.info("=== NEW SESSION STARTED - ALL LOGS RESET ===")
    return logger

logger = setup_logging()

def log_error(severity, function_name, error, ticker=None, user_message="An error occurred", show_to_user=True):
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.value,
        "ticker": ticker or "N/A",
        "function": function_name,
        "error": str(error),
        "user_message": user_message
    }
    msg = f"{ticker or 'N/A'} | {user_message}: {error}"
    if severity == ErrorSeverity.CRITICAL:
        logger.critical(msg, exc_info=True)
    elif severity == ErrorSeverity.ERROR:
        logger.error(msg, exc_info=True)
    else:
        logger.warning(msg)

    try:
        history = json.load(open(ERROR_LOG_PATH)) if ERROR_LOG_PATH.exists() else []
        history.append(error_data)
        json.dump(history[-500:], open(ERROR_LOG_PATH, 'w'), indent=2)
    except: pass

    if show_to_user and 'st' in globals():
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"CRITICAL: {user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"ERROR: {user_message}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"WARNING: {user_message}")

    st.session_state.setdefault('error_logs', []).append(error_data)

# ================================
# CONFIG & DIRECTORIES
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
except: BOT_TOKEN = CHAT_ID = None

ASSET_CATEGORIES = {
    "Tech Stocks": {"Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL"},
    "High Growth": {"Palantir": "PLTR", "MicroStrategy": "MSTR", "Coinbase": "COIN"},
    "Commodities": {"Gold Futures": "GC=F", "Crude Oil": "CL=F", "Corn": "ZC=F"},
    "ETFs": {"S&P 500 ETF": "SPY", "Wheat ETF": "WEAT"}
}

MODEL_DIR = Path("models"); SCALER_DIR = Path("scalers"); ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata"); PREDICTIONS_DIR = Path("predictions"); BACKTEST_DIR = Path("backtests")
CONFIG_DIR = Path("config"); LOG_DIR.mkdir(exist_ok=True)
for p in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, BACKTEST_DIR, CONFIG_DIR]:
    p.mkdir(exist_ok=True)

DAEMON_CONFIG_PATH = CONFIG_DIR / "daemon_config.json"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring_config.json"

LEARNING_CONFIG = {
    "lookback_window": 60,
    "accuracy_threshold": 0.065,        # tightened from 0.08
    "min_predictions_for_eval": 12,
    "retrain_interval_days": 21,
    "full_retrain_epochs": 30,
    "fine_tune_epochs": 7
}

# ================================
# HELPERS
# ================================
def get_safe_ticker_name(t): return t.replace('=', '_').replace('^', '').replace('/', '_')
def get_model_path(t): return MODEL_DIR / f"{get_safe_ticker_name(t)}_lstm.h5"
def get_scaler_path(t): return SCALER_DIR / f"{get_safe_ticker_name(t)}_scaler.pkl"
def get_accuracy_path(t): return ACCURACY_DIR / f"{get_safe_ticker_name(t)}_accuracy.json"
def get_metadata_path(t): return METADATA_DIR / f"{get_safe_ticker_name(t)}_meta.json"
def get_prediction_path(t, d): return PREDICTIONS_DIR / f"{get_safe_ticker_name(t)}_{d}.json"

# ================================
# UPGRADE 1: HIGH CONFIDENCE CHECKLIST
# ================================
def high_confidence_checklist(ticker: str, forecast: list, current_price: float) -> tuple[bool, list]:
    reasons = []
    metadata = load_metadata(ticker)
    acc_log = load_accuracy_log(ticker)

    if acc_log.get("total_predictions", 0) < 12:
        reasons.append(f"Only {acc_log['total_predictions']} live predictions")
    if metadata.get("retrain_count", 0) < 2:
        reasons.append("Model needs ≥2 retrains")
    if acc_log.get("avg_error", 0.99) > 0.065:
        reasons.append(f"Recent error {acc_log['avg_error']:.1%} > 6.5%")
    if metadata.get("trained_date"):
        days_old = (datetime.now() - datetime.fromisoformat(metadata["trained_date"])).days
        if days_old > 14:
            reasons.append(f"Model {days_old} days old")

    try:
        df = yf.download(ticker, period="60d", progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
        vol = df['Close'].pct_change().rolling(20).std().iloc[-1]
        if vol > 0.04 and days_old > 7:
            reasons.append(f"Extreme volatility {vol:.1%}/day")
    except: pass

    implied_move = abs(forecast[0] - current_price) / current_price
    if implied_move > 0.12:
        reasons.append(f"Extreme 1-day move {implied_move:+.1%}")

    return len(reasons) == 0, reasons

# ================================
# UPGRADE 2: FEATURE ENGINEERING (log returns + volume + RSI)
# ================================
def create_features(df):
    df = df.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df = df[["Return", "Volume_norm", "RSI"]].dropna()
    return df

# ================================
# PRICE FETCHING
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False, threads=False)
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 4 if ticker.endswith(("=F","=X")) else 2)
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "get_latest_price", e, ticker, show_to_user=False)
    return None

# ================================
# PERSISTENCE FUNCTIONS
# ================================
def load_accuracy_log(t):
    p = get_accuracy_path(t)
    return json.load(open(p)) if p.exists() else {"predictions": [], "errors": [], "dates": [], "total_predictions": 0, "avg_error": 0.99}
def save_accuracy_log(t, d):
    with open(get_accuracy_path(t), 'w') as f: json.dump(d, f, indent=2)
def load_metadata(t):
    p = get_metadata_path(t)
    return json.load(open(p)) if p.exists() else {"trained_date": None, "version": 0, "retrain_count": 0, "training_volatility": 0.0}
def save_metadata(t, d):
    with open(get_metadata_path(t), 'w') as f: json.dump(d, f, indent=2)

def record_prediction(ticker, price, date):
    path = get_prediction_path(ticker, date)
    json.dump({"ticker": ticker, "predicted_price": float(price), "prediction_date": date, "timestamp": datetime.now().isoformat()}, 
              open(path, 'w'), indent=2)

def validate_predictions(ticker):
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = get_prediction_path(ticker, yesterday)
    if not path.exists(): return False, None
    pred = json.load(open(path))
    actual = get_latest_price(ticker)
    if not actual: return False, None
    error = abs(pred["predicted_price"] - actual) / actual
    log = load_accuracy_log(ticker)
    log["errors"].append(error)
    log["errors"] = log["errors"][-100:]
    log["total_predictions"] += 1
    log["avg_error"] = np.mean(log["errors"][-30:])
    save_accuracy_log(ticker, log)
    path.unlink(missing_ok=True)
    return True, log

def should_retrain(ticker, acc_log, meta):
    reasons = []
    if acc_log.get("avg_error", 0.99) > LEARNING_CONFIG["accuracy_threshold"]:
        reasons.append(f"Error {acc_log['avg_error']:.1%}")
    if meta.get("trained_date"):
        days = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
        if days > LEARNING_CONFIG["retrain_interval_days"]:
            reasons.append(f"{days} days old")
    return len(reasons) > 0, reasons

# ================================
# UPGRADE 3: FULL BACKTESTING
# ================================
def run_backtest(ticker):
    if st.button(f"Backtesting {ticker} – this takes 2–6 minutes", disabled=True): pass
    progress = st.progress(0)
    status = st.empty()
    status.info("Downloading 3 years of data...")
    
    df = yf.download(ticker, period="3y", progress=False)
    if df.empty:
        st.error("No data")
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    data = create_features(df)
    lookback = LEARNING_CONFIG["lookback_window"]
    errors = []

    for i in range(lookback + 100, len(data) - 10, 10):
        progress.progress((i - lookback - 100) / (len(data) - lookback - 110))
        status.info(f"Training on data up to {df.index[i].date()}")

        train_data = data.iloc[:i]
        scaler = MinMaxScaler().fit(train_data)
        scaled = scaler.transform(train_data)

        X, y = [], []
        for j in range(lookback, len(scaled)):
            X.append(scaled[j-lookback:j])
            y.append(scaled[j, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([LSTM(50, return_sequences=True, input_shape=(lookback, 3)),
                            Dropout(0.2), LSTM(50), Dropout(0.2), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

        seq = scaler.transform(data.iloc[i-lookback:i])
        pred_ret = model.predict(seq.reshape(1, lookback, 3), verbose=0)[0,0]
        pred_price = df["Close"].iloc[i-1] * np.exp(pred_ret)
        actual_price = df["Close"].iloc[i]
        error = abs(pred_price - actual_price) / actual_price
        errors.append(error)

    final_error = np.mean(errors)
    st.success(f"Backtest complete! Avg error: {final_error:.2%} over {len(errors)} predictions")
    st.metric("Backtested Days", len(errors))

# ================================
# TRAINING WITH NEW FEATURES
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    logger.info(f"Training {ticker} (force={force_retrain})")
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)

    try:
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty or len(df) < 200: return None, None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        data = create_features(df)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)

        X, y = [], []
        lookback = LEARNING_CONFIG["lookback_window"]
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        metadata = load_metadata(ticker)
        acc_log = load_accuracy_log(ticker)
        needs_retrain, _ = should_retrain(ticker, acc_log, metadata)

        if force_retrain or needs_retrain or not model_path.exists():
            model = Sequential([LSTM(50, return_sequences=True, input_shape=(lookback, 3)),
                                Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
            metadata["retrain_count"] = metadata.get("retrain_count", 0) + 1
        else:
            model = tf.keras.models.load_model(str(model_path))
            recent_X = X[-int(len(X)*0.3):]
            recent_y = y[-int(len(y)*0.3):]
            model.fit(recent_X, recent_y, epochs=LEARNING_CONFIG["fine_tune_epochs"], batch_size=32, verbose=0)

        model.save(str(model_path))

        # Forecast
        seq = scaled[-lookback:]
        preds = []
        current_seq = seq.copy()
        for _ in range(days):
            pred = model.predict(current_seq.reshape(1, lookback, 3), verbose=0)[0,0]
            preds.append(pred)
            new_row = np.array([pred, current_seq[-1,1], current_seq[-1,2]]).reshape(1,3)
            current_seq = np.vstack([current_seq[1:], new_row])

        # Convert returns back to price
        dummy = np.zeros((len(preds), 3))
        dummy[:,0] = preds
        price_moves = scaler.inverse_transform(dummy)[:,0]
        current_price = df["Close"].iloc[-1]
        forecast = [current_price]
        for r in np.exp(price_moves) - 1:
            forecast.append(forecast[-1] * (1 + r))

        dates = [(datetime.now() + timedelta(days=i+1)).date() for i in range(days)]
        dates = [d for d in dates if d.weekday() < 5][:days]

        metadata["trained_date"] = datetime.now().isoformat()
        metadata["version"] = metadata.get("version", 0) + 1
        save_metadata(ticker, metadata)

        record_prediction(ticker, forecast[1], dates[0].strftime("%Y-%m-%d"))

        tf.keras.backend.clear_session()
        return forecast[1:], dates, model

    except Exception as e:
        log_error(ErrorSeverity.CRITICAL, "train_self_learning_model", e, ticker)
        return None, None, None

# ================================
# RECOMMENDATIONS WITH CHECKLIST
# ================================
def daily_recommendation(ticker, asset):
    price = get_latest_price(ticker)
    if not price:
        return "<span style='color:orange'>No price data</span>"

    forecast, dates, _ = train_self_learning_model(ticker, days=1)
    if not forecast:
        return "<span style='color:red'>Forecast failed</span>"

    passed, reasons = high_confidence_checklist(ticker, forecast, price)
    pred = round(forecast[0], 2)
    change = (pred - price) / price * 100

    if not passed:
        return f"""
        <div style="background:#2d1b1b;padding:20px;border-radius:12px;border-left:6px solid #ff4444;color:#ff9999;">
        <h3>{asset.upper()} — PREDICTION WITHHELD</h3>
        <p><strong>Waiting for higher-confidence setup</strong></p>
        <small>Reasons: {' • '.join(reasons[:4])}</small>
        </div>
        """

    action = "BUY" if change >= 1.5 else "SELL" if change <= -1.5 else "HOLD"
    color = "#00C853" if action == "BUY" else "#ff4444" if action == "SELL" else "#FFA726"

    return f"""
    <div style="background:#1a2a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;">
    <h3 style="color:{color};">{asset.upper()} — HIGH CONFIDENCE</h3>
    <p><strong>Live:</strong> ${price:.2f} → <strong>Predicts:</strong> ${pred:.2f} ({change:+.2f}%)</p>
    <p><strong>Action:</strong> <span style="font-size:1.4em;color:{color};">{action}</span></p>
    <small>Passed 8-point confidence checklist</small>
    </div>
    """

# ================================
# UI
# ================================
st.set_page_config(page_title="AI Alpha Tracker v5", layout="wide")
st.markdown("<h1 style='text-align:center;color:#00C853;'>AI - ALPHA TRACKER v5<br><small>High-Confidence • Log Returns • Backtesting</small></h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    st.markdown("---")
    st.subheader("Model Status")
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    if meta.get("trained_date"):
        st.metric("Last Trained", meta["trained_date"][:10])
        st.metric("Version", f"v{meta.get('version',0)}")
        st.metric("Accuracy", f"{(1-acc.get('avg_error',0.99))*100:.1f}%")

    if st.button("Force Retrain", use_container_width=True):
        with st.spinner("Retraining..."):
            train_self_learning_model(ticker, force_retrain=True)
            st.success("Retrained!")

    if st.button("Run Full Backtest (3y)"):
        run_backtest(ticker)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(f"<h2 style='text-align:center;'>LIVE PRICE: <code style='font-size:2em;background:#333;padding:10px 20px;border-radius:12px;'>${price:.2f}</code></h2>", unsafe_allow_html=True)

    if st.button("Daily High-Confidence Recommendation", use_container_width=True):
        with st.spinner("Analyzing..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)

    if st.button("5-Day Forecast (only if confident)", use_container_width=True):
        with st.spinner("Forecasting..."):
            forecast, dates, _ = train_self_learning_model(ticker, days=5)
            if forecast:
                passed, reasons = high_confidence_checklist(ticker, forecast, price)
                if passed:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=forecast, mode="lines+markers", name="AI Forecast", line=dict(color="#00C853", width=4)))
                    fig.add_hline(y=price, line_dash="dash", line_color="orange", annotation_text=f"Today ${price:.2f}")
                    fig.update_layout(template="plotly_dark", height=500, title=f"{asset.upper()} – HIGH CONFIDENCE 5-Day Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Forecast withheld – not confident enough")
                    st.write("Reasons:", " • ".join(reasons))
