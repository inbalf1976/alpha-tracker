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
            if (LOG_DIR / f).exists():
                (LOG_DIR / f).unlink()
        if ERROR_LOG_PATH.exists():
            ERROR_LOG_PATH.unlink()
        with open(ERROR_LOG_PATH, 'w') as f:
            json.dump([], f)
        return True
    except: return False

reset_all_logs_on_startup()

def setup_logging():
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'))

    file_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'))

    error_handler = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.info("=== NEW SESSION STARTED ===")
    return logger

logger = setup_logging()

def log_error(severity, function_name, error, ticker=None, user_message="An error occurred", show_to_user=True):
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.value,
        "ticker": ticker,
        "function": function_name,
        "error": str(error),
        "user_message": user_message
    }
    if severity == ErrorSeverity.ERROR:
        logger.error(f"{ticker or 'N/A'} | {user_message}", exc_info=True)
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(f"{ticker or 'N/A'} | {user_message}", exc_info=True)
    else:
        logger.warning(f"{ticker or 'N/A'} | {user_message}")

    try:
        history = json.load(open(ERROR_LOG_PATH)) if ERROR_LOG_PATH.exists() else []
        history.append(error_data)
        json.dump(history[-500:], open(ERROR_LOG_PATH, 'w'), indent=2)
    except: pass

    if show_to_user and st._is_running_with_streamlit:
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"CRITICAL: {user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"ERROR: {user_message}")

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
    "Commodities": {"Gold Futures": "GC=F", "Crude Oil": "CL=F"},
    "ETFs": {"S&P 500 ETF": "SPY"}
}

MODEL_DIR = Path("models"); SCALER_DIR = Path("scalers"); ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata"); PREDICTIONS_DIR = Path("predictions"); BACKTEST_DIR = Path("backtests")
for p in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, BACKTEST_DIR, LOG_DIR]:
    p.mkdir(exist_ok=True)

LEARNING_CONFIG = {
    "lookback_window": 60,
    "accuracy_threshold": 0.065,  # now 6.5%
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

# ================================
# UPGRADE 1: HIGH CONFIDENCE CHECKLIST
# ================================
def high_confidence_checklist(ticker: str, forecast: list, current_price: float) -> tuple[bool, list]:
    reasons = []
    metadata = load_metadata(ticker)
    acc_log = load_accuracy_log(ticker)
    price = current_price or get_latest_price(ticker)

    if acc_log.get("total_predictions", 0) < 12:
        reasons.append(f"Only {acc_log['total_predictions']} live preds")
    if metadata.get("retrain_count", 0) < 2:
        reasons.append("Model needs ≥2 retrains")
    if acc_log.get("avg_error", 0.99) > 0.065:
        reasons.append(f"Recent error {acc_log['avg_error']:.1%}")
    if metadata.get("trained_date"):
        days_old = (datetime.now() - datetime.fromisoformat(metadata["trained_date"])).days
        if days_old > 14:
            reasons.append(f"Model {days_old}d old")
    try:
        df = yf.download(ticker, period="60d", progress=False, threads=False)
        if len(df) > 0 and isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=1) if df.columns.nlevels > 1 else df['Close']
        vol = df.pct_change().rolling(20).std().iloc[-1]
        if vol > 0.04 and days_old > 7:
            reasons.append(f"High vol {vol:.1%}/day")
    except: pass

    implied_move = abs(forecast[0] - price) / price
    if implied_move > 0.12:
        reasons.append(f"Extreme move {implied_move:+.1%}")

    return len(reasons) == 0, reasons

# ================================
# UPGRADE 2: LOG RETURNS + RSI + VOLUME FEATURES
# ================================
def create_features(df):
    df = df.copy()
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volume_norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df[['Return', 'Volume_norm', 'RSI']].dropna()
    return df

# ================================
# MODEL TRAINING (now uses features)
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)

    try:
        df = yf.download(ticker, period="2y", progress=False, threads=False)
        if df.empty or len(df) < 200:
            return None, None, None
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
            y.append(scaled[i, 0])  # predict next return
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 3)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        metadata = load_metadata(ticker)
        needs_retrain, _ = should_retrain(ticker, load_accuracy_log(ticker), metadata)

        if needs_retrain or force_retrain or not model_path.exists():
            model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
            metadata["retrain_count"] = metadata.get("retrain_count", 0) + 1
        else:
            model = tf.keras.models.load_model(str(model_path))
            recent_X = X[-int(len(X)*0.3):]
            recent_y = y[-int(len(y)*0.3):]
            model.fit(recent_X, recent_y, epochs=LEARNING_CONFIG["fine_tune_epochs"], batch_size=32, verbose=0)

        model.save(str(model_path))

        # Forecast returns → convert back to price
        last_seq = scaled[-lookback:]
        preds = []
        seq = last_seq.copy()
        for _ in range(days):
            pred = model.predict(seq.reshape(1, lookback, 3), verbose=0)[0,0]
            preds.append(pred)
            new_row = np.array([pred, seq[-1,1], seq[-1,2]]).reshape(1,3)
            seq = np.vstack([seq[1:], new_row])

        # Inverse transform returns → cumulative price
        return_preds = np.array(preds).reshape(-1,1)
        dummy = np.zeros((len(return_preds), 3))
        dummy[:,0] = return_preds.flatten()
        price_moves = scaler.inverse_transform(dummy)[:,0]
        current = df['Close'].iloc[-1]
        forecast_prices = [current]
        for r in np.exp(price_moves) - 1:
            forecast_prices.append(forecast_prices[-1] * (1 + r))

        dates = [(datetime.now() + timedelta(days=i+1)).date() for i in range(days)]
        dates = [d for i, d in enumerate(dates) if (datetime.now().date() + timedelta(days=i+1)).weekday() < 5][:days]

        metadata["trained_date"] = datetime.now().isoformat()
        metadata["version"] = metadata.get("version", 0) + 1
        save_metadata(ticker, metadata)

        tf.keras.backend.clear_session()
        return forecast_prices[1:], dates, model

    except Exception as e:
        log_error(ErrorSeverity.CRITICAL, "train_self_learning_model", e, ticker=ticker)
        return None, None, None

# ================================
# UPGRADE 3: FULL BACKTESTING MODULE
# ================================
def run_backtest(ticker):
    with st.spinner(f"Backtesting {ticker} (this takes 2–5 minutes)..."):
        progress = st.progress(0)
        df = yf.download(ticker, period="3y", progress=False)
        if df.empty: 
            st.error("No data")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data = create_features(df)

        lookback = LEARNING_CONFIG["lookback_window"]
        accuracy_log = {"predictions": [], "errors": [], "dates": [], "total_predictions": 0}

        for i in range(lookback + 100, len(data) - 5, 5):  # walk forward
            progress.progress((i - lookback - 100) / (len(data) - lookback - 105))

            train_data = data.iloc[:i]
            scaler = MinMaxScaler().fit(train_data)
            scaled = scaler.transform(train_data)

            X_train, y_train = [], []
            for j in range(lookback, len(scaled)):
                X_train.append(scaled[j-lookback:j])
                y_train.append(scaled[j, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 3)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

            # Predict next day
            last_seq = scaler.transform(data.iloc[i-lookback:i])
            pred_return = model.predict(last_seq.reshape(1, lookback, 3), verbose=0)[0,0]

            actual_price = df['Close'].iloc[i]
            pred_price = df['Close'].iloc[i-1] * np.exp(pred_return)

            error = abs(pred_price - actual_price) / actual_price
            accuracy_log["errors"].append(error)
            accuracy_log["total_predictions"] += 1
            accuracy_log["avg_error"] = np.mean(accuracy_log["errors"][-30:])

        save_accuracy_log(ticker, accuracy_log)
        st.success(f"Backtest complete! Final error: {np.mean(accuracy_log['errors']):.2%}")
        st.metric("Backtested Predictions", accuracy_log["total_predictions"])
        st.metric("Final Avg Error", f"{np.mean(accuracy_log['errors']):.2%}")

# ================================
# REST OF YOUR ORIGINAL FUNCTIONS (shortened for space)
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False, threads=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except: pass
    return None

def load_accuracy_log(t): 
    p = get_accuracy_path(t)
    return json.load(open(p)) if p.exists() else {"predictions": [], "errors": [], "total_predictions": 0, "avg_error": 0.99}
def save_accuracy_log(t, d): 
    with open(get_accuracy_path(t), 'w') as f: json.dump(d, f, indent=2)
def load_metadata(t): 
    p = get_metadata_path(t)
    return json.load(open(p)) if p.exists() else {"trained_date": None, "version": 0, "retrain_count": 0}
def save_metadata(t, d): 
    with open(get_metadata_path(t), 'w') as f: json.dump(d, f, indent=2)
def should_retrain(ticker, acc_log, meta):
    if acc_log.get("avg_error", 0.99) > LEARNING_CONFIG["accuracy_threshold"]:
        return True, ["Poor accuracy"]
    if meta["trained_date"]:
        days = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
        if days > 21:
            return True, ["Stale model"]
    return False, []

# ================================
# MAIN APP
# ================================
st.set_page_config(page_title="AI Alpha Tracker v5 – High Confidence", layout="wide")
st.markdown("<h1 style='text-align:center;color:#00C853;'>AI ALPHA TRACKER v5 – HIGH CONFIDENCE MODE</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Assets")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    if st.button("RUN BACKTEST (3 years)"):
        run_backtest(ticker)

    if st.button("Force Retrain + Forecast"):
        with st.spinner("Training..."):
            forecast, dates, _ = train_self_learning_model(ticker, days=5, force_retrain=True)
            st.session_state.forecast = (forecast, dates)

col1, col2 = st.columns([1, 2])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(f"<h2 style='text-align:center;'>LIVE: ${price:.2f}</h2>", unsafe_allow_html=True)

    if st.button("5-Day High-Confidence Forecast", use_container_width=True):
        with st.spinner("Thinking..."):
            forecast, dates, _ = train_self_learning_model(ticker, days=5, force_retrain=False)
            if forecast:
                passed, reasons = high_confidence_checklist(ticker, forecast, price)
                if passed:
                    st.success("HIGH CONFIDENCE PREDICTION")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(color='#00C853', width=4)))
                    fig.add_hline(y=price, line_dash="dash", line_color="orange")
                    fig.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("PREDICTION WITHHELD – Not confident enough")
                    st.write("Reasons:", " • ".join(reasons[:4]))
