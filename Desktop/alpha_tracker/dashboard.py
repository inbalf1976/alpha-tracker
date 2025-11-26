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
from enum import Enum
from typing import Tuple, List
import functools
import threading 

# Alpha Vantage key (add to secrets.toml or environment variable)
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")

@st.cache_data(ttl=45, show_spinner=False)
def get_latest_price_robust(ticker: str):
    """100% uptime price fetcher: Yahoo → Alpha Vantage → last resort"""
    # Step 1: Try Yahoo Finance (fastest)
    for attempt in range(2):
        try:
            data = yf.download(ticker, period="2d", interval="1m", progress=False)
            data = normalize_dataframe_columns(data)
            if not data.empty and len(data) > 1:
                p = float(data['Close'].iloc[-1])
                if validate_price(ticker, p):
                    return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
        except:
            time.sleep(0.5)

    # Step 2: Alpha Vantage fallback
    if ALPHA_VANTAGE_KEY:
        try:
            # Map common futures/crypto
            symbol_map = {
                "GC=F": "XAUUSD", "CL=F": "WTICOUSD", "ZC=F": "CORN", "ZW=F": "WHEAT",
                "BTC-USD": "BTCUSD", "ETH-USD": "ETHUSD"
            }
            av_symbol = symbol_map.get(ticker, ticker.replace("=F", "").replace("^", ""))

            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_symbol}&apikey={ALPHA_VANTAGE_KEY}"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                p = float(data["Global Quote"]["05. price"])
                if validate_price(ticker, p):
                    logger.info(f"Alpha Vantage fallback used for {ticker}: ${p:.2f}")
                    return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
        except Exception as e:
            logger.debug(f"Alpha Vantage failed: {e}")

    # Step 3: Last resort Yahoo info endpoint
    try:
        info = yf.Ticker(ticker).info
        p = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if p and validate_price(ticker, p):
            return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
    except:
        pass

    log_error(ErrorSeverity.WARNING, "get_latest_price_robust", Exception("All sources failed"), ticker=ticker, show_to_user=False)
    return None

# ───────────────────────────────
# 1. YFINANCE 401 + TIMEOUT FIX – FINAL WORKING VERSION (2025)
# ───────────────────────────────
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin
from requests import Session
from pyrate_limiter import Duration, Limiter

class LimitedCachedSession(CacheMixin, LimiterMixin, Session):
    pass

# 2 requests per 1 second = 2 req/sec max (perfect for Yahoo)
session = LimitedCachedSession(
    limiter=Limiter(2, Duration.SECOND),
    bucket="yfinance",
    expire_after=300,
    backend="sqlite"
)

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
})

# GLOBAL PATCH – ALL yfinance calls now use our armored session
yf.shared._SESSION = session
yf.shared._DFS = {}
yf.shared._BASE_URL_ = "https://query2.finance.yahoo.com"

# Monkey-patch download & Ticker to always use session
yf.download = functools.partial(yf.download, session=session, auto_adjust=True, progress=False, threads=False)
yf.Ticker = lambda ticker, *args, **kwargs: yf.Ticker(ticker, session=session, *args, **kwargs)

# Suppress TF & warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Suppress Streamlit thread warnings once and for all
try:
    from streamlit.runtime.scriptrunner import script_run_context
    if hasattr(script_run_context, '_LOGGER'):
        script_run_context._LOGGER.setLevel('ERROR')
except:
    pass

# ================================
# LIVE HYBRID PATTERN MINER INTEGRATION (unchanged)
# ================================
AUTO_PATTERNS_FILE = Path("auto_patterns.json")
def check_auto_patterns(ticker: str, data: pd.DataFrame = None) -> tuple:
    if not AUTO_PATTERNS_FILE.exists():
        return 0, [], "NEUTRAL", 0
    try:
        raw = json.loads(AUTO_PATTERNS_FILE.read_text(encoding="utf-8"))
        if "patterns" not in raw:
            return 0, [], "NEUTRAL", 0
        ticker_clean = ticker.replace('=F', '').replace('^', '').split('.')[0].upper()
        now = datetime.now()
        best_match = None
        best_auc = 0
        for pat in raw["patterns"]:
            if pat.get("ticker", "").upper() != ticker_clean:
                continue
            try:
                pat_time = datetime.strptime(pat.get("timestamp", ""), "%Y-%m-%d %H:%M")
                age_hours = (now - pat_time).total_seconds() / 3600
                if age_hours > 6:
                    continue
                freshness_factor = 1.0 if age_hours < 1 else 0.8 if age_hours < 3 else 0.6
            except:
                continue
            auc = pat.get("auc_mean", 0) * freshness_factor
            if auc > best_auc:
                best_auc = auc
                best_match = pat
        if not best_match:
            return 0, [], "NEUTRAL", 0
        boost = int(best_match.get("boost", 0) * freshness_factor)
        bias = best_match.get("direction_bias", "NEUTRAL")
        direction = "DOWN" if bias == "DOWN" else "UP" if bias == "UP" else "NEUTRAL"
        confidence = min(99, int(best_match.get("auc_mean", 0) * 100 + boost // 2.5))
        triggers = [
            f"{best_match.get('model', 'unknown').upper()} AUC {best_match.get('auc_mean', 0):.3f}",
            f"Boost +{boost}",
            f"{best_match.get('timeframe', 'unknown').upper()} ELITE",
            f"Bias {bias}"
        ]
        return boost, triggers, direction, confidence
    except Exception as e:
        try:
            log_error(ErrorSeverity.WARNING, "check_auto_patterns", e, ticker=ticker, show_to_user=False)
        except:
            pass
        return 0, [], "NEUTRAL", 0

# ================================
# LOGGING & ERROR SYSTEM (THREAD-SAFE FIX)
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
            (LOG_DIR / f).unlink(missing_ok=True)
        ERROR_LOG_PATH.write_text("[]")
    except: pass
reset_all_logs_on_startup()

def setup_logging():
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    if hasattr(console.stream, 'reconfigure'):
        try: console.stream.reconfigure(encoding='utf-8')
        except: pass
    file = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    error = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    error.setLevel(logging.ERROR)
    error.setFormatter(file.formatter)
    logger.addHandler(console)
    logger.addHandler(file)
    logger.addHandler(error)
    logger.info("=== NEW SESSION STARTED - LOGS RESET ===")
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
    if severity == ErrorSeverity.DEBUG: logger.debug(user_message)
    elif severity == ErrorSeverity.INFO: logger.info(user_message)
    elif severity == ErrorSeverity.WARNING: logger.warning(user_message)
    elif severity == ErrorSeverity.ERROR: logger.error(user_message, exc_info=True)
    elif severity == ErrorSeverity.CRITICAL: logger.critical(user_message, exc_info=True)

    try:
        history = json.loads(ERROR_LOG_PATH.read_text()) if ERROR_LOG_PATH.exists() else []
        history.append(error_data)
        ERROR_LOG_PATH.write_text(json.dumps(history[-500:], indent=2))
    except: pass

    # ONLY SHOW st. messages in main thread
    if show_to_user and 'st' in globals():
        try:
            if threading.current_thread() is threading.main_thread():
                if severity == ErrorSeverity.CRITICAL: st.error(f"CRITICAL {user_message}")
                elif severity == ErrorSeverity.ERROR: st.error(f"ERROR {user_message}")
                elif severity == ErrorSeverity.WARNING: st.warning(f"WARNING {user_message}")
        except:
            pass

    try:
        if hasattr(st, 'session_state'):
            st.session_state.setdefault('error_logs', []).append(error_data)
    except: pass

def get_error_statistics():
    try:
        if not ERROR_LOG_PATH.exists(): return {"total": 0, "by_severity": {}, "recent": []}
        errors = json.loads(ERROR_LOG_PATH.read_text())
        by_sev = {}
        for e in errors:
            s = e.get('severity', 'UNKNOWN')
            by_sev[s] = by_sev.get(s, 0) + 1
        return {"total": len(errors), "by_severity": by_sev, "recent": errors[-10:]}
    except: return {"total": 0, "by_severity": {}, "recent": []}

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
    "Commodities": {"Corn Futures": "ZC=F", "Gold Futures": "GC=F", "Crude Oil": "CL=F", "Wheat": "ZW=F"},
    "ETFs": {"S&P 500 ETF": "SPY", "WHEAT": "WEAT"}
}

for d in ["models", "scalers", "accuracy_logs", "metadata", "predictions", "config", "logs", "backtest_results"]:
    Path(d).mkdir(exist_ok=True)

DAEMON_CONFIG_PATH = Path("config") / "daemon_config.json"
MONITORING_CONFIG_PATH = Path("config") / "monitoring_config.json"

LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,
    "min_predictions_for_eval": 10,
    "retrain_interval_days": 30,
    "volatility_change_threshold": 0.5,
    "fine_tune_epochs": 5,
    "full_retrain_epochs": 25,
    "lookback_window": 60
}

# Threading locks & heartbeats
model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()
config_lock = threading.Lock()
session_state_lock = threading.Lock()
heartbeat_lock = threading.Lock()

# Per-ticker training locks (prevents concurrent training of same ticker)
TRAINING_LOCKS = {}
TRAINING_LOCKS_LOCK = threading.Lock()

def get_training_lock(ticker):
    """Get or create a lock for a specific ticker"""
    with TRAINING_LOCKS_LOCK:
        if ticker not in TRAINING_LOCKS:
            TRAINING_LOCKS[ticker] = threading.Lock()
        return TRAINING_LOCKS[ticker]

THREAD_HEARTBEATS = {"learning_daemon": None, "monitoring": None, "watchdog": None}
THREAD_START_TIMES = {"learning_daemon": None, "monitoring": None, "watchdog": None}

def update_heartbeat(name):
    with heartbeat_lock: THREAD_HEARTBEATS[name] = datetime.now()

def get_thread_status(name):
    with heartbeat_lock:
        last = THREAD_HEARTBEATS.get(name)
        start = THREAD_START_TIMES.get(name)
    if not last: return {"status": "STOPPED"}
    sec = (datetime.now() - last).total_seconds()
    status = "HEALTHY" if sec < 120 else "WARNING" if sec < 300 else "DEAD"
    uptime = f"{int((datetime.now() - start).total_seconds() // 3600)}h {int((datetime.now() - start).total_seconds() % 3600 // 60)}m" if start else None
    return {"status": status, "seconds_since": int(sec), "uptime": uptime}

# ================================
# HELPERS
# ================================
def get_safe_ticker_name(t): return t.replace('=', '_').replace('^', '').replace('/', '_')
def get_model_path(t): return Path("models") / f"{get_safe_ticker_name(t)}_lstm.h5"
def get_scaler_path(t): return Path("scalers") / f"{get_safe_ticker_name(t)}_scaler.pkl"
def get_accuracy_path(t): return Path("accuracy_logs") / f"{get_safe_ticker_name(t)}_accuracy.json"
def get_metadata_path(t): return Path("metadata") / f"{get_safe_ticker_name(t)}_meta.json"
def get_prediction_path(t, d): return Path("predictions") / f"{get_safe_ticker_name(t)}_{d}.json"

def normalize_dataframe_columns(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

PRICE_RANGES = {
    "AAPL": (150, 500), "TSLA": (150, 600), "NVDA": (100, 400), "MSFT": (300, 600),
    "GOOGL": (100, 400), "PLTR": (5, 200), "MSTR": (100, 900), "COIN": (50, 500),
    "ZC=F": (300, 700), "GC=F": (1500, 5000), "CL=F": (30, 150), "ZW=F": (400, 800),
    "SPY": (400, 900), "WEAT": (3, 15)
}

def validate_price(ticker, price):
    if not price or price <= 0: return False
    if ticker in PRICE_RANGES:
        mn, mx = PRICE_RANGES[ticker]
        return mn <= price <= mx
    return True

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    time.sleep(0.3)
    methods = []
    for _ in range(2):
        try:
            data = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
            data = normalize_dataframe_columns(data)
            if not data.empty:
                p = float(data['Close'].iloc[-1])
                if validate_price(ticker, p):
                    return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
                methods.append(f"1m-invalid-{p}")
        except Exception as e: methods.append(f"1m-{type(e).__name__}")
        time.sleep(1)
    
    for interval, period in [("5m", "1d"), ("1d", "5d")]:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            data = normalize_dataframe_columns(data)
            if not data.empty:
                p = float(data['Close'].iloc[-1])
                if validate_price(ticker, p):
                    return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
        except: pass
    
    try:
        info = yf.Ticker(ticker).info
        p = info.get('regularMarketPrice') or info.get('previousClose') or info.get('currentPrice')
        if p and validate_price(ticker, p):
            return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
    except: pass
    
    log_error(ErrorSeverity.WARNING, "get_latest_price", Exception(f"Failed: {methods}"), ticker=ticker, show_to_user=False)
    return None

# ================================
# CONFIG HELPERS
# ================================
def load_daemon_config():
    try:
        if DAEMON_CONFIG_PATH.exists():
            with config_lock:
                return json.loads(DAEMON_CONFIG_PATH.read_text())
    except: pass
    return {"enabled": False}

def save_daemon_config(enabled: bool):
    try:
        DAEMON_CONFIG_PATH.parent.mkdir(exist_ok=True)
        with config_lock:
            DAEMON_CONFIG_PATH.write_text(json.dumps({"enabled": bool(enabled)}))
    except: pass

def load_monitoring_config():
    try:
        if MONITORING_CONFIG_PATH.exists():
            with config_lock:
                return json.loads(MONITORING_CONFIG_PATH.read_text())
    except: pass
    return {"enabled": False}

def save_monitoring_config(enabled: bool):
    try:
        MONITORING_CONFIG_PATH.parent.mkdir(exist_ok=True)
        with config_lock:
            MONITORING_CONFIG_PATH.write_text(json.dumps({"enabled": bool(enabled)}))
    except: pass

# ================================
# ACCURACY TRACKING
# ================================
def load_accuracy_log(ticker):
    try:
        p = get_accuracy_path(ticker)
        return json.loads(p.read_text()) if p.exists() else {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}
    except: return {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}

def save_accuracy_log(ticker, data):
    try:
        with accuracy_lock:
            get_accuracy_path(ticker).write_text(json.dumps(data, indent=2))
        return True
    except: return False

def record_prediction(ticker, price, date):
    try:
        get_prediction_path(ticker, date).write_text(json.dumps({
            "ticker": ticker, "predicted_price": float(price),
            "prediction_date": date, "timestamp": datetime.now().isoformat()
        }, indent=2))
    except: pass

def validate_predictions(ticker):
    log = load_accuracy_log(ticker)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = get_prediction_path(ticker, yesterday)
    if path.exists():
        try:
            pred = json.loads(path.read_text())
            actual = get_latest_price(ticker)
            if actual and actual > 0:
                error = abs(pred["predicted_price"] - actual) / actual
                log["predictions"].append(pred["predicted_price"])
                log["errors"].append(error)
                log["dates"].append(yesterday)
                log["total_predictions"] += 1
                log["errors"] = log["errors"][-50:]
                log["avg_error"] = np.mean(log["errors"][-30:]) if log["errors"] else 0
                save_accuracy_log(ticker, log)
                path.unlink()
                return True, log
        except: pass
    return False, log

# ================================
# METADATA
# ================================
def load_metadata(ticker):
    try:
        p = get_metadata_path(ticker)
        return json.loads(p.read_text()) if p.exists() else {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0}
    except: return {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0}

def save_metadata(ticker, meta):
    try: get_metadata_path(ticker).write_text(json.dumps(meta, indent=2))
    except: pass

def should_retrain(ticker, acc_log, meta):
    if not get_model_path(ticker).exists(): return True, ["No model"]
    if acc_log["total_predictions"] >= 10 and acc_log["avg_error"] > LEARNING_CONFIG["accuracy_threshold"]:
        return True, [f"Error {acc_log['avg_error']:.1%}"]
    if meta.get("trained_date"):
        try:
            days = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
            if days >= 30: return True, [f"{days}d old"]
        except: pass
    try:
        df = yf.download(ticker, period="30d", progress=False)
        df = normalize_dataframe_columns(df)
        if len(df) > 5:
            cur_vol = df['Close'].pct_change().std()
            old_vol = meta.get("training_volatility", 0)
            if old_vol and abs(cur_vol - old_vol)/old_vol > 0.5:
                return True, ["Volatility shift"]
    except: pass
    return False, []

# ================================
# CONFIDENCE CHECKS
# ================================
def high_confidence_checklist(ticker: str, forecast: list, current_price: float) -> tuple:
    reasons = []
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    if acc.get("total_predictions", 0) < 12: reasons.append("Few live preds")
    if meta.get("retrain_count", 0) < 2: reasons.append("Low retrains")
    if acc.get("avg_error", 0.99) > 0.065: reasons.append(f"Error {acc['avg_error']:.1%}")
    if meta.get("trained_date"):
        try:
            if (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days > 14:
                reasons.append("Model stale")
        except: pass
    if forecast and current_price:
        move = abs(forecast[0] - current_price) / current_price
        if move > 0.12: reasons.append(f"Extreme move {move:+.1%}")
    return len(reasons) == 0, reasons

def ultra_confidence_shield(ticker: str, forecast: List[float], current_price: float) -> Tuple[bool, List[str]]:
    veto = []
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    age = 999
    if meta.get("trained_date"):
        try: age = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
        except: pass

    if acc.get("total_predictions", 0) < 25: veto.append("Low history")
    if acc.get("avg_error", 0.99) > 0.038: veto.append(f"Error {acc['avg_error']:.1%}")
    if meta.get("retrain_count", 0) < 4: veto.append("Low retrains")
    if age > 9: veto.append(f"Stale {age}d")
    if forecast and current_price:
        if abs(forecast[0] - current_price)/current_price > 0.09: veto.append("Insane move")
    return len(veto) == 0, veto

# ================================
# MODEL
# ================================
def build_lstm_model():
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], 1)),
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(15), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_self_learning_model(ticker, days=5, force_retrain=False):
    # Acquire ticker-specific lock (prevents concurrent training)
    lock = get_training_lock(ticker)
    if not lock.acquire(blocking=False):
        logger.warning(f"Training {ticker} already in progress by another thread, skipping")
        return None, None, None
   
    try:
        logger.info(f"Training {ticker} (force={force_retrain})")
        updated, acc_log = validate_predictions(ticker)
        meta = load_metadata(ticker)
        needs, reasons = should_retrain(ticker, acc_log, meta)

        if needs or force_retrain:
            # Thread-safe logging — NO st. calls in background!
            logger.info(f"RETRAINING {ticker} → Reasons: {', '.join(reasons)}")
           
            # Only update session_state if we're in the main thread (user-triggered retrain)
            if threading.current_thread() is threading.main_thread():
                if 'st' in globals() and hasattr(st, 'session_state'):
                    st.session_state.setdefault('learning_log', []).append(
                        f"Retraining {ticker}: {', '.join(reasons) or 'Force'}"
                    )

        df = yf.download(ticker, period="1y", progress=False)
        df = normalize_dataframe_columns(df)
        if df is None or len(df) < 100: 
            return None, None, None
        df = df[['Close']].ffill().bfill()
        if df['Close'].isna().any(): 
            return None, None, None

        scaler_path = get_scaler_path(ticker)
        if force_retrain or not scaler_path.exists():
            scaler = MinMaxScaler()
            scaler.fit(df[['Close']])
            joblib.dump(scaler, scaler_path)
        else:
            scaler = joblib.load(scaler_path)

        scaled = scaler.transform(df[['Close']])
        X, y = [], []
        lookback = LEARNING_CONFIG["lookback_window"]
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        if len(X) == 0: 
            return None, None, None

        with model_cache_lock:
            model = None
            if force_retrain or needs:
                model = build_lstm_model()
                model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0,
                          validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
                meta["retrain_count"] = meta.get("retrain_count", 0) + 1
            else:
                try:
                    model = tf.keras.models.load_model(str(get_model_path(ticker)))
                    recent = int(len(X)*0.3)
                    model.fit(X[-recent:], y[-recent:], epochs=LEARNING_CONFIG["fine_tune_epochs"], batch_size=32, verbose=0)
                except:
                    model = build_lstm_model()
                    model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0)

            try: 
                model.save(str(get_model_path(ticker)))
            except: 
                pass

            last = scaled[-lookback:].reshape(1, lookback, 1)
            preds = []
            for _ in range(days):
                pred = model.predict(last, verbose=0)
                preds.append(pred[0,0])
                last = np.append(last[:,1:,:], pred.reshape(1,1,1), axis=1)
            forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            record_prediction(ticker, forecast[0], tomorrow)

            dates = []
            i = 1
            while len(dates) < days:
                d = datetime.now().date() + timedelta(days=i)
                if d.weekday() < 5: 
                    dates.append(d)
                i += 1

            meta.update({
                "trained_date": datetime.now().isoformat(),
                "training_samples": len(X),
                "training_volatility": float(df['Close'].pct_change().std()),
                "version": meta.get("version", 1) + 1,
                "last_accuracy": acc_log.get("avg_error", 0)
            })
            save_metadata(ticker, meta)
            tf.keras.backend.clear_session()
            return forecast, dates, model
   
    finally:
        # Always release the lock
        lock.release()

# ================================
# 6%+ DETECTOR
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def detect_pre_move_6percent(ticker, name):
    try:
        import pytz
        data = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        data = normalize_dataframe_columns(data)
        if data is None or len(data) < 60: return None

        close = data['Close'].values
        volume = data['Volume'].values
        if len(close) < 30: return None

        recent_vol = volume[-5:]
        prev_vol = volume[-20:-5]
        baseline_vol = volume[-60:-20]
        recent_vol_avg = np.mean(recent_vol)
        prev_vol_avg = np.mean(prev_vol) if len(prev_vol) > 0 else 1
        baseline_vol_avg = np.mean(baseline_vol) if len(baseline_vol) > 0 else 1
        vol_acceleration = recent_vol_avg / prev_vol_avg
        vol_spike_vs_baseline = recent_vol_avg / baseline_vol_avg

        recent_prices = close[-10:]
        prev_prices = close[-20:-10]
        recent_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        prev_momentum = (prev_prices[-1] - prev_prices[0]) / prev_prices[0]
        momentum_acceleration = abs(recent_momentum) > abs(prev_momentum) * 1.5

        score = 0
        factors = []
        direction = "UP" if recent_momentum > 0 else "DOWN"
        original_direction = direction

        if vol_acceleration > 2.0 and vol_spike_vs_baseline > 3.0:
            score += 30; factors.append(f"Vol×{vol_acceleration:.1f}")
        elif vol_acceleration > 1.5:
            score += 20; factors.append(f"Vol↑{vol_acceleration:.1f}x")
        if momentum_acceleration and abs(recent_momentum) > 0.01:
            score += 25; factors.append("MomentumAccel")
        if abs(recent_momentum) > 0.008:
            score += 15; factors.append("StrongMomentum")

        # Institutional filters
        try:
            info = yf.Ticker(ticker).info
            vwap = info.get('vwap') or info.get('regularMarketPreviousClose') or close[-1]
            if direction == "UP" and close[-1] < vwap * 0.997: return None
            if direction == "DOWN" and close[-1] > vwap * 1.003: return None
            factors.append("VWAP✓")
        except: pass

        # AI Pattern Boost
        try:
            boost, triggers, pred_dir, conf = check_auto_patterns(ticker, data)
            score += boost
            if boost > 0:
                factors.extend(triggers)
                if pred_dir != "NEUTRAL":
                    direction = pred_dir
                    factors.append(f"AI→{pred_dir} {conf}%")
        except: pass

        if score >= 75:
            # Removed MSTR trap logic - trust the AI pattern miner
            confidence = min(99, 60 + score // 2 + (30 if 'AI→' in ''.join(factors) else 0))
            return {"asset": name, "direction": direction, "confidence": confidence, "factors": factors, "score": score}
        return None
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "detect_pre_move_6percent", e, ticker=ticker, show_to_user=False)
        return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
        return r.status_code == 200
    except: return False

# ================================
# BACKTESTING
# ================================
BACKTEST_DIR = Path("backtest_results")
BACKTEST_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=86400, show_spinner="Running backtest...")
def run_backtest(ticker: str, start_date: str = "2022-01-01", end_date: str = None,
                 initial_capital: float = 10000.0, max_position_size: float = 1.0,
                 stop_loss_pct: float = 0.08, take_profit_pct: float = 0.15,
                 confidence_threshold: float = 0.65) -> dict:

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"BACKTEST: {ticker} | {start_date} → {end_date}")

    df_full = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
    df_full = normalize_dataframe_columns(df_full)
    if df_full.empty or len(df_full) < 300: return None

    df_full = df_full[['Close']].dropna()
    lookback = LEARNING_CONFIG["lookback_window"]
    min_training_days = lookback + 100
    if len(df_full) <= min_training_days + 1: return None

    equity = initial_capital
    position = 0
    entry_price = 0
    equity_curve = [initial_capital]
    dates = [df_full.index[min_training_days]]
    trades = []

    for i in range(min_training_days, len(df_full) - 1):
        current_date = df_full.index[i]
        next_day = df_full.index[i + 1]
        current_price = df_full['Close'].iloc[i]
        next_price = df_full['Close'].iloc[i + 1]

        train_end = i
        train_start = max(0, train_end - 365)
        train_df = df_full.iloc[train_start:train_end].copy()

        if len(train_df) < lookback + 50: continue

        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(train_df[['Close']])

            X, y = [], []
            for j in range(lookback, len(scaled)):
                X.append(scaled[j-lookback:j])
                y.append(scaled[j])
            X, y = np.array(X), np.array(y)

            if len(X) < 50: continue

            model = build_lstm_model()
            model.fit(X, y, epochs=15, batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

            last_seq = scaled[-lookback:].reshape((1, lookback, 1))
            pred_scaled = model.predict(last_seq, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predicted_return = (pred_price - current_price) / current_price

            preds = []
            seq = last_seq.copy()
            for _ in range(7):
                p = model.predict(seq, verbose=0)[0,0]
                preds.append(p)
                seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)
            confidence = 1.0 - (np.std(preds) / (np.mean(preds) + 1e-8))
            confidence = max(0.0, min(1.0, confidence))

            tf.keras.backend.clear_session()

        except Exception as e:
            predicted_return = 0
            confidence = 0

        in_position = position > 0
        unrealized = (current_price - entry_price) / entry_price if in_position else 0

        if in_position:
            if unrealized <= -stop_loss_pct:
                equity *= (1 + unrealized)
                trades.append({"type": "SELL", "reason": "STOP_LOSS", "price": current_price,
                              "return": unrealized*100, "date": current_date.date()})
                position = 0
            elif unrealized >= take_profit_pct:
                equity *= (1 + unrealized)
                trades.append({"type": "SELL", "reason": "TAKE_PROFIT", "price": current_price,
                              "return": unrealized*100, "date": current_date.date()})
                position = 0

        if not in_position and predicted_return > 0.025 and confidence > confidence_threshold:
            position = min(max_position_size, 1.0)
            entry_price = current_price
            trades.append({"type": "BUY", "price": current_price, "confidence": round(confidence, 3),
                          "predicted": round(predicted_return*100, 2), "date": current_date.date()})

        if position > 0:
            daily_return = (next_price - current_price) / current_price
            equity *= (1 + daily_return * position)

        equity_curve.append(equity)
        dates.append(next_day)

    if len(trades) == 0: return None

    returns = pd.Series([(df_full['Close'].iloc[i+1] - df_full['Close'].iloc[i]) / df_full['Close'].iloc[i] 
                         for i in range(min_training_days, len(df_full)-1)])
    equity_series = pd.Series(equity_curve)

    total_return = (equity / initial_capital - 1) * 100
    days = (df_full.index[-1] - df_full.index[0]).days
    cagr = ((equity / initial_capital) ** (365.25 / max(days, 1)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252) + 1e-8) if returns.std() > 0 else 0

    rolling_max = equity_series.cummax()
    drawdown = equity_series / rolling_max - 1.0
    max_dd = drawdown.min() * 100

    completed_trades = [t for t in trades if t["type"] == "SELL" and "return" in t]
    wins = [t for t in completed_trades if t["return"] > 0]
    losses = [t for t in completed_trades if t["return"] <= 0]
    win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0
    avg_win = np.mean([t["return"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["return"] for t in losses]) if losses else 0

    bh_return = (df_full['Close'].iloc[-1] / df_full['Close'].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity_curve, name="Strategy", line=dict(width=3, color="#00C853")))
    fig.add_trace(go.Scatter(x=[dates[0], dates[-1]],
                             y=[initial_capital, initial_capital * (1 + bh_return/100)],
                             name="Buy & Hold", line=dict(dash="dash", color="#888")))
    fig.update_layout(title=f"{ticker} • Strategy {total_return:+.2f}% vs B&H {bh_return:+.2f}%",
                      template="plotly_dark", height=600, hovermode="x unified")

    trade_fig = go.Figure()
    trade_fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Close'], name="Price", line=dict(color="#1f77b4")))
    buys = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]
    if buys:
        trade_fig.add_trace(go.Scatter(x=[t["date"] for t in buys], y=[t["price"] for t in buys],
                                       mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=14, color="lime")))
    if sells:
        trade_fig.add_trace(go.Scatter(x=[t["date"] for t in sells], y=[t["price"] for t in sells],
                                       mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=14, color="red")))

    trade_fig.update_layout(title=f"{ticker} • Trade Timeline", template="plotly_dark", height=500)

    result = {
        "ticker": ticker, "period": f"{start_date} → {end_date}", "initial_capital": initial_capital,
        "final_equity": round(equity, 2), "total_return_pct": round(total_return, 2), "cagr_pct": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 3), "volatility_pct": round(volatility, 2), "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1), "avg_win_pct": round(avg_win, 2), "avg_loss_pct": round(avg_loss, 2),
        "total_trades": len(completed_trades), "buy_and_hold_pct": round(bh_return, 2),
        "alpha_vs_bh": round(total_return - bh_return, 2),
        "equity_curve": {"dates": [d.strftime("%Y-%m-%d") for d in dates], "values": equity_curve},
        "trades": trades, "fig_equity": fig, "fig_trades": trade_fig, "success": True
    }

    logger.info(f"BACKTEST COMPLETE: {ticker} → {total_return:+.2f}% | Sharpe {sharpe:.3f}")
    return result

# ================================
# FIXED BACKGROUND THREADS
# ================================

def continuous_learning_daemon():
    """
    FIXED: Properly updates heartbeat during sleep and continues forever
    """
    update_heartbeat("learning_daemon")
    THREAD_START_TIMES["learning_daemon"] = datetime.now()
    logger.info("🤖 Learning Daemon STARTED")
    
    cycle_count = 0
    
    while True:  # ← FIXED: Always run, check config inside loop
        try:
            # Check if daemon is enabled
            if not load_daemon_config().get("enabled", False):
                logger.info("Learning Daemon paused (disabled in config)")
                time.sleep(30)  # Check every 30 seconds if re-enabled
                update_heartbeat("learning_daemon")
                continue
            
            cycle_count += 1
            logger.info(f"🔄 Learning Daemon: Starting cycle #{cycle_count}")
            update_heartbeat("learning_daemon")
            
            # Train all assets
            trained_count = 0
            for cat in ASSET_CATEGORIES.values():
                for name, t in cat.items():
                    try:
                        update_heartbeat("learning_daemon")  # ← Update during training
                        train_self_learning_model(t, days=1)
                        trained_count += 1
                        logger.debug(f"Trained {t} ({trained_count}/14)")
                        time.sleep(2)
                    except Exception as e:
                        log_error(ErrorSeverity.WARNING, "learning_daemon_train", e, 
                                ticker=t, show_to_user=False)
            
            logger.info(f"✅ Learning Daemon: Cycle #{cycle_count} complete ({trained_count} assets)")
            
            # Sleep for 1 hour with heartbeat updates every 60 seconds
            sleep_duration = 3600  # 1 hour
            sleep_intervals = sleep_duration // 60  # 60 intervals of 60 seconds
            
            logger.info(f"💤 Learning Daemon: Sleeping for {sleep_duration//60} minutes")
            
            for i in range(sleep_intervals):
                time.sleep(60)  # Sleep 60 seconds
                update_heartbeat("learning_daemon")  # ← FIXED: Update during sleep
                
                # Check if disabled during sleep
                if not load_daemon_config().get("enabled", False):
                    logger.info("Learning Daemon stopped during sleep")
                    break
            
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "continuous_learning_daemon", e, 
                    user_message="Learning daemon error - will retry", show_to_user=False)
            time.sleep(60)  # Wait before retry
            update_heartbeat("learning_daemon")


def monitor_6percent_pre_move():
    """
    FIXED: Properly updates heartbeat during sleep and continues forever
    NO MORE TRAINING - Only reads cached forecasts
    """
    update_heartbeat("monitoring")
    THREAD_START_TIMES["monitoring"] = datetime.now()
    logger.info("📡 6%+ Monitoring STARTED")
    
    alerted = set()
    scan_count = 0
    
    while True:  # ← FIXED: Always run, check config inside loop
        try:
            # Check if monitoring is enabled
            if not load_monitoring_config().get("enabled", False):
                logger.info("6%+ Monitoring paused (disabled in config)")
                time.sleep(30)
                update_heartbeat("monitoring")
                continue
            
            scan_count += 1
            update_heartbeat("monitoring")
            logger.debug(f"📡 Monitoring: Scan #{scan_count}")
            
            alerts_sent = 0
            
            # Scan all assets
            for cat in ASSET_CATEGORIES.values():
                for name, t in cat.items():
                    try:
                        update_heartbeat("monitoring")  # ← Update during scan
                        
                        signal = detect_pre_move_6percent(t, name)
                        
                        if signal and signal["confidence"] >= 90:
                            key = f"{t}_{signal['direction']}"
                            
                            if key not in alerted:
                                # FIXED: Read cached forecast instead of retraining
                                current_price = get_latest_price_robust(t)
                                if current_price:
                                    # Get tomorrow's date for cached forecast
                                    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                                    forecast_path = get_prediction_path(t, tomorrow)
                                    
                                    # Read cached forecast (created by learning daemon)
                                    if forecast_path.exists():
                                        try:
                                            pred_data = json.loads(forecast_path.read_text())
                                            forecast = [pred_data.get("predicted_price")]
                                            
                                            if forecast and forecast[0]:
                                                ultra_passed, ultra_reasons = ultra_confidence_shield(t, forecast, current_price)
                                                
                                                if ultra_passed:
                                                    text = (f"🔮 NUCLEAR PRE-MOVE DETECTED\n\n"
                                                           f"{signal['asset']} → {signal['direction']}\n"
                                                           f"Confidence: {signal['confidence']}%\n"
                                                           f"Factors: {', '.join(signal['factors'])}\n\n"
                                                           f"✅ ULTRA-CONFIDENCE: PASSED")
                                                    
                                                    if send_telegram_alert(text):
                                                        alerted.add(key)
                                                        alerts_sent += 1
                                                        logger.info(f"🚨 ALERT SENT: {signal['asset']} → {signal['direction']}")
                                                else:
                                                    logger.debug(f"Ultra-confidence failed for {t}: {ultra_reasons}")
                                            else:
                                                logger.debug(f"No valid forecast for {t}")
                                        except Exception as e:
                                            logger.warning(f"Failed to read forecast for {t}: {e}")
                                    else:
                                        logger.debug(f"No cached forecast for {t} (learning daemon hasn't trained yet)")
                    
                    except Exception as e:
                        log_error(ErrorSeverity.WARNING, "monitor_6percent_scan", e, 
                                ticker=t, user_message=f"Error scanning {t}: {str(e)}", show_to_user=False)
            
            if alerts_sent > 0:
                logger.info(f"✅ Monitoring: Scan #{scan_count} complete - {alerts_sent} alert(s) sent")
            
            # Sleep for 60 seconds with heartbeat updates every 15 seconds
            for i in range(4):  # 4 intervals of 15 seconds = 60 seconds
                time.sleep(15)
                update_heartbeat("monitoring")
                
                if not load_monitoring_config().get("enabled", False):
                    logger.info("6%+ Monitoring stopped")
                    alerted.clear()
                    break
        
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "monitor_6percent_pre_move", e,
                    user_message=f"Monitoring error: {str(e)}", show_to_user=False)
            time.sleep(30)
            update_heartbeat("monitoring")


def thread_watchdog():
    """
    FIXED: Better status detection and logging
    """
    update_heartbeat("watchdog")
    THREAD_START_TIMES["watchdog"] = datetime.now()
    logger.info("🐕 Watchdog STARTED")
    
    while True:
        try:
            update_heartbeat("watchdog")
            
            for name in ["learning_daemon", "monitoring"]:
                status = get_thread_status(name)
                
                # Only log if thread is supposed to be running
                if name == "learning_daemon":
                    enabled = load_daemon_config().get("enabled", False)
                elif name == "monitoring":
                    enabled = load_monitoring_config().get("enabled", False)
                else:
                    enabled = False
                
                if enabled:
                    if status["status"] == "DEAD":
                        logger.error(f"🔴 Thread {name} is DEAD (no heartbeat for {status['seconds_since']}s)")
                    elif status["status"] == "WARNING":
                        logger.warning(f"⚠️ Thread {name} is WARNING (last heartbeat {status['seconds_since']}s ago)")
                    elif status["status"] == "HEALTHY":
                        # Only log healthy status every 10 minutes to reduce spam
                        if status["seconds_since"] < 30:  # Just started or recently updated
                            logger.debug(f"✅ Thread {name} is HEALTHY (uptime: {status['uptime']})")
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "thread_watchdog", e, show_to_user=False)
            time.sleep(30)


def show_error_dashboard():
    st.subheader("System Diagnostics")
    stats = get_error_statistics()
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Errors", stats["total"])
    with col2: st.metric("Critical", stats["by_severity"].get("CRITICAL", 0))
    with col3: st.metric("Warnings", stats["by_severity"].get("WARNING", 0))
    if stats["recent"]:
        with st.expander("Recent Errors"):
            for e in stats["recent"][::-1]:
                st.write(f"**{e['severity']}** | {e['function']} | {e['user_message']}")

# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.2", layout="wide")

if 'alert_history' not in st.session_state: st.session_state.alert_history = {}
if 'learning_log' not in st.session_state: st.session_state.learning_log = []
if 'error_logs' not in st.session_state: st.session_state.error_logs = []

def initialize_background_threads():
    """
    FIXED: Properly start threads and track their state
    """
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        logger.info("🚀 Initializing background threads...")
        
        # Always start watchdog
        watchdog_thread = threading.Thread(target=thread_watchdog, daemon=True, name="WatchdogThread")
        watchdog_thread.start()
        logger.info("✅ Watchdog thread started")
        
        # Start learning daemon if enabled
        if load_daemon_config().get("enabled", False):
            learning_thread = threading.Thread(target=continuous_learning_daemon, daemon=True, name="LearningDaemon")
            learning_thread.start()
            logger.info("✅ Learning daemon thread started")
        
        # Start monitoring if enabled
        if load_monitoring_config().get("enabled", False):
            monitoring_thread = threading.Thread(target=monitor_6percent_pre_move, daemon=True, name="MonitoringThread")
            monitoring_thread.start()
            logger.info("✅ Monitoring thread started")

initialize_background_threads()

def add_header():
    st.markdown("<div style='text-align:center;padding:15px;background:#1a1a1a;color:#00C853;border-radius:8px;'><h2>🧠 AI - ALPHA STOCK TRACKER v4.2</h2><p>Self-Learning • Nuclear Alerts • Backtesting • Institutional Grade</p></div>", unsafe_allow_html=True)

def add_footer():
    st.markdown("<div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'><p>© 2025 AI - Alpha Stock Tracker v4.2 | The Final Form</p></div>", unsafe_allow_html=True)

add_header()

# Sidebar
with st.sidebar:
    st.header("⚙️ Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]
    
    st.markdown("---")
    st.subheader("🔧 Controls")
    if st.button("🔄 Force Retrain", use_container_width=True):
        with st.spinner("Retraining..."):
            train_self_learning_model(ticker, force_retrain=True)
            st.success("✅ Done!")
            st.rerun()
    
    if st.button("🚀 Bootstrap All Models", use_container_width=True):
        with st.spinner("Training all models... (5-10 min)"):
            all_tickers = [t for cat in ASSET_CATEGORIES.values() for _, t in cat.items()]
            progress = st.progress(0)
            for idx, t in enumerate(all_tickers):
                try:
                    train_self_learning_model(t, days=5, force_retrain=True)
                except: pass
                progress.progress((idx + 1) / len(all_tickers))
            st.success("✅ All models trained!")
            time.sleep(2)
            st.rerun()
    
    st.markdown("---")
    st.subheader("🤖 Learning Daemon")
    dc = load_daemon_config()
    st.write("**Status:**", "🟢 RUNNING" if dc.get("enabled") else "🔴 STOPPED")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start", key="dstart", use_container_width=True):
            save_daemon_config(True)
            threading.Thread(target=continuous_learning_daemon, daemon=True, name="LearningDaemon").start()
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", key="dstop", use_container_width=True):
            save_daemon_config(False)
            st.rerun()
    
    st.markdown("---")
    st.subheader("📡 6%+ Monitoring")
    mc = load_monitoring_config()
    st.write("**Status:**", "🟢 RUNNING" if mc.get("enabled") else "🔴 STOPPED")
    
    if st.button("🧪 Test Telegram", use_container_width=True):
        success = send_telegram_alert("✅ TEST ALERT\nAI - Alpha Tracker v4.2")
        st.success("✅ Sent!") if success else st.error("❌ Check keys")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start", key="mstart", use_container_width=True):
            save_monitoring_config(True)
            threading.Thread(target=monitor_6percent_pre_move, daemon=True, name="MonitoringThread").start()
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", key="mstop", use_container_width=True):
            save_monitoring_config(False)
            st.rerun()

# Main
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(f"<h2 style='text-align:center;'>LIVE: <code style='font-size:2em;background:#333;padding:10px 20px;border-radius:12px;'>${price:.2f}</code></h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Market closed or no data")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 5-Day Forecast", "🏥 All Models", "🔬 Backtesting", "🔍 Diagnostics"])

with tab1:
    if st.button("Daily Recommendation", use_container_width=True):
        with st.spinner("Analyzing..."):
            forecast, _, _ = train_self_learning_model(ticker, days=1)
            
            # ← BULLETPROOF FORECAST CHECK
            if forecast is not None and len(np.array(forecast).flatten()) > 0:
                forecast_val = float(np.array(forecast).flatten()[0])
                passed, reasons = high_confidence_checklist(ticker, [forecast_val], price or 100)
                
                if passed:
                    change_pct = (forecast_val - price) / price * 100 if price else 0
                    if change_pct >= 3:
                        action = "STRONG BUY"
                    elif change_pct >= 1.5:
                        action = "BUY"
                    elif change_pct <= -1.5:
                        action = "SELL"
                    else:
                        action = "HOLD"
                    st.success(f"AI Predicts: ${forecast_val:.2f} ({change_pct:+.2f}%) → **{action}**")
                else:
                    st.warning("Low Confidence\n\n" + "\n".join([f"• {r}" for r in reasons]))
            else:
                st.error("Forecast failed or no data")

with tab2:
    if st.button("Generate 5-Day Forecast", use_container_width=True):
        with st.spinner("Forecasting..."):
            forecast, dates, _ = train_self_learning_model(ticker, days=5)
            
            # ← BULLETPROOF FORECAST + DATES CHECK
            if (forecast is not None and 
                dates is not None and 
                len(forecast) > 0 and 
                len(dates) > 0):
                
                # Force to Python lists & flatten
                forecast = np.array(forecast).flatten().tolist()
                if len(forecast) > len(dates):
                    forecast = forecast[:len(dates)]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=forecast,
                    mode='lines+markers',
                    name='AI Forecast',
                    line=dict(color="#00C853", width=4)
                ))
                if price:
                    fig.add_trace(go.Scatter(
                        x=dates[:1], y=[price],
                        mode='markers',
                        name='Current Price',
                        marker=dict(color="white", size=14, symbol="circle")
                    ))

                fig.update_layout(
                    title=f"{asset} - 5-Day AI Forecast",
                    template="plotly_dark",
                    height=620,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Safe indexing
                f1 = forecast[0]
                f3 = forecast[min(2, len(forecast)-1)]
                f5 = forecast[min(4, len(forecast)-1)]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tomorrow", f"${f1:.2f}", 
                             f"{(f1-price)/price*100:+.2f}%" if price else None)
                with col2:
                    st.metric("Day +3", f"${f3:.2f}", 
                             f"{(f3-price)/price*100:+.2f}%" if price else None)
                with col3:
                    st.metric("Day +5", f"${f5:.2f}", 
                             f"{(f5-price)/price*100:+.2f}%" if price else None)
            else:
                st.error("Forecast failed — no valid data returned")

with tab3:
    st.subheader("All Models — Institutional Health Dashboard")
    try:
        all_assets = []
        broken_models = []

        for cat_name, assets in ASSET_CATEGORIES.items():
            for asset_name, asset_ticker in assets.items():
                meta = load_metadata(asset_ticker)
                acc_log = load_accuracy_log(asset_ticker)

                try:
                    current_price = get_latest_price(asset_ticker)
                    current_price_str = f"${current_price:.2f}" if current_price else "N/A"
                except:
                    current_price = None
                    current_price_str = "N/A"

                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                pred_path = get_prediction_path(asset_ticker, tomorrow)
                predicted_price = None
                price_change = None

                if pred_path.exists():
                    try:
                        pred_data = json.loads(pred_path.read_text())
                        predicted_price = pred_data.get("predicted_price")
                        if predicted_price and current_price:
                            price_change = (predicted_price - current_price) / current_price * 100
                    except: pass

                pred_str = "N/A"
                if predicted_price:
                    change_str = f"{price_change:+.1f}%" if price_change is not None else ""
                    pred_str = f"${predicted_price:.2f} ({change_str})".strip()

                total_preds = acc_log.get("total_predictions", 0)
                mape = None
                directional_acc = None

                if total_preds > 0:
                    mape = acc_log.get('avg_error', 0) * 100
                    correct = 0
                    pred_list = acc_log.get("predictions", [])
                    if len(pred_list) >= 2:
                        for i in range(1, min(30, len(pred_list))):
                            pred_move = pred_list[-i] - pred_list[-i-1]
                            try:
                                hist = yf.download(asset_ticker, period="60d", progress=False, threads=False)['Close']
                                if len(hist) > i:
                                    actual_move = hist.iloc[-i] - hist.iloc[-i-1]
                                    if (pred_move > 0) == (actual_move > 0):
                                        correct += 1
                            except: pass
                        directional_acc = (correct / min(30, len(pred_list)-1)) * 100

                if total_preds == 0:
                    health = "No data"
                elif mape is None:
                    health = "No data"
                elif directional_acc is not None and directional_acc >= 75 and mape <= 6:
                    health = "Excellent"
                elif directional_acc is not None and directional_acc >= 65 and mape <= 9:
                    health = "Good"
                elif mape <= 12:
                    health = "Fair"
                elif mape <= 25:
                    health = "Poor"
                else:
                    health = "Broken"
                    broken_models.append(asset_ticker)

                all_assets.append({
                    "Health": health,
                    "Asset": asset_name,
                    "Ticker": asset_ticker,
                    "Last Close": current_price_str,
                    "Tomorrow": pred_str,
                    "Version": meta.get("version", "1.0"),
                    "Retrains": meta.get("retrain_count", 0),
                    "MAPE": f"{mape:.1f}%" if mape else "N/A",
                    "Direction%": f"{directional_acc:.1f}%" if directional_acc else "N/A",
                    "Predictions": total_preds,
                    "Last Trained": meta.get("trained_date", "")[:10] if meta.get("trained_date") else "Never"
                })

        df = pd.DataFrame(all_assets)
        health_order = {"Excellent":0, "Good":1, "Fair":2, "Poor":3, "Broken":4, "No data":5}
        df["sort"] = df["Health"].map(health_order)
        df = df.sort_values("sort").drop("sort", axis=1).reset_index(drop=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Health Legend**  
        • Excellent = ≥75% directional + ≤6% error  
        • Good = ≥65% directional + ≤9% error  
        • Fair = ≤12% average error  
        • Poor = High error  
        • Broken = Catastrophic (auto-fix below)  
        • No data = Model needs live predictions
        """)

        if broken_models:
            st.error(f"🔴 {len(broken_models)} model(s) BROKEN")
            if st.button("🔧 REBUILD ALL BROKEN MODELS NOW", type="primary", use_container_width=True):
                with st.spinner("Rebuilding broken models..."):
                    for t in broken_models:
                        get_model_path(t).unlink(missing_ok=True)
                        get_scaler_path(t).unlink(missing_ok=True)
                        train_self_learning_model(t, days=5, force_retrain=True)
                        st.success(f"✅ Rebuilt {t}")
                    st.rerun()

    except Exception as e:
        log_error(ErrorSeverity.ERROR, "all_models_tab", e)
        st.error("Dashboard error — check logs")

with tab4:
    st.header("🔬 Institutional Backtesting Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        bt_option = st.selectbox("Asset", options=[f"{name} ({t})" for cat in ASSET_CATEGORIES.values() for name, t in cat.items()])
        bt_ticker = bt_option.split(" (")[1][:-1]
    with col2:
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_cap = st.number_input("Initial Capital ($)", 1000, 1000000, 10000)
    with col2:
        sl = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5) / 100
    with col3:
        tp = st.slider("Take Profit %", 5.0, 50.0, 15.0, 1.0) / 100
    
    if st.button("▶️ RUN INSTITUTIONAL BACKTEST", type="primary", use_container_width=True):
        result = run_backtest(ticker=bt_ticker, start_date=start_date.strftime("%Y-%m-%d"), 
                             initial_capital=initial_cap, stop_loss_pct=sl, take_profit_pct=tp)
        
        if result and result.get("success"):
            st.success(f"✅ BACKTEST COMPLETE → ${result['final_equity']:,.2f}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Return", f"{result['total_return_pct']:+.2f}%", 
                         f"vs B&H {result['buy_and_hold_pct']:+.2f}%")
                st.metric("Alpha", f"{result['alpha_vs_bh']:+.2f}%")
            with c2:
                st.metric("CAGR", f"{result['cagr_pct']:.2f}%")
                st.metric("Sharpe", f"{result['sharpe_ratio']:.3f}")
            with c3:
                st.metric("Max DD", f"{result['max_drawdown_pct']:.2f}%")
                st.metric("Win Rate", f"{result['win_rate_pct']:.1f}%")
            with c4:
                st.metric("Trades", result['total_trades'])
                st.metric("Avg Win/Loss", f"{result['avg_win_pct']:+.2f}% / {result['avg_loss_pct']:+.2f}%")
            
            st.plotly_chart(result['fig_equity'], use_container_width=True)
            st.plotly_chart(result['fig_trades'], use_container_width=True)
            
            with st.expander("📋 View All Trades"):
                st.dataframe(pd.DataFrame(result['trades']), use_container_width=True)
        else:
            st.error("❌ Backtest failed or no valid trades")

with tab5:
    show_error_dashboard()

add_footer()
