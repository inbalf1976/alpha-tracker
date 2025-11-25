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
from typing import Tuple, List, Optional
import random

# ================================
# GLOBAL YFINANCE PROTECTION
# ================================
_yf_semaphore = threading.Semaphore(1)
_yf_session = requests.Session()
_yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
})

def safe_yf_download(ticker: str, **kwargs) -> pd.DataFrame:
    """100% safe yfinance with backoff + throttling"""
    with _yf_semaphore:
        time.sleep(random.uniform(1.8, 4.2))
        for attempt in range(5):
            try:
                data = yf.download(
                    ticker,
                    session=_yf_session,
                    progress=False,
                    threads=False,
                    timeout=20,
                    **kwargs
                )
                if not data.empty:
                    return normalize_dataframe_columns(data)
            except Exception as e:
                if "401" in str(e) or "Timeout" in str(e):
                    wait = (2 ** attempt) + random.random() * 5
                    if attempt < 3:
                        time.sleep(wait)
                    continue
                time.sleep(2)
        return pd.DataFrame()

# Monkey-patch Ticker
orig_ticker = yf.Ticker
class SafeTicker:
    def __init__(self, ticker):
        self.ticker = orig_ticker(ticker, session=_yf_session)
    def __getattr__(self, name):
        return getattr(self.ticker, name)
yf.Ticker = SafeTicker

# ================================
# SUPPRESSIONS
# ================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

try:
    from streamlit.runtime.scriptrunner import script_run_context
    if hasattr(script_run_context, '_LOGGER'):
        script_run_context._LOGGER.setLevel('ERROR')
except:
    pass

# ================================
# AUTO PATTERNS
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
        confidence = min(99, int(best_auc * 100 + boost // 2.5))
        triggers = [
            f"{best_match.get('model','?').upper()} AUC {best_auc:.3f}",
            f"Boost +{boost}",
            f"{best_match.get('timeframe','?').upper()}",
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
# LOGGING & ERROR TRACKING
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
    except:
        pass
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
    if show_to_user and 'st' in globals():
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"{user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"{user_message}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"{user_message}")
    try:
        if hasattr(st, 'session_state'):
            st.session_state.setdefault('error_logs', []).append(error_data)
    except: pass

def get_error_statistics():
    try:
        if not ERROR_LOG_PATH.exists():
            return {"total": 0, "by_severity": {}, "recent": []}
        errors = json.loads(ERROR_LOG_PATH.read_text())
        by_sev = {}
        for e in errors:
            s = e.get('severity', 'UNKNOWN')
            by_sev[s] = by_sev.get(s, 0) + 1
        return {"total": len(errors), "by_severity": by_sev, "recent": errors[-10:]}
    except:
        return {"total": 0, "by_severity": {}, "recent": []}

# ================================
# CONFIG & ASSETS
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
except:
    BOT_TOKEN = CHAT_ID = None

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

# Threading
model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()
config_lock = threading.Lock()
heartbeat_lock = threading.Lock()
session_state_lock = threading.Lock()
THREAD_HEARTBEATS = {"learning_daemon": None, "monitoring": None, "watchdog": None}
THREAD_START_TIMES = {"learning_daemon": None, "monitoring": None, "watchdog": None}

def update_heartbeat(name):
    with heartbeat_lock:
        THREAD_HEARTBEATS[name] = datetime.now()

def get_thread_status(name):
    with heartbeat_lock:
        last = THREAD_HEARTBEATS.get(name)
        start = THREAD_START_TIMES.get(name)
        if not last:
            return {"status": "STOPPED"}
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
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

PRICE_RANGES = {
    "AAPL": (150, 500), "TSLA": (150, 600), "NVDA": (100, 400), "MSFT": (300, 600), "GOOGL": (100, 400),
    "PLTR": (5, 200), "MSTR": (100, 900), "COIN": (50, 500), "ZC=F": (300, 700), "GC=F": (1500, 5000),
    "CL=F": (30, 150), "ZW=F": (400, 800), "SPY": (400, 900), "WEAT": (3, 15)
}

def validate_price(ticker, price):
    if not price or price <= 0: return False
    if ticker in PRICE_RANGES:
        mn, mx = PRICE_RANGES[ticker]
        return mn <= price <= mx
    return True

@st.cache_data(ttl=90, show_spinner=False)
def get_latest_price(ticker):
    for _ in range(3):
        try:
            data = safe_yf_download(ticker, period="5d", interval="5m")
            if not data.empty:
                p = float(data['Close'].iloc[-1])
                if validate_price(ticker, p):
                    return round(p, 4) if ticker.endswith(("=F", "=X")) else round(p, 2)
        except: pass
        time.sleep(2)
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
# ACCURACY & METADATA
# ================================
def load_accuracy_log(ticker):
    try:
        p = get_accuracy_path(ticker)
        return json.loads(p.read_text()) if p.exists() else {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}
    except:
        return {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}

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

def load_metadata(ticker):
    try:
        p = get_metadata_path(ticker)
        return json.loads(p.read_text()) if p.exists() else {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0}
    except:
        return {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0}

def save_metadata(ticker, meta):
    try:
        get_metadata_path(ticker).write_text(json.dumps(meta, indent=2))
    except: pass

def should_retrain(ticker, acc_log, meta):
    if not get_model_path(ticker).exists():
        return True, ["No model"]
    if acc_log["total_predictions"] >= 10 and acc_log["avg_error"] > LEARNING_CONFIG["accuracy_threshold"]:
        return True, [f"Error {acc_log['avg_error']:.1%}"]
    if meta.get("trained_date"):
        try:
            days = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
            if days >= 30:
                return True, [f"{days}d old"]
        except: pass
    try:
        df = safe_yf_download(ticker, period="30d")
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
    if acc.get("total_predictions", 0) < 12:
        reasons.append("Few live preds")
    if meta.get("retrain_count", 0) < 2:
        reasons.append("Low retrains")
    if acc.get("avg_error", 0.99) > 0.065:
        reasons.append(f"Error {acc['avg_error']:.1%}")
    if meta.get("trained_date"):
        try:
            if (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days > 14:
                reasons.append("Model stale")
        except: pass
    if forecast and current_price:
        move = abs(forecast[0] - current_price) / current_price
        if move > 0.12:
            reasons.append(f"Extreme move {move:+.1%}")
    return len(reasons) == 0, reasons

def ultra_confidence_shield(ticker: str, forecast: List[float], current_price: float) -> Tuple[bool, List[str]]:
    veto = []
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    age = 999
    if meta.get("trained_date"):
        try:
            age = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
        except: pass
    if acc.get("total_predictions", 0) < 25:
        veto.append("Low history")
    if acc.get("avg_error", 0.99) > 0.038:
        veto.append(f"Error {acc['avg_error']:.1%}")
    if meta.get("retrain_count", 0) < 4:
        veto.append("Low retrains")
    if age > 9:
        veto.append(f"Stale {age}d")
    if forecast and current_price:
        if abs(forecast[0] - current_price)/current_price > 0.09:
            veto.append("Insane move")
    return len(veto) == 0, veto

# ================================
# MODEL TRAINING
# ================================
def build_lstm_model():
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], 1)),
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(15),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_self_learning_model(ticker, days=5, force_retrain=False):
    logger.info(f"Training {ticker} (force={force_retrain})")
    updated, acc_log = validate_predictions(ticker)
    meta = load_metadata(ticker)
    needs, reasons = should_retrain(ticker,  acc_log, meta)
    if not (needs or force_retrain):
        return None, None, None

    df = safe_yf_download(ticker, period="1y")
    if df is None or len(df) < 100:
        return None, None, None
    df = df[['Close']].ffill().bfill()

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
            model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0, validation_split=0.1,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
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
        except: pass

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

# ================================
# 6%+ DETECTOR
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def detect_pre_move_6percent(ticker, name):
    try:
        data = safe_yf_download(ticker, period="1d", interval="1m")
        if data is None or len(data) < 60:
            return None
        close = data['Close'].values
        volume = data['Volume'].values
        if len(close) < 30:
            return None

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

        if vol_acceleration > 2.0 and vol_spike_vs_baseline > 3.0:
            score += 30; factors.append(f"Vol×{vol_acceleration:.1f}")
        elif vol_acceleration > 1.5:
            score += 20; factors.append(f"Vol↑{vol_acceleration:.1f}x")
        if momentum_acceleration and abs(recent_momentum) > 0.01:
            score += 25; factors.append("MomentumAccel")
        if abs(recent_momentum) > 0.008:
            score += 15; factors.append("StrongMomentum")

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
            confidence = min(99, 60 + score // 2 + (30 if 'AI→' in ''.join(factors) else 0))
            return {"asset": name, "direction": direction, "confidence": confidence, "factors": factors, "score": score}
        return None
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "detect_pre_move_6percent", e, ticker=ticker, show_to_user=False)
        return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                         data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except:
        return False

# ================================
# BACKTESTING (safe_yf_download used)
# ================================
BACKTEST_DIR = Path("backtest_results")
BACKTEST_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=86400, show_spinner="Running backtest...")
def run_backtest(ticker: str, start_date: str = "2022-01-01", end_date: str = None, initial_capital: float = 10000.0,
                max_position_size: float = 1.0, stop_loss_pct: float = 0.08, take_profit_pct: float = 0.15,
                confidence_threshold: float = 0.65) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    df_full = safe_yf_download(ticker, start=start_date, end=end_date)
    if df_full.empty or len(df_full) < 300:
        return None
    df_full = df_full[['Close']].dropna()
    lookback = LEARNING_CONFIG["lookback_window"]
    min_training_days = lookback + 100
    if len(df_full) <= min_training_days + 1:
        return None

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
        if len(train_df) < lookback + 50:
            continue

        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(train_df[['Close']])
            X, y = [], []
            for j in range(lookback, len(scaled)):
                X.append(scaled[j-lookback:j])
                y.append(scaled[j])
            X, y = np.array(X), np.array(y)
            if len(X) < 50:
                continue

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
        except:
            predicted_return = 0
            confidence = 0

        in_position = position > 0
        unrealized = (current_price - entry_price) / entry_price if in_position else 0

        if in_position:
            if unrealized <= -stop_loss_pct:
                equity *= (1 + unrealized)
                trades.append({"type": "SELL", "reason": "STOP_LOSS", "price": current_price, "return": unrealized*100, "date": current_date.date()})
                position = 0
            elif unrealized >= take_profit_pct:
                equity *= (1 + unrealized)
                trades.append({"type": "SELL", "reason": "TAKE_PROFIT", "price": current_price, "return": unrealized*100, "date": current_date.date()})
                position = 0

        if not in_position and predicted_return > 0.025 and confidence > confidence_threshold:
            position = min(max_position_size, 1.0)
            entry_price = current_price
            trades.append({"type": "BUY", "price": current_price, "confidence": round(confidence, 3), "predicted": round(predicted_return*100, 2), "date": current_date.date()})

        if position > 0:
            daily_return = (next_price - current_price) / current_price
            equity *= (1 + daily_return * position)

        equity_curve.append(equity)
        dates.append(next_day)

    if len(trades) == 0:
        return None

    returns = pd.Series([(df_full['Close'].iloc[i+1] - df_full['Close'].iloc[i]) / df_full['Close'].iloc[i] for i in range(min_training_days, len(df_full)-1)])
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
    win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0
    bh_return = (df_full['Close'].iloc[-1] / df_full['Close'].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity_curve, name="Strategy", line=dict(width=3, color="#00C853")))
    fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[initial_capital, initial_capital * (1 + bh_return/100)], name="Buy & Hold", line=dict(dash="dash", color="#888")))
    fig.update_layout(title=f"{ticker} • Strategy {total_return:+.2f}% vs B&H {bh_return:+.2f}%", template="plotly_dark", height=600)

    result = {
        "ticker": ticker, "final_equity": round(equity, 2), "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2), "sharpe_ratio": round(sharpe, 3), "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1), "buy_and_hold_pct": round(bh_return, 2),
        "fig_equity": fig, "success": True
    }
    return result

# ================================
# FIXED BACKGROUND THREADS
# ================================
def continuous_learning_daemon():
    update_heartbeat("learning_daemon")
    THREAD_START_TIMES["learning_daemon"] = datetime.now()
    logger.info("Learning Daemon STARTED (Smart Mode)")
    cycle_count = 0
    while True:
        try:
            if not load_daemon_config().get("enabled", False):
                time.sleep(60)
                update_heartbeat("learning_daemon")
                continue
            cycle_count += 1
            logger.info(f"Learning Cycle #{cycle_count}")
            trained = 0
            for cat in ASSET_CATEGORIES.values():
                for name, t in cat.items():
                    try:
                        update_heartbeat("learning_daemon")
                        meta = load_metadata(t)
                        acc = load_accuracy_log(t)
                        needs, reasons = should_retrain(t, acc, meta)
                        if needs or random.random() < 0.12:
                            logger.info(f"Training {t} → {' / '.join(reasons) if reasons else 'Routine'}")
                            train_self_learning_model(t, days=1)
                            trained += 1
                        time.sleep(2)
                    except Exception as e:
                        log_error(ErrorSeverity.WARNING, "learning_daemon", e, ticker=t, show_to_user=False)
            logger.info(f"Cycle #{cycle_count} complete — {trained} models updated")
            for _ in range(60):
                time.sleep(60)
                update_heartbeat("learning_daemon")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "learning_daemon_crash", e)
            time.sleep(60)

def monitor_6percent_pre_move():
    update_heartbeat("monitoring")
    THREAD_START_TIMES["monitoring"] = datetime.now()
    logger.info("6%+ Monitoring STARTED (12-min cycle)")
    alerted = set()
    while True:
        try:
            if not load_monitoring_config().get("enabled", False):
                time.sleep(60)
                update_heartbeat("monitoring")
                continue
            update_heartbeat("monitoring")
            for cat in ASSET_CATEGORIES.values():
                for name, t in cat.items():
                    try:
                        signal = detect_pre_move_6percent(t, name)
                        if signal and signal["confidence"] >= 92:
                            key = f"{t}_{signal['direction']}"
                            if key not in alerted:
                                current_price = get_latest_price(t)
                                if current_price:
                                    forecast, _, _ = train_self_learning_model(t, days=1)
                                    if forecast:
                                        passed, reasons = ultra_confidence_shield(t, forecast, current_price)
                                        if passed:
                                            text = f"NUCLEAR ALERT\n{signal['asset']} → {signal['direction']}\nConfidence: {signal['confidence']}%\n{', '.join(signal['factors'])}"
                                            if send_telegram_alert(text):
                                                alerted.add(key)
                                                logger.info(f"ALERT SENT: {signal['asset']}")
                    except Exception as e:
                        log_error(ErrorSeverity.WARNING, "monitor_scan", e, ticker=t, show_to_user=False)
            for _ in range(24):
                time.sleep(30)
                update_heartbeat("monitoring")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "monitoring_crash", e)
            time.sleep(60)

def thread_watchdog():
    update_heartbeat("watchdog")
    THREAD_START_TIMES["watchdog"] = datetime.now()
    logger.info("Watchdog STARTED")
    while True:
        try:
            update_heartbeat("watchdog")
            for name in ["learning_daemon", "monitoring"]:
                status = get_thread_status(name)
                enabled = (name == "learning_daemon" and load_daemon_config().get("enabled", False)) or \
                          (name == "monitoring" and load_monitoring_config().get("enabled", False))
                if enabled and status["status"] in ["WARNING", "DEAD"]:
                    logger.warning(f"Thread {name} → {status['status']} ({status['seconds_since']}s)")
            time.sleep(30)
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "watchdog", e)
            time.sleep(30)

def show_error_dashboard():
    st.subheader("System Diagnostics")
    stats = get_error_statistics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Errors", stats["total"])
    with col2:
        st.metric("Critical", stats["by_severity"].get("CRITICAL", 0))
    with col3:
        st.metric("Warnings", stats["by_severity"].get("WARNING", 0))
    if stats["recent"]:
        with st.expander("Recent Errors"):
            for e in stats["recent"][::-1]:
                st.write(f"**{e['severity']}** | {e['function']} | {e['user_message']}")

# ================================
# STREAMLIT APP
# ================================
st.set_page_config(page_title="AI Alpha Tracker v4.3", layout="wide")

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = {}
if 'learning_log' not in st.session_state:
    st.session_state.learning_log = []
if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []

def initialize_background_threads():
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        logger.info("Initializing background threads...")
        threading.Thread(target=thread_watchdog, daemon=True).start()
        if load_daemon_config().get("enabled", False):
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
        if load_monitoring_config().get("enabled", False):
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()

initialize_background_threads()

def add_header():
    st.markdown("<div style='text-align:center;padding:20px;background:#0f0f0f;color:#00ff41;border-radius:12px;'><h1>AI ALPHA TRACKER v4.3</h1><p>Self-Learning • Nuclear Alerts • 24/7</p></div>", unsafe_allow_html=True)

def add_footer():
    st.markdown("<div style='text-align:center;padding:20px;background:#0f0f0f;color:#666;margin-top:50px;border-radius:12px;'><p>© 2025 AI Alpha Tracker v4.3 | Final Form</p></div>", unsafe_allow_html=True)

add_header()

with st.sidebar:
    st.header("Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    st.markdown("---")
    st.subheader("Controls")

    if st.button("Force Retrain", width="stretch"):
        with st.spinner("Retraining..."):
            train_self_learning_model(ticker, force_retrain=True)
        st.success("Done!")
        st.rerun()

    st.markdown("---")
    st.subheader("Learning Daemon")
    dc = load_daemon_config()
    st.write("**Status:**", "RUNNING" if dc.get("enabled") else "STOPPED")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", key="dstart", width="stretch"):
            save_daemon_config(True)
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.rerun()
    with c2:
        if st.button("Stop", key="dstop", width="stretch"):
            save_daemon_config(False)
            st.rerun()

    st.markdown("---")
    st.subheader("6%+ Monitoring")
    mc = load_monitoring_config()
    st.write("**Status:**", "RUNNING" if mc.get("enabled") else "STOPPED")
    if st.button("Test Telegram", width="stretch"):
        success = send_telegram_alert("TEST ALERT\nAI Alpha Tracker v4.3 Online")
        st.success("Sent!") if success else st.error("Failed")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", key="mstart", width="stretch"):
            save_monitoring_config(True)
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.rerun()
    with c2:
        if st.button("Stop", key="mstop", width="stretch"):
            save_monitoring_config(False)
            st.rerun()

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(f"<h2 style='text-align:center;'>LIVE: <code style='font-size:2.5em;background:#111;padding:15px 30px;border-radius:15px;'>${price:.2f}</code></h2>", unsafe_allow_html=True)
    else:
        st.warning("Market closed or no data")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "5-Day Forecast", "All Models", "Backtesting", "Diagnostics"])

with tab1:
    if st.button("Daily Recommendation", width="stretch"):
        with st.spinner("Analyzing..."):
            forecast, _, _ = train_self_learning_model(ticker, days=1)
            if forecast is not None and len(forecast) > 0:
                forecast_val = float(forecast[0])
                passed, reasons = high_confidence_checklist(ticker, [forecast_val], price or 100)
                if passed and price:
                    change_pct = (forecast_val - price) / price * 100
                    action = "STRONG BUY" if change_pct >= 3 else "BUY" if change_pct >= 1.5 else "SELL" if change_pct <= -1.5 else "HOLD"
                    st.success(f"AI Predicts: ${forecast_val:.2f} ({change_pct:+.2f}%) → **{action}**")
                else:
                    st.warning("Low Confidence\n\n" + "\n".join([f"• {r}" for r in reasons]))
            else:
                st.error("No forecast")

with tab5:
    show_error_dashboard()

add_footer()
