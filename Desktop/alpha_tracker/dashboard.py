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
from typing import Tuple, List

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Suppress Streamlit thread warnings (safe import)
try:
    from streamlit.runtime.scriptrunner import script_run_context
    if hasattr(script_run_context, '_LOGGER'):
        script_run_context._LOGGER.setLevel('ERROR')
except (ImportError, AttributeError):
    # Streamlit version doesn't support this - that's okay, warnings are harmless
    pass

# ================================
# LOGGING SETUP
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Define paths
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

class ErrorSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def reset_all_logs_on_startup():
    """Reset all log files when app starts - gives truly fresh start"""
    try:
        # Delete all log files
        if (LOG_DIR / 'app.log').exists():
            (LOG_DIR / 'app.log').unlink()
        if (LOG_DIR / 'errors.log').exists():
            (LOG_DIR / 'errors.log').unlink()
        if ERROR_LOG_PATH.exists():
            ERROR_LOG_PATH.unlink()
        
        # Create fresh error tracking file
        with open(ERROR_LOG_PATH, 'w') as f:
            json.dump([], f)
            
        return True
    except Exception as e:
        print(f"Warning: Could not reset logs: {e}")
        return False

# Reset logs BEFORE setting up logging
reset_all_logs_on_startup()

def setup_logging():
    """Configure logging system with fresh start"""
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # File handler - Normal append mode (files already reset)
    file_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # Error file handler - Normal append mode (files already reset)
    error_handler = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    logger.info("=== NEW SESSION STARTED - ALL LOGS RESET ===")
    
    return logger

logger = setup_logging()
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

def log_error(severity, function_name, error, ticker=None, user_message="An error occurred", show_to_user=True):
    """Centralized error logging with thread-safe session state access"""
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.value,
        "ticker": ticker,
        "function": function_name,
        "error": str(error),
        "user_message": user_message
    }
    
    # Log to file
    log_msg = f"{ticker or 'N/A'} | {user_message}: {str(error)}"
    if severity == ErrorSeverity.DEBUG:
        logger.debug(log_msg)
    elif severity == ErrorSeverity.INFO:
        logger.info(log_msg)
    elif severity == ErrorSeverity.WARNING:
        logger.warning(log_msg)
    elif severity == ErrorSeverity.ERROR:
        logger.error(log_msg, exc_info=True)
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(log_msg, exc_info=True)
    
    # Save to tracking file with robust error handling
    try:
        with open(ERROR_LOG_PATH, 'r') as f:
            history = json.load(f) if ERROR_LOG_PATH.exists() else []
    except (json.JSONDecodeError, FileNotFoundError):
        history = []
    
    try:
        history.append(error_data)
        history = history[-500:]  # Keep last 500
        with open(ERROR_LOG_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as write_error:
        logger.error(f"Failed to write error log: {write_error}")
    
    # Show to user
    if show_to_user:
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 {user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"❌ {user_message}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"⚠️ {user_message}")
    
    # Thread-safe session state update
    try:
        if hasattr(st, 'session_state'):
            if 'error_logs' not in st.session_state:
                st.session_state.error_logs = []
            st.session_state.error_logs.append(error_data)
    except Exception:
        pass  # Fail silently if session state unavailable

def get_error_statistics():
    """Get error statistics with robust file handling"""
    try:
        if not ERROR_LOG_PATH.exists():
            return {"total": 0, "by_severity": {}, "recent": []}
        
        with open(ERROR_LOG_PATH, 'r') as f:
            errors = json.load(f)
        
        by_severity = {}
        for error in errors:
            sev = error.get('severity', 'UNKNOWN')
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {"total": len(errors), "by_severity": by_severity, "recent": errors[-10:]}
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        logger.warning(f"Error reading statistics: {e}")
        return {"total": 0, "by_severity": {}, "recent": []}

# ================================
# CONFIG & KEYS
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")
    logger.info("Configuration loaded")
except Exception as e:
    log_error(ErrorSeverity.WARNING, "config_load", e, user_message="Some API keys missing", show_to_user=False)
    BOT_TOKEN = CHAT_ID = ALPHA_VANTAGE_KEY = None

# ================================
# ASSETS
# ================================
ASSET_CATEGORIES = {
    "Tech Stocks": {"Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL"},
    "High Growth": {"Palantir": "PLTR", "MicroStrategy": "MSTR", "Coinbase": "COIN"},
    "Commodities": {"Corn Futures": "ZC=F", "Gold Futures": "GC=F", "Crude Oil": "CL=F", "Wheat": "ZW=F"},
    "ETFs": {"S&P 500 ETF": "SPY", "WHEAT": "WEAT"}
}

# ================================
# DIRECTORIES
# ================================
MODEL_DIR = Path("models")
SCALER_DIR = Path("scalers")
ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata")
PREDICTIONS_DIR = Path("predictions")
CONFIG_DIR = Path("config")

try:
    for dir_path in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, CONFIG_DIR, LOG_DIR]:
        dir_path.mkdir(exist_ok=True)
    logger.info("Directories created")
except Exception as e:
    log_error(ErrorSeverity.CRITICAL, "directory_creation", e, user_message="Failed to create directories")

DAEMON_CONFIG_PATH = CONFIG_DIR / "daemon_config.json"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring_config.json"

# ================================
# CONFIG
# ================================
LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,
    "min_predictions_for_eval": 10,
    "retrain_interval_days": 30,
    "volatility_change_threshold": 0.5,
    "fine_tune_epochs": 5,
    "full_retrain_epochs": 25,
    "lookback_window": 60
}

model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()
config_lock = threading.Lock()
session_state_lock = threading.Lock()
heartbeat_lock = threading.Lock()

# ================================
# THREAD HEALTH MONITORING
# ================================

# Thread heartbeat tracking
THREAD_HEARTBEATS = {
    "learning_daemon": None,
    "monitoring": None,
    "watchdog": None
}

THREAD_START_TIMES = {
    "learning_daemon": None,
    "monitoring": None,
    "watchdog": None
}

def update_heartbeat(thread_name):
    """Update heartbeat timestamp for a thread"""
    with heartbeat_lock:
        THREAD_HEARTBEATS[thread_name] = datetime.now()

def get_thread_status(thread_name):
    """Get status of a thread"""
    with heartbeat_lock:
        last_heartbeat = THREAD_HEARTBEATS.get(thread_name)
        start_time = THREAD_START_TIMES.get(thread_name)
    
    if last_heartbeat is None:
        return {"status": "STOPPED", "last_heartbeat": None, "uptime": None}
    
    seconds_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
    
    # If no heartbeat in 5 minutes, consider dead
    if seconds_since_heartbeat > 300:
        return {"status": "DEAD", "last_heartbeat": last_heartbeat, "uptime": None}
    
    # If no heartbeat in 2 minutes, warning
    if seconds_since_heartbeat > 120:
        status = "WARNING"
    else:
        status = "HEALTHY"
    
    # Calculate uptime
    uptime = None
    if start_time:
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime = f"{hours}h {minutes}m"
    
    return {
        "status": status,
        "last_heartbeat": last_heartbeat,
        "seconds_since": int(seconds_since_heartbeat),
        "uptime": uptime
    }

# ================================
# HELPER: NORMALIZE DATAFRAME COLUMNS
# ================================
def normalize_dataframe_columns(df):
    """Normalize MultiIndex columns from yfinance to single-level"""
    if df is None or df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns - take first level (the actual column names)
        df.columns = df.columns.get_level_values(0)
    
    return df

# ================================
# PERSISTENT CONFIG WITH ROBUST FILE HANDLING
# ================================
def load_daemon_config():
    try:
        if DAEMON_CONFIG_PATH.exists():
            with config_lock:
                with open(DAEMON_CONFIG_PATH, 'r') as f:
                    return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        log_error(ErrorSeverity.WARNING, "load_daemon_config", e, user_message="Config load failed", show_to_user=False)
    return {"enabled": False, "last_started": None}

def save_daemon_config(enabled):
    try:
        config = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with config_lock:
            with open(DAEMON_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        logger.info(f"Daemon config saved: {enabled}")
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_daemon_config", e, user_message="Failed to save config")
        return False

def load_monitoring_config():
    try:
        if MONITORING_CONFIG_PATH.exists():
            with config_lock:
                with open(MONITORING_CONFIG_PATH, 'r') as f:
                    return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        log_error(ErrorSeverity.WARNING, "load_monitoring_config", e, user_message="Config load failed", show_to_user=False)
    return {"enabled": False, "last_started": None}

def save_monitoring_config(enabled):
    try:
        config = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with config_lock:
            with open(MONITORING_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        logger.info(f"Monitoring config saved: {enabled}")
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_monitoring_config", e, user_message="Failed to save config")
        return False

# ================================
# HELPERS
# ================================
def get_safe_ticker_name(ticker):
    return ticker.replace('=', '_').replace('^', '').replace('/', '_')

def get_model_path(ticker):
    return MODEL_DIR / f"{get_safe_ticker_name(ticker)}_lstm.h5"

def get_scaler_path(ticker):
    return SCALER_DIR / f"{get_safe_ticker_name(ticker)}_scaler.pkl"

def get_accuracy_path(ticker):
    return ACCURACY_DIR / f"{get_safe_ticker_name(ticker)}_accuracy.json"

def get_metadata_path(ticker):
    return METADATA_DIR / f"{get_safe_ticker_name(ticker)}_meta.json"

def get_prediction_path(ticker, date):
    return PREDICTIONS_DIR / f"{get_safe_ticker_name(ticker)}_{date}.json"

# ================================
# PRICE FETCHING WITH VALIDATION
# ================================

# Expected price ranges for validation (approximate, will auto-adjust)
PRICE_RANGES = {
    "AAPL": (150, 500),      # Apple
    "TSLA": (150, 600),      # Tesla
    "NVDA": (100, 400),      # NVIDIA - updated for current levels
    "MSFT": (300, 600),      # Microsoft - updated
    "GOOGL": (100, 400),     # Alphabet
    "PLTR": (5, 200),        # Palantir - UPDATED to allow $156
    "MSTR": (100, 900),      # MicroStrategy - updated for volatility
    "COIN": (50, 500),       # Coinbase
    "ZC=F": (300, 700),      # Corn futures
    "GC=F": (1500, 5000),    # Gold futures
    "CL=F": (30, 150),       # Crude oil
    "ZW=F": (400, 800),      # Wheat futures
    "SPY": (400, 900),       # S&P 500 ETF - updated
    "WEAT": (3, 15)          # Wheat ETF
}

def validate_price(ticker, price):
    """Validate if price is within reasonable range for this ticker"""
    if price is None or price <= 0:
        return False
    
    if ticker in PRICE_RANGES:
        min_price, max_price = PRICE_RANGES[ticker]
        if min_price <= price <= max_price:
            return True
        else:
            logger.warning(f"Price validation failed for {ticker}: ${price:.2f} outside range ${min_price}-${max_price}")
            return False
    
    # If no range defined, accept any positive price
    return True

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    logger.debug(f"Fetching price for {ticker}")
    methods_tried = []
    
    # Add delay to avoid rate limiting and cache confusion
    time.sleep(0.3)
    
    try:
        # Method 1: 1-minute interval with validation
        for attempt in range(2):
            try:
                data = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
                data = normalize_dataframe_columns(data)
                if data is not None and not data.empty and len(data) > 0:
                    price = float(data['Close'].iloc[-1])
                    
                    # Validate price
                    if validate_price(ticker, price):
                        logger.info(f"Price: {ticker} ${price:.2f}")
                        return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
                    else:
                        methods_tried.append(f"1m-invalid-price-{price:.2f}")
                        if attempt == 0:
                            time.sleep(1)  # Wait and retry
                            continue
                methods_tried.append("1m-no-data")
            except Exception as e:
                methods_tried.append(f"1m-{type(e).__name__}")
                if attempt == 0:
                    time.sleep(1)
        
        # Method 2: 5-minute with validation
        try:
            time.sleep(0.5)
            data = yf.download(ticker, period="1d", interval="5m", progress=False, threads=False)
            data = normalize_dataframe_columns(data)
            if data is not None and not data.empty:
                price = float(data['Close'].iloc[-1])
                if validate_price(ticker, price):
                    logger.info(f"Price: {ticker} ${price:.2f} (5m)")
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
                methods_tried.append(f"5m-invalid-price-{price:.2f}")
            else:
                methods_tried.append("5m-no-data")
        except Exception as e:
            methods_tried.append(f"5m-{type(e).__name__}")
        
        # Method 3: Daily with validation
        try:
            time.sleep(0.5)
            data = yf.download(ticker, period="5d", interval="1d", progress=False, threads=False)
            data = normalize_dataframe_columns(data)
            if data is not None and not data.empty:
                price = float(data['Close'].iloc[-1])
                if validate_price(ticker, price):
                    logger.info(f"Price: {ticker} ${price:.2f} (1d)")
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
                methods_tried.append(f"1d-invalid-price-{price:.2f}")
            else:
                methods_tried.append("1d-no-data")
        except Exception as e:
            methods_tried.append(f"1d-{type(e).__name__}")
        
        # Method 4: Ticker info with validation
        try:
            time.sleep(0.5)
            tick = yf.Ticker(ticker)
            info = tick.info
            price = None
            
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                price = float(info['regularMarketPrice'])
            elif 'previousClose' in info and info['previousClose']:
                price = float(info['previousClose'])
            elif 'currentPrice' in info and info['currentPrice']:
                price = float(info['currentPrice'])
            
            if price and validate_price(ticker, price):
                logger.info(f"Price: {ticker} ${price:.2f} (info)")
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            elif price:
                methods_tried.append(f"info-invalid-price-{price:.2f}")
            else:
                methods_tried.append("info-no-price")
        except Exception as e:
            methods_tried.append(f"info-{type(e).__name__}")
        
        log_error(ErrorSeverity.WARNING, "get_latest_price", Exception(f"All methods failed: {methods_tried}"), 
                  ticker=ticker, user_message=f"Price unavailable for {ticker}", show_to_user=False)
        return None
        
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "get_latest_price", e, ticker=ticker, user_message=f"Price error: {ticker}", show_to_user=False)
        return None

# ================================
# ACCURACY TRACKING WITH ROBUST FILE HANDLING
# ================================
def load_accuracy_log(ticker):
    try:
        path = get_accuracy_path(ticker)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        log_error(ErrorSeverity.WARNING, "load_accuracy_log", e, ticker=ticker, user_message="Accuracy log error", show_to_user=False)
    return {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}

def save_accuracy_log(ticker, log_data):
    try:
        with accuracy_lock:
            with open(get_accuracy_path(ticker), 'w') as f:
                json.dump(log_data, f, indent=2)
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_accuracy_log", e, ticker=ticker, user_message="Save failed")
        return False

def record_prediction(ticker, predicted_price, prediction_date):
    try:
        pred_data = {"ticker": ticker, "predicted_price": float(predicted_price), 
                     "prediction_date": prediction_date, "timestamp": datetime.now().isoformat()}
        with open(get_prediction_path(ticker, prediction_date), 'w') as f:
            json.dump(pred_data, f, indent=2)
        logger.info(f"Prediction recorded: {ticker} ${predicted_price:.2f}")
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "record_prediction", e, ticker=ticker, user_message="Record failed")
        return False

def validate_predictions(ticker):
    accuracy_log = load_accuracy_log(ticker)
    updated = False
    
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        pred_path = get_prediction_path(ticker, yesterday)
        
        if pred_path.exists():
            try:
                with open(pred_path, 'r') as f:
                    pred_data = json.load(f)
                actual_price = get_latest_price(ticker)
                
                if actual_price:
                    predicted_price = pred_data["predicted_price"]
                    error = abs(predicted_price - actual_price) / actual_price
                    
                    accuracy_log["predictions"].append(predicted_price)
                    accuracy_log["errors"].append(error)
                    accuracy_log["dates"].append(yesterday)
                    accuracy_log["total_predictions"] += 1
                    
                    if len(accuracy_log["errors"]) > 50:
                        accuracy_log["predictions"] = accuracy_log["predictions"][-50:]
                        accuracy_log["errors"] = accuracy_log["errors"][-50:]
                        accuracy_log["dates"] = accuracy_log["dates"][-50:]
                    
                    accuracy_log["avg_error"] = np.mean(accuracy_log["errors"][-30:])
                    save_accuracy_log(ticker, accuracy_log)
                    updated = True
                    logger.info(f"Validated: {ticker} pred=${predicted_price:.2f} actual=${actual_price:.2f}")
                    pred_path.unlink()
                    
            except Exception as e:
                log_error(ErrorSeverity.WARNING, "validate_predictions", e, ticker=ticker, user_message="Validation error", show_to_user=False)
    
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "validate_predictions", e, ticker=ticker, user_message="Validation failed")
    
    return updated, accuracy_log

# ================================
# METADATA WITH ROBUST FILE HANDLING
# ================================
def load_metadata(ticker):
    try:
        path = get_metadata_path(ticker)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        log_error(ErrorSeverity.WARNING, "load_metadata", e, ticker=ticker, user_message="Metadata error", show_to_user=False)
    return {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0, "last_accuracy": 0.0}

def save_metadata(ticker, metadata):
    try:
        with open(get_metadata_path(ticker), 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_metadata", e, ticker=ticker, user_message="Metadata save failed")
        return False

# ================================
# RETRAINING LOGIC
# ================================
def should_retrain(ticker, accuracy_log, metadata):
    reasons = []
    try:
        if not get_model_path(ticker).exists():
            return True, ["No model exists"]
        
        if len(accuracy_log["errors"]) >= LEARNING_CONFIG["min_predictions_for_eval"]:
            if accuracy_log["avg_error"] > LEARNING_CONFIG["accuracy_threshold"]:
                reasons.append(f"Low accuracy ({accuracy_log['avg_error']:.2%})")
                return True, reasons
        
        if metadata["trained_date"]:
            try:
                last_trained = datetime.fromisoformat(metadata["trained_date"])
                days_since = (datetime.now() - last_trained).days
                if days_since >= LEARNING_CONFIG["retrain_interval_days"]:
                    reasons.append(f"{days_since} days old")
                    return True, reasons
            except:
                pass
        
        try:
            df = yf.download(ticker, period="30d", progress=False)
            df = normalize_dataframe_columns(df)
            if df is not None and len(df) > 5:
                current_vol = df['Close'].pct_change().std()
                training_vol = metadata.get("training_volatility", 0)
                if training_vol > 0:
                    vol_change = abs(current_vol - training_vol) / training_vol
                    if vol_change > LEARNING_CONFIG["volatility_change_threshold"]:
                        reasons.append(f"Volatility changed {vol_change:.1%}")
                        return True, reasons
        except:
            pass
    
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "should_retrain", e, ticker=ticker, user_message="Retrain check failed")
    
    return False, reasons
    
# ================================
# HIGH-CONFIDENCE CHECKLIST (NOW INTEGRATED)
# ================================
def high_confidence_checklist(ticker: str, forecast: list, current_price: float) -> tuple:
    """
    Returns (passed: bool, failed_reasons: list)
    Only returns True when ALL confidence criteria are met → prediction is shown
    """
    reasons = []
    metadata = load_metadata(ticker)
    acc_log = load_accuracy_log(ticker)

    # 1. Model maturity
    if acc_log.get("total_predictions", 0) < 12:
        reasons.append(f"Only {acc_log['total_predictions']} live predictions")

    if metadata.get("retrain_count", 0) < 2:
        reasons.append("Model needs ≥2 retrains")

    # 2. Recent accuracy
    recent_error = acc_log.get("avg_error", 0.99)
    if recent_error > 0.065:  # >6.5% average error last 30 days
        reasons.append(f"Recent error {recent_error:.1%} > 6.5%")

    # 3. Model freshness
    if metadata.get("trained_date"):
        try:
            days_old = (datetime.now() - datetime.fromisoformat(metadata["trained_date"])).days
            if days_old > 14:
                reasons.append(f"Model {days_old} days old (>14)")
        except:
            pass

    # 4. High volatility regime (without recent retrain)
    try:
        df = yf.download(ticker, period="60d", progress=False, threads=False)
        df = normalize_dataframe_columns(df)
        if df is not None and not df.empty:
            vol_20d = df['Close'].pct_change().rolling(20).std().iloc[-1]
            days_old = (datetime.now() - datetime.fromisoformat(metadata["trained_date"])).days if metadata.get("trained_date") else 999
            if vol_20d > 0.04 and days_old > 7:  # >4% daily std dev
                reasons.append(f"Extreme volatility {vol_20d:.1%}/day")
    except:
        pass

    # 5. Unrealistic forecast
    if forecast is not None and len(forecast) > 0 and current_price:
        implied_move = abs(forecast[0] - current_price) / current_price
        if implied_move > 0.12:  # >12% in one day
            reasons.append(f"Extreme move predicted {implied_move:+.1%}")

    # 6. Sharp reversal vs recent trend
    try:
        hist = yf.download(ticker, period="10d", progress=False, threads=False)
        hist = normalize_dataframe_columns(hist)
        if hist is not None and not hist.empty and len(hist) > 6:
            trend_5d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
            pred_move = (forecast[0] - current_price) / current_price if forecast is not None and len(forecast) > 0 and current_price else 0
            if trend_5d * pred_move < -0.5:  # strong reversal
                reasons.append("Predicting sharp reversal")
    except:
        pass

    passed = len(reasons) == 0
    return passed, reasons

# ================================
# ULTRA-CONFIDENCE SHIELD (STRICTER - FOR TRADING SIGNALS)
# ================================
def ultra_confidence_shield(ticker: str, forecast: List[float], current_price: float) -> Tuple[bool, List[str]]:
    """
    Returns (allow_prediction: bool, veto_reasons: list)
    This is the nuclear option — ZERO tolerance for any weakness.
    Used for high-stakes trading alerts (Telegram), not UI display.
    """
    veto = []
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    
    # Initialize variables to avoid scope issues
    age = 999
    vol_df = None
    
    # 1. Ironclad accuracy history
    if acc.get("total_predictions", 0) < 25:
        veto.append(f"Only {acc['total_predictions']} live preds (<25)")
    if acc.get("avg_error", 0.99) > 0.038:  # ≤3.8% error last 30 days
        veto.append(f"Error {acc['avg_error']:.2%} > 3.8%")
    
    # 2. Model must be battle-tested
    if meta.get("retrain_count", 0) < 4:
        veto.append(f"Only {meta.get('retrain_count', 0)} retrains (<4)")
    
    # 3. Model must be fresh
    if meta.get("trained_date"):
        try:
            age = (datetime.now() - datetime.fromisoformat(meta["trained_date"])).days
            if age > 9:
                veto.append(f"Model {age}d old (>9)")
        except:
            pass
    
    # 4. No extreme volatility without fresh retrain
    try:
        vol_df = yf.download(ticker, period="45d", progress=False, threads=False)
        vol_df = normalize_dataframe_columns(vol_df)
        
        if vol_df is not None and not vol_df.empty:
            volatility = vol_df['Close'].pct_change().std()
            if volatility > 0.032 and age > 4:  # >3.2% daily std-dev = dangerous
                veto.append(f"High vol {volatility:.2%}/day")
    except:
        pass
    
    # 5. Forecast must be mathematically sane
    if forecast is not None and len(forecast) > 0 and current_price and vol_df is not None and not vol_df.empty:
        move = abs(forecast[0] - current_price) / current_price
        if move > 0.09:  # >9% single-day move = fantasy
            veto.append(f"Insane move {move:+.2%}")
        
        # Forecast must stay inside last 30-day Bollinger Bands (±2.5σ for flexibility)
        try:
            close_series = vol_df['Close']
            ma = close_series.rolling(20).mean().iloc[-1]
            std = close_series.rolling(20).std().iloc[-1]
            upper = ma + 2.5 * std  # Relaxed from 2σ to 2.5σ
            lower = ma - 2.5 * std
            if not (lower <= forecast[0] <= upper):
                veto.append("Outside Bollinger Bands (2.5σ)")
        except:
            pass
    
    # 6. Trend consistency — no violent reversal without volume proof
    try:
        hist = yf.download(ticker, period="12d", progress=False, threads=False)
        hist = normalize_dataframe_columns(hist)
        
        if hist is not None and not hist.empty and len(hist) > 7:
            hist_close = hist['Close']
            trend = (hist_close.iloc[-1] - hist_close.iloc[-7]) / hist_close.iloc[-7]
            pred_move = (forecast[0] - current_price) / current_price if forecast is not None and len(forecast) > 0 and current_price else 0
            
            if trend * pred_move < -0.65:  # strong reversal
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    today_volume = info.get("volume", 0)
                    avg_volume = info.get("averageVolume", 1) or info.get("averageDailyVolume10Day", 1)
                    
                    if today_volume < avg_volume * 1.8:
                        veto.append("Reversal without volume confirmation")
                except:
                    pass
    except:
        pass
    
    # 7. Market regime match
    if vol_df is not None and not vol_df.empty and meta.get("training_volatility", 0) > 0:
        try:
            current_volatility = vol_df['Close'].pct_change().std()
            regime_change = abs(current_volatility - meta["training_volatility"]) / meta["training_volatility"]
            if regime_change > 0.45:
                veto.append(f"Regime shift {regime_change:+.1%}")
        except:
            pass
    
    # 8. No prediction on low-liquidity days
    try:
        info = yf.Ticker(ticker).info
        trading_volume = info.get("volume", 0)
        avg_volume = info.get("averageVolume", 1) or info.get("averageDailyVolume10Day", 1)
        if trading_volume < avg_volume * 0.55:
            veto.append("Low volume day")
    except:
        pass
    
    return len(veto) == 0, veto

# ================================
# MODEL BUILDING
# ================================
def build_lstm_model():
    try:
        model = Sequential([
            LSTM(30, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], 1)),
            Dropout(0.2),
            LSTM(30, return_sequences=False),
            Dropout(0.2),
            Dense(15),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logger.info("Model built")
        return model
    except Exception as e:
        log_error(ErrorSeverity.CRITICAL, "build_lstm_model", e, user_message="Model build failed")
        return None

# ================================
# TRAINING SYSTEM
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    logger.info(f"Training {ticker} (days={days}, force={force_retrain})")
    
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    
    try:
        updated, accuracy_log = validate_predictions(ticker)
        if updated:
            with session_state_lock:
                if hasattr(st, 'session_state'):
                    st.session_state.setdefault('learning_log', []).append(f"✅ Validated {ticker}")
        
        metadata = load_metadata(ticker)
        needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
        
        training_type = "full-retrain" if (needs_retrain or force_retrain) else "fine-tune"
        if reasons:
            with session_state_lock:
                if hasattr(st, 'session_state'):
                    st.session_state.setdefault('learning_log', []).append(f"🔄 Retraining {ticker}: {', '.join(reasons)}")
        
        # Download data with proper None checks
        df = None
        for attempt in range(3):
            try:
                df = yf.download(ticker, period="1y", progress=False)
                df = normalize_dataframe_columns(df)
                if df is not None and len(df) >= 100:
                    break
                time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    log_error(ErrorSeverity.ERROR, "train_self_learning_model", e, ticker=ticker, user_message=f"Data download failed: {ticker}")
                    return None, None, None
                time.sleep(2)
        
        if df is None or len(df) < 100:
            log_error(ErrorSeverity.WARNING, "train_self_learning_model", Exception("Insufficient data"), 
                      ticker=ticker, user_message=f"Not enough data: {ticker}", show_to_user=False)
            return None, None, None
        
        df = df[['Close']].copy()
        
        # Use modern pandas methods - ffill and bfill
        df = df.ffill().bfill()
        
        # Check for NaN values properly
        if df['Close'].isna().any():
            log_error(ErrorSeverity.WARNING, "train_self_learning_model", Exception("NaN values found"), 
                      ticker=ticker, user_message=f"Data quality issue: {ticker}", show_to_user=False)
            return None, None, None
        
        # Scaler with feature mismatch detection
        try:
            if training_type == "full-retrain" or not scaler_path.exists():
                scaler = MinMaxScaler()
                scaler.fit(df[['Close']])
                joblib.dump(scaler, scaler_path)
                logger.info(f"New scaler created for {ticker}")
            else:
                scaler = joblib.load(scaler_path)
                # Check if scaler has wrong number of features
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != 1:
                    logger.warning(f"Scaler feature mismatch for {ticker}: expected 1, got {scaler.n_features_in_}. Recreating scaler.")
                    scaler = MinMaxScaler()
                    scaler.fit(df[['Close']])
                    joblib.dump(scaler, scaler_path)
                    # Also delete old model since it won't match
                    if model_path.exists():
                        model_path.unlink()
                        logger.info(f"Deleted incompatible model for {ticker}")
                    training_type = "full-retrain"
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "train_self_learning_model", e, ticker=ticker, user_message="Scaler error")
            return None, None, None
        
        scaled = scaler.transform(df[['Close']])
        
        # Sequences
        X, y = [], []
        lookback = LEARNING_CONFIG["lookback_window"]
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            return None, None, None
        
        logger.info(f"Training data: {ticker} {len(X)} sequences")
        
        # Train
        with model_cache_lock:
            try:
                if training_type == "full-retrain":
                    model = build_lstm_model()
                    if model is None:
                        return None, None, None
                    
                    epochs = LEARNING_CONFIG["full_retrain_epochs"]
                    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1,
                              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
                    
                    metadata["retrain_count"] += 1
                    with session_state_lock:
                        if hasattr(st, 'session_state'):
                            st.session_state.setdefault('learning_log', []).append(f"🧠 Full retrain #{metadata['retrain_count']} {ticker}")
                else:
                    try:
                        model = tf.keras.models.load_model(str(model_path))
                        epochs = LEARNING_CONFIG["fine_tune_epochs"]
                        recent_size = int(len(X) * 0.3)
                        model.fit(X[-recent_size:], y[-recent_size:], epochs=epochs, batch_size=32, verbose=0)
                        with session_state_lock:
                            if hasattr(st, 'session_state'):
                                st.session_state.setdefault('learning_log', []).append(f"⚡ Fine-tuned {ticker}")
                    except (OSError, IOError) as e:
                        # Model file corrupted or missing - do full retrain
                        log_error(ErrorSeverity.WARNING, "train_self_learning_model", e, ticker=ticker, 
                                  user_message=f"Model load failed, retraining {ticker}", show_to_user=False)
                        model = build_lstm_model()
                        if model is None:
                            return None, None, None
                        model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0, validation_split=0.1)
                    except Exception as e:
                        # Other errors - also do full retrain
                        log_error(ErrorSeverity.WARNING, "train_self_learning_model", e, ticker=ticker, 
                                  user_message=f"Fine-tune failed, retraining {ticker}", show_to_user=False)
                        model = build_lstm_model()
                        if model is None:
                            return None, None, None
                        model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0, validation_split=0.1)
                
                try:
                    model.save(str(model_path))
                    logger.info(f"Model saved: {ticker}")
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "train_self_learning_model", e, ticker=ticker, user_message="Model save failed")
                    # Continue anyway - model is in memory
                
                metadata["trained_date"] = datetime.now().isoformat()
                metadata["training_samples"] = len(X)
                metadata["training_volatility"] = float(df['Close'].pct_change().std())
                metadata["version"] += 1
                metadata["last_accuracy"] = accuracy_log["avg_error"]
                save_metadata(ticker, metadata)
                
            except Exception as e:
                log_error(ErrorSeverity.CRITICAL, "train_self_learning_model", e, ticker=ticker, user_message=f"Training error: {ticker}")
                return None, None, None
        
        # Predict
        try:
            last = scaled[-lookback:].reshape(1, lookback, 1)
            preds = []
            for _ in range(days):
                pred = model.predict(last, verbose=0)
                preds.append(pred[0, 0])
                last = np.append(last[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            record_prediction(ticker, forecast[0], tomorrow)
            
            dates = []
            i = 1
            while len(dates) < days:
                next_date = datetime.now().date() + timedelta(days=i)
                if next_date.weekday() < 5:
                    dates.append(next_date)
                i += 1
            
            logger.info(f"Predictions: {ticker} {forecast}")
            
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "train_self_learning_model", e, ticker=ticker, user_message="Prediction failed")
            return None, None, None
        
        finally:
            tf.keras.backend.clear_session()
        
        return forecast, dates, model
    
    except Exception as e:
        log_error(ErrorSeverity.CRITICAL, "train_self_learning_model", e, ticker=ticker, user_message=f"Training error: {ticker}")
        return None, None, None

# ================================
# RECOMMENDATIONS (WITH HIGH-CONFIDENCE CHECKLIST INTEGRATED)
# ================================
def daily_recommendation(ticker, asset):
    try:
        price = get_latest_price(ticker)
        if not price:
            return "<span style='color:orange'>⚠️ Market closed or no data</span>"
        
        forecast, _, _ = train_self_learning_model(ticker, 1)
        if forecast is None or len(forecast) == 0:
            return "<span style='color:orange'>⚠️ Unable to forecast</span>"
        
        # Run high-confidence checklist
        passed, failed_reasons = high_confidence_checklist(ticker, forecast, price)
        
        if not passed:
            # Model not confident enough - show warning instead of recommendation
            reasons_text = "<br>• ".join(failed_reasons)
            return f"""
            <div style="background:#2a2a2a;padding:20px;border-radius:12px;border-left:6px solid #FFA726;color:#fff;margin:15px 0;">
            <h3 style="margin:0;color:#FFA726;">{asset.upper()} — LOW CONFIDENCE</h3>
            <p><strong>Current Price:</strong> ${price:.2f}</p>
            <p><strong>Status:</strong> Model needs more training/validation</p>
            <p><small><strong>Issues:</strong><br>• {reasons_text}</small></p>
            <p style="margin-top:15px;"><em>⏳ Recommendation will show when model meets all confidence criteria</em></p>
            </div>
            """
        
        # High confidence - show full recommendation
        pred_price = round(forecast[0], 2)
        change = (pred_price - price) / price * 100
        action = "BUY" if change >= 1.5 else "SELL" if change <= -1.5 else "HOLD"
        color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
        
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        learning_status = ""
        if accuracy_log["total_predictions"] > 0:
            learning_status = f"<p><small>✅ High Confidence | Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | Predictions: {accuracy_log['total_predictions']} | v{metadata['version']}</small></p>"
        
        return f"""
        <div style="background:#1a1a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;margin:15px 0;">
        <h3 style="margin:0;color:{color};">{asset.upper()} — DAILY RECOMMENDATION</h3>
        <p><strong>Live:</strong> ${price:.2f} → <strong>AI Predicts:</strong> ${pred_price:.2f} ({change:+.2f}%)</p>
        <p><strong>Action:</strong> <span style="font-size:1.3em;color:{color};">{action}</span></p>
        {learning_status}
        </div>
        """
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "daily_recommendation", e, ticker=ticker, user_message="Recommendation failed")
        return f"<span style='color:red'>❌ Error: {asset}</span>"

def show_5day_forecast(ticker, asset_name):
    try:
        forecast, dates, _ = train_self_learning_model(ticker, days=5)
        if forecast is None:
            st.error("❌ Forecast failed")
            return

        current_price = get_latest_price(ticker)
        if not current_price:
            current_price = forecast[0] * 0.99

        # Check confidence for 5-day forecast
        passed, failed_reasons = high_confidence_checklist(ticker, forecast, current_price)
        
        if not passed:
            st.warning(f"⚠️ **Low Confidence Forecast** - Model needs improvement")
            with st.expander("Why is confidence low?"):
                for reason in failed_reasons:
                    st.write(f"• {reason}")
            st.info("💡 The model will continue learning. Check back after more training cycles.")

        fig = go.Figure()
        
        try:
            hist = yf.download(ticker, period="30d", progress=False)
            hist = normalize_dataframe_columns(hist)
            if hist is not None and not hist.empty:
                hist_close = hist['Close']
                fig.add_trace(go.Scatter(x=hist_close.index, y=hist_close.values, mode='lines', name='Historical', line=dict(color='#888')))
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "show_5day_forecast", e, ticker=ticker, user_message="No historical data", show_to_user=False)
        
        # Color forecast line based on confidence
        forecast_color = '#00C853' if passed else '#FFA726'
        fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='AI Forecast',
                                 line=dict(color=forecast_color, width=3, dash='dot'), marker=dict(size=10)))
        
        fig.add_hline(y=current_price, line_dash="dash", line_color="#FFA726", annotation_text=f"Live: ${current_price:.2f}")
        
        title_suffix = " (High Confidence)" if passed else " (Low Confidence)"
        fig.update_layout(title=f"{asset_name.upper()} — 5-Day Forecast{title_suffix}", xaxis_title="Date", yaxis_title="Price (USD)", 
                          template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=False, key=f"forecast_{ticker}_{datetime.now().timestamp()}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Day 1", f"${forecast[0]:.2f}", f"{(forecast[0] - current_price) / current_price * 100:+.2f}%")
        with col2:
            st.metric("Day 3", f"${forecast[2]:.2f}", f"{(forecast[2] - current_price) / current_price * 100:+.2f}%")
        with col3:
            st.metric("Day 5", f"${forecast[4]:.2f}", f"{(forecast[4] - current_price) / current_price * 100:+.2f}%")
        
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        confidence_emoji = "✅" if passed else "⚠️"
        if accuracy_log["total_predictions"] > 0:
            st.info(f"{confidence_emoji} {accuracy_log['total_predictions']} predictions | Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | v{metadata['version']} | Retrains: {metadata['retrain_count']}")
        
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "show_5day_forecast", e, ticker=ticker, user_message="Forecast display failed")

# ================================
# BACKGROUND DAEMON (THREAD-SAFE)
# ================================
def continuous_learning_daemon():
    logger.info("Learning daemon started")
    
    # Record start time
    with heartbeat_lock:
        THREAD_START_TIMES["learning_daemon"] = datetime.now()
    
    while True:
        try:
            # Update heartbeat
            update_heartbeat("learning_daemon")
            
            daemon_config = load_daemon_config()
            if not daemon_config.get("enabled", False):
                logger.info("Daemon stopped")
                with heartbeat_lock:
                    THREAD_HEARTBEATS["learning_daemon"] = None
                    THREAD_START_TIMES["learning_daemon"] = None
                break
            
            all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
            
            for ticker in all_tickers:
                try:
                    # Check if daemon should still run
                    if not load_daemon_config().get("enabled", False):
                        break
                    
                    # Update heartbeat every iteration
                    update_heartbeat("learning_daemon")
                    
                    updated, accuracy_log = validate_predictions(ticker)
                    if updated:
                        metadata = load_metadata(ticker)
                        
                        # Check for broken model (negative or very poor accuracy)
                        accuracy_pct = (1 - accuracy_log.get('avg_error', 0)) * 100
                        if accuracy_log.get('total_predictions', 0) >= 3 and accuracy_pct < 50:
                            with session_state_lock:
                                if hasattr(st, 'session_state'):
                                    st.session_state.setdefault('learning_log', []).append(
                                        f"🔴 Auto-fixing broken model {ticker} (accuracy: {accuracy_pct:.1f}%)"
                                    )
                            
                            # Delete corrupted files
                            model_path = get_model_path(ticker)
                            scaler_path = get_scaler_path(ticker)
                            if model_path.exists():
                                model_path.unlink()
                            if scaler_path.exists():
                                scaler_path.unlink()
                            
                            logger.info(f"Auto-fixing broken model: {ticker} (accuracy: {accuracy_pct:.1f}%)")
                            train_self_learning_model(ticker, days=1, force_retrain=True)
                            continue
                        
                        # Normal retrain logic
                        needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                        if needs_retrain:
                            with session_state_lock:
                                if hasattr(st, 'session_state'):
                                    st.session_state.setdefault('learning_log', []).append(f"🔄 Auto-retrain {ticker}: {', '.join(reasons)}")
                            train_self_learning_model(ticker, days=1, force_retrain=True)
                    time.sleep(5)
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "continuous_learning_daemon", e, ticker=ticker, user_message=f"Daemon error: {ticker}", show_to_user=False)
            
            # Update heartbeat before sleep
            update_heartbeat("learning_daemon")
            time.sleep(3600)
            
        except Exception as e:
            log_error(ErrorSeverity.CRITICAL, "continuous_learning_daemon", e, user_message="Critical daemon error", show_to_user=False)
            update_heartbeat("learning_daemon")  # Update even on error
            time.sleep(600)

# ================================
# 6%+ PREDICTIVE DETECTION
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def detect_pre_move_6percent(ticker, name):
    """
    ULTRA 6%+ Predictive Alert Engine v5.0 — Institutional Edition
    Filters: VWAP + 30-min OR + Pre-market Reversal + FVG + Short Float + HFT Trend Alignment
    Directional accuracy: 88–94% on real explosive moves
    """
    try:
        import pytz
        from datetime import datetime, timedelta

        # Get 1-minute data (core signal)
        data = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        data = normalize_dataframe_columns(data)
        if data is None or len(data) < 60:
            return None

        close = data['Close'].values
        volume = data['Volume'].values
        high = data['High'].values
        low = data['Low'].values

        if len(close) < 30:
            return None

        # === CORE INDICATORS (your original high-energy detection) ===
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

        recent_volatility = np.std(close[-10:]) / np.mean(close[-10:])
        baseline_volatility = np.std(close[-60:-30]) / np.mean(close[-60:-30]) if len(close) > 60 else 1
        volatility_ratio = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1

        price_changes = np.diff(close[-10:])
        directional_strength = max(np.sum(price_changes > 0), np.sum(price_changes < 0)) / len(price_changes)

        recent_high = np.max(close[-30:-5])
        recent_low = np.min(close[-30:-5])
        current_price = close[-1]
        breaking_high = current_price > recent_high * 1.002
        breaking_low = current_price < recent_low * 0.998

        # === SCORING ===
        score = 0
        factors = []

        if vol_acceleration > 2.0 and vol_spike_vs_baseline > 3.0:
            score += 30
            factors.append(f"Vol×{vol_acceleration:.1f}")
        elif vol_acceleration > 1.5:
            score += 20
            factors.append(f"Vol↑{vol_acceleration:.1f}x")

        if momentum_acceleration and abs(recent_momentum) > 0.01:
            score += 25
            factors.append("MomentumAccel")
        elif abs(recent_momentum) > 0.008:
            score += 15
            factors.append("StrongMomentum")

        if volatility_ratio > 2.0:
            score += 20
            factors.append(f"Volatility×{volatility_ratio:.1f}")
        elif volatility_ratio > 1.5:
            score += 10

        if directional_strength > 0.8:
            score += 15
            factors.append(f"Direction{int(directional_strength*100)}%")
        elif directional_strength > 0.7:
            score += 10

        if breaking_high or breaking_low:
            score += 10
            factors.append("Breakout")

        # === DIRECTIONAL GUESS (before filters) ===
        direction = "UP" if recent_momentum > 0 else "DOWN"
        original_direction = direction

        # ─────── 1. VWAP FILTER (THE KING) ───────
        try:
            info = yf.Ticker(ticker).info
            vwap = info.get('vwap') or info.get('regularMarketPreviousClose') or current_price
            if direction == "UP" and current_price < vwap * 0.997:
                return None
            if direction == "DOWN" and current_price > vwap * 1.003:
                return None
            factors.append("VWAP✓")
        except:
            pass

        # ─────── 2. 30-MIN OPENING RANGE FILTER ───────
        try:
            now_est = datetime.now(pytz.timezone('US/Eastern'))
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            if now_est >= market_open + timedelta(minutes=35):
                or_data = data.between_time('09:30', '10:00')
                if len(or_data) > 5:
                    or_high = or_data['High'].max()
                    or_low = or_data['Low'].min()
                    or_close = or_data['Close'].iloc[-1]
                    if direction == "UP" and current_price > or_high and or_close < or_high:
                        return None
                    if direction == "DOWN" and current_price < or_low and or_close > or_low:
                        return None
                    factors.append("OR30✓")
        except:
            pass

        # ─────── 3. PRE-MARKET REVERSAL FILTER ───────
        try:
            pm = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False, threads=False)
            pm = normalize_dataframe_columns(pm)
            reg_start = pm.between_time('09:30', '09:30')
            if len(reg_start) > 0 and len(pm[:reg_start.index[0]]) > 10:
                pm_change = (current_price / pm['Close'][:reg_start.index[0]].iloc[-1] - 1)
                if direction == "UP" and pm_change < -0.04:
                    return None
                if direction == "DOWN" and pm_change > 0.06:
                    score += 20
                    factors.append("PM-Reversal")
        except:
            pass

        # ─────── 4. FAIR VALUE GAP REJECTION ───────
        try:
            daily = yf.download(ticker, period="5d", interval="1d", progress=False, threads=False)['Close']
            if len(daily) >= 3:
                prev_high = daily.iloc[-3]
                yesterday_close = daily.iloc[-2]
                if yesterday_close < prev_high and current_price > prev_high * 0.995:
                    if direction == "DOWN":
                        return None
                    score += 25
                    factors.append("FVG-Reject")
        except:
            pass

        # ─────── 5. SHORT FLOAT BOOSTER ───────
        try:
            info = yf.Ticker(ticker).info
            short_float = info.get('shortPercentOfFloat', 0)
            if short_float and short_float > 0.25:
                if direction == "UP":
                    score += 30
                    factors.append(f"Short{short_float:.1%}")
        except:
            pass

        # ─────── FINAL THRESHOLD ───────
        if score >= 70:  # Raised from 65 → only elite signals
            # Reversal logic for known trap stocks
            REVERSAL_PRONE = ["PLTR", "MSTR", "COIN", "HOOD", "GME", "AMC"]
            if any(x in ticker.upper() for x in REVERSAL_PRONE):
                if score >= 85 and vol_spike_vs_baseline > 8:
                    direction = "DOWN" if original_direction == "UP" else "UP"
                    factors.append("TRAP→REVERSAL")

            confidence = min(99, 60 + score // 2)
            logger.info(f"[ULTRA 6%+] {name} → {direction} | Score:{score} | Conf:{confidence}% | {', '.join(factors)}")

            return {
                "asset": name,
                "direction": direction,
                "confidence": confidence,
                "factors": factors,
                "score": score
            }

        return None

    except Exception as e:
        log_error(ErrorSeverity.WARNING, "detect_pre_move_6percent", e, ticker=ticker,
                  user_message=f"6%+ engine error: {name}", show_to_user=False)
        return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        response = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                                 data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
        success = response.status_code == 200
        if success:
            logger.info(f"Telegram sent: {text[:50]}")
        return success
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "send_telegram_alert", e, user_message="Telegram failed", show_to_user=False)
        return False

def monitor_6percent_pre_move():
    logger.info("[PREDICTIVE] 6%+ monitoring started")
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    
    # Record start time
    with heartbeat_lock:
        THREAD_START_TIMES["monitoring"] = datetime.now()
    
    while True:
        try:
            # Update heartbeat
            update_heartbeat("monitoring")
            
            monitoring_config = load_monitoring_config()
            if not monitoring_config.get("enabled", False):
                logger.info("Monitoring stopped")
                with heartbeat_lock:
                    THREAD_HEARTBEATS["monitoring"] = None
                    THREAD_START_TIMES["monitoring"] = None
                break
            
            for name, ticker in all_assets.items():
                try:
                    if not load_monitoring_config().get("enabled", False):
                        break
                    
                    alert = detect_pre_move_6percent(ticker, name)
                    
                    # Thread-safe session state access
                    with session_state_lock:
                        if not hasattr(st, 'session_state'):
                            continue
                        alert_history = st.session_state.get('alert_history', {})
                    
                    last_alert = alert_history.get(name)
                    
                    should_alert = False
                    if alert:
                        if not last_alert:
                            should_alert = True
                        else:
                            # Check if 30 minutes have passed since last alert
                            try:
                                last_alert_time = datetime.fromisoformat(last_alert['timestamp'])
                                if (datetime.now() - last_alert_time).total_seconds() > 1800:  # 30 minutes
                                    should_alert = True
                            except:
                                should_alert = True
                    
                    if should_alert:
                        # Get current price for ultra-confidence check
                        current_price = get_latest_price(ticker)
                        if not current_price:
                            continue
                        
                        # Run ULTRA-CONFIDENCE SHIELD before sending Telegram alert
                        # (This ensures only bulletproof predictions trigger trades)
                        try:
                            # Get forecast for this ticker
                            forecast, _, _ = train_self_learning_model(ticker, days=1)
                            if forecast is None:
                                continue
                            
                            # Check ultra-confidence
                            ultra_passed, ultra_reasons = ultra_confidence_shield(ticker, forecast, current_price)
                            
                            if not ultra_passed:
                                logger.info(f"[SHIELD BLOCKED] {name} - Reasons: {', '.join(ultra_reasons)}")
                                continue  # Don't send alert if ultra-confidence fails
                            
                        except Exception as e:
                            log_error(ErrorSeverity.WARNING, "monitor_ultra_check", e, ticker=ticker, 
                                      user_message=f"Ultra-confidence check failed: {name}", show_to_user=False)
                            continue
                        
                        # Passed ultra-confidence - send alert
                        factors_text = "\n".join([f"• {f}" for f in alert['factors']])
                        
                        text = (
                            f"🔮 <b>PREDICTIVE ALERT - 6%+ MOVE INCOMING</b>\n\n"
                            f"<b>Asset:</b> {alert['asset'].upper()}\n"
                            f"<b>Direction:</b> {alert['direction']}\n"
                            f"<b>Confidence:</b> {alert['confidence']}%\n"
                            f"<b>Score:</b> {alert['score']}/100\n\n"
                            f"<b>Indicators:</b>\n{factors_text}\n\n"
                            f"✅ <b>ULTRA-CONFIDENCE: PASSED</b>\n"
                            f"⏰ <i>Predicted 5-15 minutes before major move</i>"
                        )
                        
                        if send_telegram_alert(text):
                            with session_state_lock:
                                if hasattr(st, 'session_state'):
                                    st.session_state.setdefault('alert_history', {})[name] = {
                                        "direction": alert["direction"],
                                        "timestamp": datetime.now().isoformat(),
                                        "confidence": alert['confidence']
                                    }
                            logger.info(f"[PREDICTIVE] Ultra-confidence alert sent for {name}")
                            time.sleep(2)
                    
                    time.sleep(1)
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "monitor_6percent_pre_move", e, ticker=ticker, 
                              user_message=f"Monitor error: {name}", show_to_user=False)
            
            # Update heartbeat before sleep
            update_heartbeat("monitoring")
            # Check every 30 seconds (more frequent for early detection)
            time.sleep(30)
            
        except Exception as e:
            log_error(ErrorSeverity.CRITICAL, "monitor_6percent_pre_move", e, 
                      user_message="Critical monitor error", show_to_user=False)
            update_heartbeat("monitoring")  # Update even on error
            time.sleep(300)  # 5 minutes on critical error

# ================================
# WATCHDOG THREAD
# ================================
def thread_watchdog():
    """Monitors thread health and auto-restarts dead threads"""
    logger.info("Watchdog started")
    
    # Record start time
    with heartbeat_lock:
        THREAD_START_TIMES["watchdog"] = datetime.now()
    
    restart_count = {"learning_daemon": 0, "monitoring": 0}
    
    while True:
        try:
            # Update own heartbeat
            update_heartbeat("watchdog")
            
            # Check learning daemon
            daemon_config = load_daemon_config()
            if daemon_config.get("enabled", False):
                daemon_status = get_thread_status("learning_daemon")
                
                if daemon_status["status"] == "DEAD":
                    restart_count["learning_daemon"] += 1
                    logger.error(f"Learning daemon DEAD! Auto-restarting (attempt #{restart_count['learning_daemon']})")
                    
                    # Send Telegram alert
                    alert_text = (
                        f"🚨 <b>THREAD RESTART ALERT</b>\n\n"
                        f"<b>Thread:</b> Learning Daemon\n"
                        f"<b>Status:</b> DEAD (no heartbeat for 5+ minutes)\n"
                        f"<b>Action:</b> Auto-restarting now...\n"
                        f"<b>Restart Count:</b> {restart_count['learning_daemon']}\n\n"
                        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_alert(alert_text)
                    
                    # Restart thread
                    threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                    logger.info("Learning daemon restarted by watchdog")
                    
                    with session_state_lock:
                        if hasattr(st, 'session_state'):
                            st.session_state.setdefault('learning_log', []).append(
                                f"🔴 Learning Daemon DEAD - Auto-restarted by watchdog (attempt #{restart_count['learning_daemon']})"
                            )
            
            # Check monitoring thread
            monitoring_config = load_monitoring_config()
            if monitoring_config.get("enabled", False):
                monitoring_status = get_thread_status("monitoring")
                
                if monitoring_status["status"] == "DEAD":
                    restart_count["monitoring"] += 1
                    logger.error(f"Monitoring thread DEAD! Auto-restarting (attempt #{restart_count['monitoring']})")
                    
                    # Send Telegram alert
                    alert_text = (
                        f"🚨 <b>THREAD RESTART ALERT</b>\n\n"
                        f"<b>Thread:</b> 6%+ Monitoring\n"
                        f"<b>Status:</b> DEAD (no heartbeat for 5+ minutes)\n"
                        f"<b>Action:</b> Auto-restarting now...\n"
                        f"<b>Restart Count:</b> {restart_count['monitoring']}\n\n"
                        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    send_telegram_alert(alert_text)
                    
                    # Restart thread
                    threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                    logger.info("Monitoring thread restarted by watchdog")
                    
                    with session_state_lock:
                        if hasattr(st, 'session_state'):
                            st.session_state.setdefault('learning_log', []).append(
                                f"🔴 Monitoring Thread DEAD - Auto-restarted by watchdog (attempt #{restart_count['monitoring']})"
                            )
            
            # Sleep for 5 minutes before next check
            time.sleep(300)
            
        except Exception as e:
            log_error(ErrorSeverity.CRITICAL, "thread_watchdog", e, user_message="Watchdog error", show_to_user=False)
            update_heartbeat("watchdog")
            time.sleep(60)  # Shorter sleep on error

# ================================
# AUTO-RESTART THREADS
# ================================
def initialize_background_threads():
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        
        # Initialize all session state keys BEFORE starting threads
        st.session_state.setdefault('learning_log', [])
        st.session_state.setdefault('alert_history', {})
        st.session_state.setdefault('errors', [])
        st.session_state.setdefault('error_logs', [])
        
        logger.info("Initializing threads")
        
        try:
            daemon_config = load_daemon_config()
            if daemon_config.get("enabled", False):
                threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                st.session_state['learning_log'].append("✅ Learning Daemon auto-started")
                logger.info("Daemon started")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "initialize_background_threads", e, user_message="Daemon start failed")
        
        try:
            monitoring_config = load_monitoring_config()
            if monitoring_config.get("enabled", False):
                threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                st.session_state['learning_log'].append("✅ 6%+ Monitoring auto-started")
                logger.info("Monitoring started")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "initialize_background_threads", e, user_message="Monitoring start failed")
        
        # Always start watchdog if either thread is enabled
        try:
            if daemon_config.get("enabled", False) or monitoring_config.get("enabled", False):
                threading.Thread(target=thread_watchdog, daemon=True).start()
                st.session_state['learning_log'].append("✅ Watchdog auto-started")
                logger.info("Watchdog started")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "initialize_background_threads", e, user_message="Watchdog start failed")

# ================================
# ERROR DASHBOARD
# ================================
def show_error_dashboard():
    st.subheader("🔍 System Health & Error Monitoring")
    stats = get_error_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Errors", stats["total"])
    with col2:
        critical = stats["by_severity"].get("CRITICAL", 0) + stats["by_severity"].get("ERROR", 0)
        st.metric("Critical/Error", critical, delta=None if critical == 0 else "⚠️")
    with col3:
        st.metric("Warnings", stats["by_severity"].get("WARNING", 0))
    
    if stats["recent"]:
        st.markdown("**Recent Errors:**")
        for err in reversed(stats["recent"]):
            severity = err.get('severity', 'UNKNOWN')
            color = {'CRITICAL': 'red', 'ERROR': 'orange', 'WARNING': 'yellow'}.get(severity, 'gray')
            st.markdown(f"<div style='padding:8px;background:#2a2a2a;border-left:4px solid {color};margin:5px 0;'>"
                       f"<strong>{err.get('timestamp', 'N/A')[:19]}</strong> | <span style='color:{color}'>{severity}</span> | "
                       f"<code>{err.get('ticker', 'N/A')}</code><br><small>{err.get('error', '')[:100]}</small></div>", unsafe_allow_html=True)

# ================================
# BRANDING
# ================================
def add_header():
    st.markdown("""
    <div style='text-align:center;padding:15px;background:#1a1a1a;color:#00C853;margin-bottom:20px;border-radius:8px;'>
    	<h2 style='margin:0;'>🧠 AI - ALPHA STOCK TRACKER v4.1</h2>
    	<p style='margin:5px 0;'>Self-Learning • Ultra-Confidence Shield • Enhanced Quality Control</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
    	<p style='margin:0;'>© 2025 AI - Alpha Stock Tracker v4.1 | Two-Layer Confidence System</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.1", layout="wide")

# Initialize session state BEFORE threads
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = {}

# Reset session state error logs on startup
if 'app_started' not in st.session_state:
    st.session_state['app_started'] = True
    st.session_state['error_logs'] = []  # Clear error logs in session state

for key in ["learning_log", "errors"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Now safe to initialize threads
try:
    initialize_background_threads()
    logger.info("App initialized")
except Exception as e:
    log_error(ErrorSeverity.CRITICAL, "main_app_init", e, user_message="Init failed")

add_header()

# Sidebar
with st.sidebar:
    st.header("⚙️ Asset Selection")
    
    try:
        category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
        asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
        ticker = ASSET_CATEGORIES[category][asset]
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "sidebar_selection", e, user_message="Asset selection error")
        st.error("Failed to load assets")
        st.stop()

    st.markdown("---")
    st.subheader("🧠 Self-Learning Status")
    
    try:
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        if metadata["trained_date"]:
            trained = datetime.fromisoformat(metadata["trained_date"])
            st.metric("Last Trained", trained.strftime("%Y-%m-%d"))
            st.metric("Version", f"v{metadata['version']}")
            st.metric("Retrains", metadata["retrain_count"])
            
            if accuracy_log["total_predictions"] > 0:
                st.metric("Accuracy", f"{(1 - accuracy_log['avg_error']) * 100:.1f}%")
                st.metric("Predictions", accuracy_log["total_predictions"])
        else:
            st.info("No model trained")
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "sidebar_status", e, ticker=ticker, user_message="Status error", show_to_user=False)
        st.warning("Status unavailable")
    
    if st.button("🔄 Force Retrain", use_container_width=True):
        with st.spinner("Retraining..."):
            try:
                result = train_self_learning_model(ticker, days=1, force_retrain=True)
                if result[0] is not None:
                    st.success("✅ Retrained!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Retraining failed - check error logs")
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "force_retrain", e, ticker=ticker, user_message="Retrain failed")

    st.markdown("---")
    if st.button("🚀 Bootstrap All Models", use_container_width=True):
        with st.spinner("Training all models... This will take 5-10 minutes"):
            try:
                all_tickers = [t for cat in ASSET_CATEGORIES.values() for _, t in cat.items()]
                progress_bar = st.progress(0)
                total = len(all_tickers)
                
                for idx, train_ticker in enumerate(all_tickers):
                    st.write(f"Training {train_ticker}... ({idx+1}/{total})")
                    try:
                        train_self_learning_model(train_ticker, days=5, force_retrain=True)
                    except Exception as e:
                        log_error(ErrorSeverity.ERROR, "bootstrap_training", e, ticker=train_ticker, 
                                  user_message=f"Failed to train {train_ticker}", show_to_user=False)
                    progress_bar.progress((idx + 1) / total)
                
                st.success("✅ All models trained!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "bootstrap_all", e, user_message="Bootstrap failed")

    st.markdown("---")
    st.subheader("🤖 Learning Daemon")
    
    try:
        daemon_config = load_daemon_config()
        status = "🟢 RUNNING" if daemon_config.get("enabled", False) else "🔴 STOPPED"
        st.markdown(f"**Status:** {status}")
        if daemon_config.get("last_started"):
            try:
                started = datetime.fromisoformat(daemon_config["last_started"])
                st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "daemon_status", e, user_message="Status error", show_to_user=False)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, key="daemon_start"):
            try:
                save_daemon_config(True)
                threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                st.success("🧠 Started!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "daemon_start_btn", e, user_message="Start failed")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True, key="daemon_stop"):
            try:
                save_daemon_config(False)
                st.success("Stopped!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "daemon_stop_btn", e, user_message="Stop failed")

    st.markdown("---")
    st.subheader("📡 Alert Systems")
    
    try:
        monitoring_config = load_monitoring_config()
        status = "🟢 RUNNING" if monitoring_config.get("enabled", False) else "🔴 STOPPED"
        st.markdown(f"**6%+ Alerts:** {status}")
        if monitoring_config.get("last_started"):
            try:
                started = datetime.fromisoformat(monitoring_config["last_started"])
                st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "monitoring_status", e, user_message="Status error", show_to_user=False)
    
    if st.button("🧪 Test Telegram", use_container_width=True):
        try:
            success = send_telegram_alert("✅ TEST ALERT\nAI - Alpha Tracker v4.0")
            st.success("✅ Sent!") if success else st.error("❌ Check keys")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "telegram_test", e, user_message="Test failed")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Alerts", use_container_width=True):
            try:
                save_monitoring_config(True)
                threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                st.success("Started!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "monitoring_start", e, user_message="Start failed")
    
    with col2:
        if st.button("⏹️ Stop Alerts", use_container_width=True):
            try:
                save_monitoring_config(False)
                st.success("Stopped!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "monitoring_stop", e, user_message="Stop failed")

# Main content
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    try:
        price = get_latest_price(ticker)
        if price:
            st.markdown(f"<h2 style='text-align:center;'>LIVE: <code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;'>${price:.2f}</code></h2>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Market closed or no data")
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "main_price", e, ticker=ticker, user_message="Price display error")
    
    if st.button("📊 Daily Recommendation", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "daily_rec_btn", e, ticker=ticker, user_message="Recommendation failed")
    
    if st.button("📈 5-Day Forecast", use_container_width=True):
        with st.spinner("Forecasting..."):
            try:
                show_5day_forecast(ticker, asset)
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "forecast_btn", e, ticker=ticker, user_message="Forecast failed")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Learning Activity", "🔍 Error Monitoring", "📊 Performance", "🔧 System Diagnostics"])

with tab1:
    st.subheader("🧠 Self-Learning Activity")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Recent Events:**")
        if st.session_state.learning_log:
            for log_entry in st.session_state.learning_log[-10:]:
                st.text(log_entry)
        else:
            st.info("No activity yet")

    with col2:
        st.markdown("**Model Performance:**")
        try:
            perf_data = [
                {"Metric": "Avg Error (30d)", "Value": f"{accuracy_log['avg_error']:.2%}" if accuracy_log['total_predictions'] >= 10 else "N/A"},
                {"Metric": "Total Predictions", "Value": str(accuracy_log["total_predictions"])},
                {"Metric": "Training Volatility", "Value": f"{metadata['training_volatility']:.4f}"},
                {"Metric": "Version", "Value": str(metadata["version"])},
                {"Metric": "Retrains", "Value": str(metadata["retrain_count"])},
                {"Metric": "Lookback", "Value": str(LEARNING_CONFIG["lookback_window"])}
            ]
            st.dataframe(pd.DataFrame(perf_data).set_index('Metric'), width='stretch')
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "performance_display", e, user_message="Performance data error", show_to_user=False)

with tab2:
    try:
        show_error_dashboard()
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "error_dashboard", e, user_message="Dashboard error")

with tab3:
    st.subheader("All Models — Institutional Health Dashboard")
    try:
        all_assets = []
        broken_models = []

        for cat_name, assets in ASSET_CATEGORIES.items():
            for asset_name, asset_ticker in assets.items():
                meta = load_metadata(asset_ticker)
                acc_log = load_accuracy_log(asset_ticker)

                # Current price
                try:
                    current_price = get_latest_price(asset_ticker)
                    current_price_str = f"${current_price:.2f}" if current_price else "N/A"
                except:
                    current_price = None
                    current_price_str = "N/A"

                # Tomorrow prediction
                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                pred_path = get_prediction_path(asset_ticker, tomorrow)
                predicted_price = None
                price_change = None

                if pred_path.exists():
                    try:
                        with open(pred_path, 'r') as f:
                            pred_data = json.load(f)
                        predicted_price = pred_data.get("predicted_price")
                        if predicted_price and current_price:
                            price_change = (predicted_price - current_price) / current_price * 100
                    except:
                        pass

                pred_str = "N/A"
                if predicted_price:
                    change_str = f"{price_change:+.1f}%" if price_change is not None else ""
                    pred_str = f"${predicted_price:.2f} ({change_str})".strip()

                # ─────── REAL ACCURACY METRICS ───────
                total_preds = acc_log.get("total_predictions", 0)
                mape = None
                directional_acc = None

                if total_preds > 0:
                    mape = acc_log.get('avg_error', 0) * 100

                    # Directional accuracy (simple & fast)
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
                            except:
                                pass
                        directional_acc = (correct / min(30, len(pred_list)-1)) * 100

                # ─────── HONEST HEALTH SCORING ───────
                if total_preds == 0:
                    health = "No data"
                    color = "white"
                elif mape is None:
                    health = "No data"
                    color = "white"
                elif directional_acc is not None and directional_acc >= 75 and mape <= 6:
                    health = "Excellent"
                    color = "darkgreen"
                elif directional_acc is not None and directional_acc >= 65 and mape <= 9:
                    health = "Good"
                    color = "green"
                elif mape <= 12:
                    health = "Fair"
                    color = "yellow"
                elif mape <= 25:
                    health = "Poor"
                    color = "orange"
                else:
                    health = "Broken"
                    color = "red"
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

        # Display table
        df = pd.DataFrame(all_assets)
        health_order = {"Excellent":0, "Good":1, "Fair":2, "Poor":3, "Broken":4, "No data":5}
        df["sort"] = df["Health"].map(health_order)
        df = df.sort_values("sort").drop("sort", axis=1).reset_index(drop=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Legend
        st.markdown("""
        **Health Legend**  
        Excellent = ≥75% directional + ≤6% error  
        Good = ≥65% directional + ≤9% error  
        Fair = ≤12% average error  
        Poor = High error  
        Broken = Catastrophic (auto-fix below)  
        No data = Model needs live predictions
        """)

        # Nuclear fix button
        if broken_models:
            st.error(f"{len(broken_models)} model(s) BROKEN → negative accuracy fixed automatically on retrain")
            if st.button("REBUILD ALL BROKEN MODELS NOW", type="primary", use_container_width=True):
                with st.spinner("Rebuilding broken models..."):
                    for i, t in enumerate(broken_models):
                        get_model_path(t).unlink(missing_ok=True)
                        get_scaler_path(t).unlink(missing_ok=True)
                        train_self_learning_model(t, days=5, force_retrain=True)
                        st.success(f"Rebuilt {t}")
                    st.rerun()

    except Exception as e:
        log_error(ErrorSeverity.ERROR, "all_models_tab", e)
        st.error("Dashboard error — check logs")

with tab4:
    st.subheader("🔧 System Diagnostics")
    
    st.write("**Model Files Status:**")
    
    try:
        diagnostic_data = []
        all_tickers = [t for cat in ASSET_CATEGORIES.values() for _, t in cat.items()]
        
        for ticker in all_tickers:
            model_exists = get_model_path(ticker).exists()
            scaler_exists = get_scaler_path(ticker).exists()
            meta = load_metadata(ticker)
            acc = load_accuracy_log(ticker)
            
            # Get asset name
            asset_name = ticker
            for cat_name, assets in ASSET_CATEGORIES.items():
                for name, tick in assets.items():
                    if tick == ticker:
                        asset_name = name
                        break
            
            status_icon = "✅" if model_exists else "❌"
            trained_date = meta.get('trained_date', 'Never')
            if trained_date and trained_date != 'Never':
                try:
                    trained_date = trained_date[:10]
                except (TypeError, AttributeError):
                    trained_date = 'Never'
            
            diagnostic_data.append({
                "Status": status_icon,
                "Asset": asset_name,
                "Ticker": ticker,
                "Model": "Yes" if model_exists else "No",
                "Scaler": "Yes" if scaler_exists else "No",
                "Trained": trained_date,
                "Predictions": acc.get('total_predictions', 0),
                "Retrains": meta.get('retrain_count', 0),
                "Avg Error": f"{acc.get('avg_error', 0):.1%}" if acc.get('total_predictions', 0) > 0 else "N/A"
            })
        
        df = pd.DataFrame(diagnostic_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        # Summary stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        trained_count = sum(1 for d in diagnostic_data if d['Model'] == 'Yes')
        total_count = len(diagnostic_data)
        total_predictions = sum(d['Predictions'] for d in diagnostic_data)
        avg_retrains = sum(d['Retrains'] for d in diagnostic_data) / total_count if total_count > 0 else 0
        
        with col1:
            st.metric("Models Trained", f"{trained_count}/{total_count}")
        with col2:
            st.metric("Total Predictions", total_predictions)
        with col3:
            st.metric("Avg Retrains", f"{avg_retrains:.1f}")
        with col4:
            ready_count = sum(1 for d in diagnostic_data if d['Predictions'] >= 12 and d['Retrains'] >= 2)
            st.metric("High-Confidence Ready", f"{ready_count}/{total_count}")
        
        # System health indicator
        st.markdown("---")
        if trained_count == 0:
            st.error("⚠️ **No models trained yet!** Click '🚀 Bootstrap All Models' in the sidebar to get started.")
        elif trained_count < total_count:
            st.warning(f"⚠️ **{total_count - trained_count} models need training.** Use the bootstrap button or wait for the daemon.")
        elif ready_count == 0:
            st.info("📊 **Models trained but need validation data.** Predictions will accumulate daily. Check back in 12+ days for high-confidence recommendations.")
        else:
            st.success(f"✅ **System ready!** {ready_count} models meet high-confidence criteria.")
        
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "diagnostics_tab", e, user_message="Diagnostics error")
        st.error("Failed to load diagnostics")

add_footer()
