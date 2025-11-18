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

class ErrorSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def setup_logging():
    """Configure logging system"""
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # Error file handler
    error_handler = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

def log_error(severity, function_name, error, ticker=None, user_message="An error occurred", show_to_user=True):
    """Centralized error logging"""
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
    
    # Save to tracking file
    try:
        history = json.load(open(ERROR_LOG_PATH, 'r')) if ERROR_LOG_PATH.exists() else []
        history.append(error_data)
        history = history[-500:]  # Keep last 500
        json.dump(history, open(ERROR_LOG_PATH, 'w'), indent=2)
    except:
        pass
    
    # Show to user
    if show_to_user:
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 {user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"❌ {user_message}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"⚠️ {user_message}")
    
    # Add to session state
    st.session_state.setdefault('error_logs', []).append(error_data)

def get_error_statistics():
    """Get error statistics"""
    try:
        if not ERROR_LOG_PATH.exists():
            return {"total": 0, "by_severity": {}, "recent": []}
        
        errors = json.load(open(ERROR_LOG_PATH, 'r'))
        by_severity = {}
        for error in errors:
            sev = error.get('severity', 'UNKNOWN')
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {"total": len(errors), "by_severity": by_severity, "recent": errors[-10:]}
    except:
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
    "Commodities": {"Corn Futures": "ZC=F", "Gold Futures": "GC=F", "Coffee Futures": "KC=F", "Crude Oil": "CL=F", "Wheat": "ZW=F"},
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

# ================================
# PERSISTENT CONFIG
# ================================
def load_daemon_config():
    try:
        if DAEMON_CONFIG_PATH.exists():
            with config_lock:
                return json.load(open(DAEMON_CONFIG_PATH, 'r'))
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "load_daemon_config", e, user_message="Config load failed", show_to_user=False)
    return {"enabled": False, "last_started": None}

def save_daemon_config(enabled):
    try:
        config = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with config_lock:
            json.dump(config, open(DAEMON_CONFIG_PATH, 'w'), indent=2)
        logger.info(f"Daemon config saved: {enabled}")
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_daemon_config", e, user_message="Failed to save config")
        return False

def load_monitoring_config():
    try:
        if MONITORING_CONFIG_PATH.exists():
            with config_lock:
                return json.load(open(MONITORING_CONFIG_PATH, 'r'))
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "load_monitoring_config", e, user_message="Config load failed", show_to_user=False)
    return {"enabled": False, "last_started": None}

def save_monitoring_config(enabled):
    try:
        config = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with config_lock:
            json.dump(config, open(MONITORING_CONFIG_PATH, 'w'), indent=2)
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
# PRICE FETCHING
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    logger.debug(f"Fetching price for {ticker}")
    methods_tried = []
    
    try:
        # Method 1: 1-minute interval
        try:
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not data.empty and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                logger.info(f"Price: {ticker} ${price:.2f}")
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            methods_tried.append("1m-no-data")
        except Exception as e:
            methods_tried.append(f"1m-{type(e).__name__}")
        
        # Method 2: 5-minute
        try:
            data = yf.download(ticker, period="1d", interval="5m", progress=False)
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            methods_tried.append("5m-no-data")
        except Exception as e:
            methods_tried.append(f"5m-{type(e).__name__}")
        
        # Method 3: Daily
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not data.empty:
                price = float(data['Close'].iloc[-1])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            methods_tried.append("1d-no-data")
        except Exception as e:
            methods_tried.append(f"1d-{type(e).__name__}")
        
        # Method 4: Ticker info
        try:
            tick = yf.Ticker(ticker)
            info = tick.info
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                price = float(info['regularMarketPrice'])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            elif 'previousClose' in info and info['previousClose']:
                price = float(info['previousClose'])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
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
# ACCURACY TRACKING
# ================================
def load_accuracy_log(ticker):
    try:
        path = get_accuracy_path(ticker)
        if path.exists():
            return json.load(open(path, 'r'))
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "load_accuracy_log", e, ticker=ticker, user_message="Accuracy log error", show_to_user=False)
    return {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}

def save_accuracy_log(ticker, log_data):
    try:
        with accuracy_lock:
            json.dump(log_data, open(get_accuracy_path(ticker), 'w'), indent=2)
        return True
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "save_accuracy_log", e, ticker=ticker, user_message="Save failed")
        return False

def record_prediction(ticker, predicted_price, prediction_date):
    try:
        pred_data = {"ticker": ticker, "predicted_price": float(predicted_price), 
                     "prediction_date": prediction_date, "timestamp": datetime.now().isoformat()}
        json.dump(pred_data, open(get_prediction_path(ticker, prediction_date), 'w'), indent=2)
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
                pred_data = json.load(open(pred_path, 'r'))
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
# METADATA
# ================================
def load_metadata(ticker):
    try:
        path = get_metadata_path(ticker)
        if path.exists():
            return json.load(open(path, 'r'))
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "load_metadata", e, ticker=ticker, user_message="Metadata error", show_to_user=False)
    return {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0, "last_accuracy": 0.0}

def save_metadata(ticker, metadata):
    try:
        json.dump(metadata, open(get_metadata_path(ticker), 'w'), indent=2)
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
            if len(df) > 5:
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
            st.session_state.setdefault('learning_log', []).append(f"✅ Validated {ticker}")
        
        metadata = load_metadata(ticker)
        needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
        
        training_type = "full-retrain" if (needs_retrain or force_retrain) else "fine-tune"
        if reasons:
            st.session_state.setdefault('learning_log', []).append(f"🔄 Retraining {ticker}: {', '.join(reasons)}")
        
        # Download data
        df = None
        for attempt in range(3):
            try:
                df = yf.download(ticker, period="1y", progress=False)
                if len(df) >= 100:
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
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Close']].copy().ffill().bfill()
        
        # Check for NaN values properly - convert to numpy to avoid Series issues
        if df['Close'].isna().values.any():
            log_error(ErrorSeverity.WARNING, "train_self_learning_model", Exception("NaN values found"), 
                      ticker=ticker, user_message=f"Data quality issue: {ticker}", show_to_user=False)
            return None, None, None
        
        # Scaler
        try:
            if training_type == "full-retrain" or not scaler_path.exists():
                scaler = MinMaxScaler()
                scaler.fit(df[['Close']])
                joblib.dump(scaler, scaler_path)
            else:
                scaler = joblib.load(scaler_path)
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
                    st.session_state.setdefault('learning_log', []).append(f"🧠 Full retrain #{metadata['retrain_count']} {ticker}")
                else:
                    try:
                        model = tf.keras.models.load_model(str(model_path))
                        epochs = LEARNING_CONFIG["fine_tune_epochs"]
                        recent_size = int(len(X) * 0.3)
                        model.fit(X[-recent_size:], y[-recent_size:], epochs=epochs, batch_size=32, verbose=0)
                        st.session_state.setdefault('learning_log', []).append(f"⚡ Fine-tuned {ticker}")
                    except:
                        model = build_lstm_model()
                        if model is None:
                            return None, None, None
                        model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], batch_size=32, verbose=0, validation_split=0.1)
                
                try:
                    model.save(str(model_path))
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "train_self_learning_model", e, ticker=ticker, user_message="Model save failed")
                
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
# RECOMMENDATIONS
# ================================
def daily_recommendation(ticker, asset):
    try:
        price = get_latest_price(ticker)
        if not price:
            return "<span style='color:orange'>⚠️ Market closed or no data</span>"
        
        forecast, _, _ = train_self_learning_model(ticker, 1)
        if forecast is None or len(forecast) == 0:
            return "<span style='color:orange'>⚠️ Unable to forecast</span>"
        
        pred_price = round(forecast[0], 2)
        change = (pred_price - price) / price * 100
        action = "BUY" if change >= 1.5 else "SELL" if change <= -1.5 else "HOLD"
        color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
        
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        learning_status = ""
        if accuracy_log["total_predictions"] > 0:
            learning_status = f"<p><small>Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | Predictions: {accuracy_log['total_predictions']} | v{metadata['version']}</small></p>"
        
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

        fig = go.Figure()
        
        try:
            hist = yf.download(ticker, period="30d", progress=False)['Close']
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='Historical', line=dict(color='#888')))
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "show_5day_forecast", e, ticker=ticker, user_message="No historical data", show_to_user=False)
        
        fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='AI Forecast',
                                 line=dict(color='#00C853', width=3, dash='dot'), marker=dict(size=10)))
        
        fig.add_hline(y=current_price, line_dash="dash", line_color="#FFA726", annotation_text=f"Live: ${current_price:.2f}")
        fig.update_layout(title=f"{asset_name.upper()} — 5-Day Forecast", xaxis_title="Date", yaxis_title="Price (USD)", 
                          template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Day 1", f"${forecast[0]:.2f}", f"{(forecast[0] - current_price) / current_price * 100:+.2f}%")
        with col2:
            st.metric("Day 3", f"${forecast[2]:.2f}", f"{(forecast[2] - current_price) / current_price * 100:+.2f}%")
        with col3:
            st.metric("Day 5", f"${forecast[4]:.2f}", f"{(forecast[4] - current_price) / current_price * 100:+.2f}%")
        
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        if accuracy_log["total_predictions"] > 0:
            st.info(f"🧠 {accuracy_log['total_predictions']} predictions | Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | v{metadata['version']} | Retrains: {metadata['retrain_count']}")
        
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "show_5day_forecast", e, ticker=ticker, user_message="Forecast display failed")

# ================================
# BACKGROUND DAEMON
# ================================
def continuous_learning_daemon():
    logger.info("Learning daemon started")
    while True:
        try:
            daemon_config = load_daemon_config()
            if not daemon_config.get("enabled", False):
                logger.info("Daemon stopped")
                break
            
            all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
            
            for ticker in all_tickers:
                try:
                    if not load_daemon_config().get("enabled", False):
                        break
                    
                    updated, accuracy_log = validate_predictions(ticker)
                    if updated:
                        metadata = load_metadata(ticker)
                        needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                        if needs_retrain:
                            st.session_state.setdefault('learning_log', []).append(f"🔄 Auto-retrain {ticker}: {', '.join(reasons)}")
                            train_self_learning_model(ticker, days=1, force_retrain=True)
                    time.sleep(5)
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "continuous_learning_daemon", e, ticker=ticker, user_message=f"Daemon error: {ticker}", show_to_user=False)
            
            time.sleep(3600)
        except Exception as e:
            log_error(ErrorSeverity.CRITICAL, "continuous_learning_daemon", e, user_message="Critical daemon error", show_to_user=False)
            time.sleep(600)

# ================================
# 6%+ DETECTION
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def detect_pre_move_6percent(ticker, name):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if len(data) < 60:
            return None

        close = data['Close'].values
        volume = data['Volume'].values
        recent = close[-15:]
        momentum = (recent[-1] - recent[0]) / recent[0]
        vol_spike = volume[-1] / np.mean(volume[-15:-1]) if np.mean(volume[-15:-1]) > 0 else 1
        
        if (abs(momentum) > 0.015 and vol_spike > 2.5):
            direction = "UP" if momentum > 0 else "DOWN"
            confidence = min(98, int(60 + vol_spike * 8))
            logger.info(f"6%+ alert: {name} {direction} ({confidence}%)")
            return {"asset": name, "direction": direction, "confidence": confidence}
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "detect_pre_move_6percent", e, ticker=ticker, user_message=f"Detection failed: {name}", show_to_user=False)
    return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        response = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                                 data={"chat_id": CHAT_ID, "text": text}, timeout=5)
        success = response.status_code == 200
        if success:
            logger.info(f"Telegram sent: {text[:50]}")
        return success
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "send_telegram_alert", e, user_message="Telegram failed", show_to_user=False)
        return False

def monitor_6percent_pre_move():
    logger.info("6%+ monitoring started")
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    
    while True:
        try:
            monitoring_config = load_monitoring_config()
            if not monitoring_config.get("enabled", False):
                logger.info("Monitoring stopped")
                break
            
            for name, ticker in all_assets.items():
                try:
                    if not load_monitoring_config().get("enabled", False):
                        break
                    
                    alert = detect_pre_move_6percent(ticker, name)
                    if alert and alert["asset"] not in st.session_state.get('alert_history', {}):
                        text = f"🚨 6%+ MOVE\n{alert['asset'].upper()} {alert['direction']}\nCONFIDENCE: {alert['confidence']}%"
                        if send_telegram_alert(text):
                            st.session_state.setdefault('alert_history', {})[alert["asset"]] = {
                                "direction": alert["direction"], "timestamp": datetime.now().isoformat()
                            }
                            time.sleep(2)
                    time.sleep(1)
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "monitor_6percent_pre_move", e, ticker=ticker, user_message=f"Monitor error: {name}", show_to_user=False)
            
            time.sleep(60)
        except Exception as e:
            log_error(ErrorSeverity.CRITICAL, "monitor_6percent_pre_move", e, user_message="Critical monitor error", show_to_user=False)
            time.sleep(600)

# ================================
# AUTO-RESTART THREADS
# ================================
def initialize_background_threads():
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        logger.info("Initializing threads")
        
        try:
            daemon_config = load_daemon_config()
            if daemon_config.get("enabled", False):
                threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                st.session_state.setdefault('learning_log', []).append("✅ Learning Daemon auto-started")
                logger.info("Daemon started")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "initialize_background_threads", e, user_message="Daemon start failed")
        
        try:
            monitoring_config = load_monitoring_config()
            if monitoring_config.get("enabled", False):
                threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                st.session_state.setdefault('learning_log', []).append("✅ 6%+ Monitoring auto-started")
                logger.info("Monitoring started")
        except Exception as e:
            log_error(ErrorSeverity.ERROR, "initialize_background_threads", e, user_message="Monitoring start failed")

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
    	<h2 style='margin:0;'>🧠 AI - ALPHA STOCK TRACKER v4.0</h2>
    	<p style='margin:5px 0;'>Self-Learning • Persistent 24/7 • Enhanced Logging</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
    	<p style='margin:0;'>© 2025 AI - Alpha Stock Tracker | Enhanced Error Handling</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.0", layout="wide")

if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = {}

for key in ["learning_log", "errors", "error_logs"]:
    if key not in st.session_state:
        st.session_state[key] = []

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
                train_self_learning_model(ticker, days=1, force_retrain=True)
                st.success("✅ Retrained!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                log_error(ErrorSeverity.ERROR, "force_retrain", e, ticker=ticker, user_message="Retrain failed")

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
tab1, tab2, tab3 = st.tabs(["📈 Learning Activity", "🔍 Error Monitoring", "📊 Performance"])

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
            st.dataframe(pd.DataFrame(perf_data).set_index('Metric'), width="stretch")
        except Exception as e:
            log_error(ErrorSeverity.WARNING, "performance_display", e, user_message="Performance data error", show_to_user=False)

with tab2:
    try:
        show_error_dashboard()
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "error_dashboard", e, user_message="Dashboard error")

with tab3:
    st.subheader("📊 All Models")
    try:
        all_assets = []
        for cat_name, assets in ASSET_CATEGORIES.items():
            for asset_name, asset_ticker in assets.items():
                meta = load_metadata(asset_ticker)
                acc_log = load_accuracy_log(asset_ticker)
                if meta["trained_date"]:
                    all_assets.append({
                        "Asset": asset_name,
                        "Ticker": asset_ticker,
                        "Version": meta["version"],
                        "Retrains": meta["retrain_count"],
                        "Accuracy": f"{(1 - acc_log['avg_error'])*100:.1f}%" if acc_log['total_predictions'] > 0 else "N/A",
                        "Predictions": acc_log["total_predictions"],
                        "Last Trained": meta["trained_date"][:10]
                    })
        
        if all_assets:
            st.dataframe(pd.DataFrame(all_assets), width="stretch", hide_index=True)
        else:
            st.info("No models trained")
    except Exception as e:
        log_error(ErrorSeverity.ERROR, "all_models_tab", e, user_message="Models data error")

add_footer()
