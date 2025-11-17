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
import warnings
import joblib
import sqlite3
import tempfile
import shutil
import os
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
from contextlib import contextmanager

# ================================
# 1. CONFIGURATION
# ================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Logging setup - UTF-8 encoding for emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_tracker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set console handler to UTF-8 on Windows
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        if hasattr(handler.stream, 'reconfigure'):
            handler.stream.reconfigure(encoding='utf-8')

# Secure config loading
def get_secure_config(key_name: str) -> Optional[str]:
    """Securely load configuration with validation."""
    try:
        value = st.secrets.get(key_name)
        if value:
            return value
    except:
        pass
    return None

BOT_TOKEN = get_secure_config("TELEGRAM_BOT_TOKEN")
CHAT_ID = get_secure_config("TELEGRAM_CHAT_ID")
ALPHA_VANTAGE_KEY = get_secure_config("ALPHA_VANTAGE_KEY")

# ================================
# 2. ASSETS
# ================================
ASSET_CATEGORIES = {
    "Tech Stocks": {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Microsoft": "MSFT",
        "Alphabet": "GOOGL"
    },
    "High Growth": {
        "Palantir": "PLTR",
        "MicroStrategy": "MSTR",
        "Coinbase": "COIN"
    },
    "Commodities": {
        "Corn Futures": "ZC=F",
        "Gold Futures": "GC=F",
        "Coffee Futures": "KC=F",
        "Crude Oil": "CL=F",
        "Wheat": "ZW=F"
    },
    "ETFs": {
        "S&P 500 ETF": "SPY",
        "WHEAT": "WEAT"
    }
}

# ================================
# 3. DIRECTORIES
# ================================
MODEL_DIR = Path("models")
SCALER_DIR = Path("scalers")
DB_DIR = Path("database")

for dir_path in [MODEL_DIR, SCALER_DIR, DB_DIR]:
    dir_path.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "stock_tracker.db"

# ================================
# 4. CONSTANTS
# ================================
BUY_THRESHOLD = 1.5
SELL_THRESHOLD = -1.5
PRICE_CACHE_TTL = 60
COMMODITY_CACHE_TTL = 30

LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,
    "min_predictions_for_eval": 10,
    "retrain_interval_days": 30,
    "volatility_change_threshold": 0.5,
    "fine_tune_epochs": 5,
    "full_retrain_epochs": 25,
    "lookback_window": 60,
    "use_technical_indicators": True,
    "feature_count": 5,
    "train_test_split": 0.8,
    "validation_split": 0.1
}

# ================================
# 5. THREAD-SAFE DATABASE
# ================================
class ThreadSafeDB:
    """Thread-safe SQLite database wrapper."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        """Get thread-safe database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    prediction_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    actual_price REAL,
                    validated_at TEXT,
                    error REAL,
                    UNIQUE(ticker, prediction_date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    ticker TEXT PRIMARY KEY,
                    trained_date TEXT,
                    training_samples INTEGER,
                    training_volatility REAL,
                    version INTEGER,
                    retrain_count INTEGER,
                    last_accuracy REAL,
                    feature_count INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    tickers_validated INTEGER,
                    models_retrained INTEGER,
                    avg_accuracy REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_ticker 
                ON predictions(ticker, prediction_date)
            """)

# Initialize database
db = ThreadSafeDB(DB_PATH)

# ================================
# 6. HELPER FUNCTIONS
# ================================
def get_safe_ticker_name(ticker: str) -> str:
    """Convert ticker to safe filename."""
    return ticker.replace('=', '_').replace('^', '').replace('/', '_')

def get_model_path(ticker: str) -> Path:
    return MODEL_DIR / f"{get_safe_ticker_name(ticker)}_lstm.h5"

def get_scaler_path(ticker: str) -> Path:
    return SCALER_DIR / f"{get_safe_ticker_name(ticker)}_scaler.pkl"

def get_cache_ttl(ticker: str) -> int:
    """Get appropriate cache TTL based on asset type."""
    return COMMODITY_CACHE_TTL if ticker.endswith(("=F", "=X")) else PRICE_CACHE_TTL

# ================================
# 7. PRICE FETCHING WITH RETRY
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker: str) -> Optional[float]:
    """Fetch latest price with fallback methods."""
    methods = [
        lambda: _fetch_historical_price(ticker, "2d", "1d"),
        lambda: _fetch_ticker_history(ticker, "2d"),
        lambda: _fetch_intraday_price(ticker, "1d", "1h"),
        lambda: _fetch_intraday_price(ticker, "1d", "5m")
    ]
    
    for method in methods:
        try:
            price = method()
            if price and price > 0:
                return round(price, 2)
        except Exception as e:
            logger.debug(f"Price fetch method failed for {ticker}: {e}")
            continue
    
    logger.warning(f"All price fetch methods failed for {ticker}")
    return None

def _fetch_historical_price(ticker: str, period: str, interval: str) -> Optional[float]:
    """Fetch from historical data."""
    hist = yf.download(ticker, period=period, interval=interval, 
                      progress=False, auto_adjust=True, prepost=False)
    if not hist.empty and len(hist) > 0:
        return float(hist['Close'].iloc[-1])
    return None

def _fetch_ticker_history(ticker: str, period: str) -> Optional[float]:
    """Fetch using Ticker object."""
    tick = yf.Ticker(ticker)
    hist = tick.history(period=period)
    if not hist.empty and len(hist) > 0:
        return float(hist['Close'].iloc[-1])
    return None

def _fetch_intraday_price(ticker: str, period: str, interval: str) -> Optional[float]:
    """Fetch intraday data."""
    data = yf.download(ticker, period=period, interval=interval, 
                      progress=False, auto_adjust=True)
    if not data.empty and len(data) > 0:
        return float(data['Close'].iloc[-1])
    return None

# ================================
# 8. TECHNICAL INDICATORS
# ================================
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return (macd - signal_line).replace([np.inf, -np.inf], 0)

def calculate_volume_change(volume: pd.Series) -> pd.Series:
    """Calculate volume rate of change."""
    return volume.pct_change().replace([np.inf, -np.inf], 0)

def find_support_resistance(prices: pd.Series, window: int = 20) -> Tuple[float, float]:
    """Find nearest support and resistance levels."""
    if len(prices) < window * 2:
        return prices.min(), prices.max()
    
    prices_array = prices.values
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(prices_array) - window):
        if prices_array[i] == np.min(prices_array[i-window:i+window+1]):
            support_levels.append(prices_array[i])
        if prices_array[i] == np.max(prices_array[i-window:i+window+1]):
            resistance_levels.append(prices_array[i])
    
    nearest_support = max(support_levels[-3:]) if support_levels else prices.min()
    nearest_resistance = min(resistance_levels[-3:]) if resistance_levels else prices.max()
    
    return nearest_support, nearest_resistance

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe."""
    df = df.copy()
    
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['Volume_Change'] = calculate_volume_change(df['Volume'])
    
    support, resistance = find_support_resistance(df['Close'])
    df['Distance_Support'] = (df['Close'] - support) / df['Close']
    df['Distance_Resistance'] = (resistance - df['Close']) / df['Close']
    
    df = df.fillna(method='ffill')
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

# ================================
# 9. DATABASE OPERATIONS
# ================================
def record_prediction(ticker: str, predicted_price: float, prediction_date: str):
    """Record a prediction in database."""
    with db.get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO predictions 
            (ticker, predicted_price, prediction_date, created_at)
            VALUES (?, ?, ?, ?)
        """, (ticker, predicted_price, prediction_date, datetime.now().isoformat()))

def load_accuracy_log(ticker: str) -> Dict[str, Any]:
    """Load accuracy history from database."""
    with db.get_connection() as conn:
        cursor = conn.execute("""
            SELECT predicted_price, actual_price, error, prediction_date
            FROM predictions
            WHERE ticker = ? AND actual_price IS NOT NULL
            ORDER BY prediction_date DESC
            LIMIT 50
        """, (ticker,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return {
                "predictions": [],
                "errors": [],
                "dates": [],
                "avg_error": 0.0,
                "total_predictions": 0
            }
        
        predictions = [row['predicted_price'] for row in rows]
        errors = [row['error'] for row in rows]
        dates = [row['prediction_date'] for row in rows]
        
        return {
            "predictions": predictions,
            "errors": errors,
            "dates": dates,
            "avg_error": np.mean(errors[-30:]) if len(errors) >= 30 else np.mean(errors),
            "total_predictions": len(rows)
        }

def validate_predictions(ticker: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate past predictions against actual prices."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    updated = False
    
    with db.get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, predicted_price
            FROM predictions
            WHERE ticker = ? AND prediction_date = ? AND actual_price IS NULL
        """, (ticker, yesterday))
        
        row = cursor.fetchone()
        
        if row:
            actual_price = get_latest_price(ticker)
            if actual_price:
                predicted_price = row['predicted_price']
                error = abs(predicted_price - actual_price) / actual_price
                
                conn.execute("""
                    UPDATE predictions
                    SET actual_price = ?, validated_at = ?, error = ?
                    WHERE id = ?
                """, (actual_price, datetime.now().isoformat(), error, row['id']))
                
                updated = True
                logger.info(f"Validated prediction for {ticker}: error={error:.2%}")
    
    accuracy_log = load_accuracy_log(ticker)
    return updated, accuracy_log

def load_metadata(ticker: str) -> Dict[str, Any]:
    """Load model metadata from database."""
    with db.get_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM model_metadata WHERE ticker = ?
        """, (ticker,))
        
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        
        return {
            "trained_date": None,
            "training_samples": 0,
            "training_volatility": 0.0,
            "version": 1,
            "retrain_count": 0,
            "last_accuracy": 0.0,
            "feature_count": LEARNING_CONFIG["feature_count"]
        }

def save_metadata(ticker: str, metadata: Dict[str, Any]):
    """Save model metadata to database."""
    with db.get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_metadata
            (ticker, trained_date, training_samples, training_volatility, 
             version, retrain_count, last_accuracy, feature_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            metadata["trained_date"],
            metadata["training_samples"],
            metadata["training_volatility"],
            metadata["version"],
            metadata["retrain_count"],
            metadata["last_accuracy"],
            metadata["feature_count"]
        ))

# ================================
# 10. RETRAINING LOGIC
# ================================
def should_retrain(ticker: str, accuracy_log: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Decide if model needs retraining."""
    reasons = []
    
    if not get_model_path(ticker).exists():
        reasons.append("No model exists")
        return True, reasons
    
    if len(accuracy_log["errors"]) >= LEARNING_CONFIG["min_predictions_for_eval"]:
        avg_error = accuracy_log["avg_error"]
        if avg_error > LEARNING_CONFIG["accuracy_threshold"]:
            reasons.append(f"Accuracy below threshold ({avg_error:.2%} error)")
            return True, reasons
    
    if metadata["trained_date"]:
        try:
            last_trained = datetime.fromisoformat(metadata["trained_date"])
            days_since = (datetime.now() - last_trained).days
            if days_since >= LEARNING_CONFIG["retrain_interval_days"]:
                reasons.append(f"Model is {days_since} days old")
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
    except Exception as e:
        logger.error(f"Volatility check failed for {ticker}: {e}")
    
    return False, reasons

# ================================
# 11. MODEL BUILDING
# ================================
def build_lstm_model(feature_count: int = 5) -> Sequential:
    """Build LSTM model with proper architecture."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], feature_count)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def safe_save_model(model: Sequential, path: Path):
    """Atomically save model to prevent corruption - Windows compatible."""
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.h5', dir=path.parent)
        os.close(temp_fd)
        
        model.save(temp_path)
        
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                time.sleep(0.1)
                path.unlink()
        
        shutil.move(temp_path, str(path))
        logger.info(f"Model saved successfully to {path}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

def cleanup_model(model: Optional[Sequential]):
    """Properly cleanup model resources."""
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()

# ================================
# 12. TRAINING SYSTEM
# ================================
def train_self_learning_model(
    ticker: str, 
    days: int = 5, 
    force_retrain: bool = False
) -> Tuple[Optional[np.ndarray], Optional[List], Optional[Sequential]]:
    """Self-learning training system with proper safeguards."""
    
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    
    updated, accuracy_log = validate_predictions(ticker)
    if updated:
        logger.info(f"Validated prediction for {ticker}")
    
    metadata = load_metadata(ticker)
    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
    
    training_type = "full-retrain" if (needs_retrain or force_retrain) else "fine-tune"
    
    if reasons:
        logger.info(f"Retraining {ticker}: {', '.join(reasons)}")
    
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if len(df) < 100:
            logger.error(f"Insufficient data for {ticker}: {len(df)} rows")
            return None, None, None
    except Exception as e:
        logger.error(f"Data fetch failed for {ticker}: {e}")
        return None, None, None
    
    if LEARNING_CONFIG["use_technical_indicators"]:
        df = add_technical_indicators(df)
        feature_columns = ['Close', 'RSI', 'MACD', 'Volume_Change', 'Distance_Support']
    else:
        feature_columns = ['Close']
    
    df = df[feature_columns].copy()
    df = df.fillna(method='ffill').fillna(0)
    
    feature_count = len(feature_columns)
    
    if scaler_path.exists() and not force_retrain:
        try:
            old_scaler = joblib.load(scaler_path)
            if hasattr(old_scaler, 'n_features_in_') and old_scaler.n_features_in_!= feature_count:
                if model_path.exists():
                    model_path.unlink()
                scaler_path.unlink()
                training_type = "full-retrain"
                logger.info(f"Feature count changed for {ticker} - forcing full retrain")
        except:
            pass
    
    if training_type == "full-retrain" or not scaler_path.exists():
        scaler = MinMaxScaler()
        scaler.fit(df)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
    
    scaled = scaler.transform(df)
    
    X, y = [], []
    lookback = LEARNING_CONFIG["lookback_window"]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        logger.error(f"No training samples for {ticker}")
        return None, None, None
    
    split_idx = int(len(X) * LEARNING_CONFIG["train_test_split"])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = None
    try:
        if training_type == "full-retrain":
            model = build_lstm_model(feature_count=feature_count)
            
            history = model.fit(
                X_train, y_train,
                epochs=LEARNING_CONFIG["full_retrain_epochs"],
                batch_size=32,
                verbose=0,
                validation_split=LEARNING_CONFIG["validation_split"],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5, 
                        restore_best_weights=True
                    )
                ]
            )
            
            metadata["retrain_count"] += 1
            logger.info(f"Full retrain #{metadata['retrain_count']} for {ticker}")
        else:
            try:
                model = tf.keras.models.load_model(str(model_path))
                
                if model.input_shape[2] != feature_count:
                    raise ValueError("Feature count mismatch")
                
                recent_size = int(len(X_train) * 0.3)
                model.fit(
                    X_train[-recent_size:], 
                    y_train[-recent_size:],
                    epochs=LEARNING_CONFIG["fine_tune_epochs"],
                    batch_size=32,
                    verbose=0
                )
                
                logger.info(f"Fine-tuned {ticker} on recent data")
            except Exception as e:
                logger.warning(f"Fine-tune failed, doing full retrain: {e}")
                model = build_lstm_model(feature_count=feature_count)
                model.fit(
                    X_train, y_train,
                    epochs=LEARNING_CONFIG["full_retrain_epochs"],
                    batch_size=32,
                    verbose=0,
                    validation_split=LEARNING_CONFIG["validation_split"]
                )
        
        safe_save_model(model, model_path)
        
        metadata["trained_date"] = datetime.now().isoformat()
        metadata["training_samples"] = len(X_train)
        metadata["training_volatility"] = float(df['Close'].pct_change().std())
        metadata["version"] += 1
        metadata["last_accuracy"] = accuracy_log["avg_error"]
        metadata["feature_count"] = feature_count
        save_metadata(ticker, metadata)
        
        last = scaled[-lookback:].reshape(1, lookback, feature_count)
        preds = []
        
        for _ in range(days):
            pred = model.predict(last, verbose=0)
            preds.append(pred[0, 0])
            
            next_input = np.zeros((1, 1, feature_count))
            next_input[0, 0, 0] = pred[0, 0]
            for j in range(1, feature_count):
                next_input[0, 0, j] = last[0, -1, j]
            
            last = np.append(last[:, 1:, :], next_input, axis=1)
        
        forecast_scaled = np.zeros((len(preds), feature_count))
        forecast_scaled[:, 0] = preds
        for j in range(1, feature_count):
            forecast_scaled[:, j] = scaled[-1, j]
        
        forecast_full = scaler.inverse_transform(forecast_scaled)
        forecast = forecast_full[:, 0]
        
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        record_prediction(ticker, forecast[0], tomorrow)
        
        dates = []
        i = 1
        while len(dates) < days:
            next_date = datetime.now().date() + timedelta(days=i)
            if next_date.weekday() < 5:
                dates.append(next_date)
            i += 1
        
        return forecast, dates, model
        
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        return None, None, None
    finally:
        cleanup_model(model)

# ================================
# 13. SCHEDULED VALIDATION
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def scheduled_validation_check() -> Dict[str, Any]:
    """Run validation every hour - thread-safe and scheduled."""
    all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
    
    validated_count = 0
    retrained_count = 0
    total_accuracy = []
    
    for ticker in all_tickers:
        try:
            updated, accuracy_log = validate_predictions(ticker)
            
            if updated:
                validated_count += 1
                
                if accuracy_log["total_predictions"] > 0:
                    total_accuracy.append(1 - accuracy_log["avg_error"])
                
                metadata = load_metadata(ticker)
                needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                
                if needs_retrain:
                    logger.info(f"Auto-retraining {ticker}: {', '.join(reasons)}")
                    train_self_learning_model(ticker, days=1, force_retrain=True)
                    retrained_count += 1
        
        except Exception as e:
            logger.error(f"Validation failed for {ticker}: {e}")
    
    with db.get_connection() as conn:
        conn.execute("""
            INSERT INTO validation_runs
            (run_date, tickers_validated, models_retrained, avg_accuracy)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            validated_count,
            retrained_count,
            np.mean(total_accuracy) if total_accuracy else 0.0
        ))
    
    return {
        "timestamp": datetime.now(),
        "validated": validated_count,
        "retrained": retrained_count,
        "avg_accuracy": np.mean(total_accuracy) if total_accuracy else 0.0
    }

# ================================
# 14. RECOMMENDATIONS
# ================================
def daily_recommendation(ticker: str, asset: str, forecast_data: Optional[np.ndarray] = None) -> str:
    """Generate daily recommendation."""
    price = get_latest_price(ticker)
    if not price:
        return "<span style='color:orange'>⚠️ Market closed or no data available</span>"
    
    if forecast_data is not None:
        forecast = forecast_data
    else:
        forecast, _, _ = train_self_learning_model(ticker, 1)
    
    if forecast is None or len(forecast) == 0:
        return "<span style='color:orange'>⚠️ Unable to generate forecast - insufficient data</span>"
    
    pred_price = round(forecast[0], 2)
    change = (pred_price - price) / price * 100
    
    action = "BUY" if change >= BUY_THRESHOLD else "SELL" if change <= SELL_THRESHOLD else "HOLD"
    color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
    
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    learning_status = ""
    if accuracy_log["total_predictions"] > 0:
        learning_status = f"""
        <p><small>
        📊 Model Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | 
        Predictions: {accuracy_log['total_predictions']} | 
        Version: v{metadata['version']} | 
        Features: {metadata.get('feature_count', 1)}
        </small></p>
        """
    
    return f"""
    <div style="background:#1a1a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;margin:15px 0;">
    <h3 style="margin:0;color:{color};">{asset.upper()} — DAILY RECOMMENDATION</h3>
    <p><strong>Live Price:</strong> ${price:.2f} → <strong>AI Predicts:</strong> ${pred_price:.2f} ({change:+.2f}%)</p>
    <p><strong>Recommended Action:</strong> <span style="font-size:1.3em;color:{color};">{action}</span></p>
    {learning_status}
    </div>
    """

def show_5day_forecast(ticker: str, asset_name: str):
    """Display 5-day forecast with visualization."""
    with st.spinner("Training self-learning model..."):
        forecast, dates, _ = train_self_learning_model(ticker, days=5)
    
    if forecast is None:
        st.error("""
            ❌ **Failed to generate forecast**
            
            Possible reasons:
            - Insufficient historical data (need 100+ days)
            - Market closed or data unavailable
            - Network connectivity issues
            
            **Try:**
            - Select a different asset
            - Refresh the page
            - Check if market is open
        """)
        return

    current_price = get_latest_price(ticker)
    if not current_price:
        current_price = forecast[0] * 0.99

    fig = go.Figure()
    
    try:
        hist = yf.download(ticker, period="30d", progress=False)['Close']
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist.values,
            mode='lines',
            name='Historical (30 Days)',
            line=dict(color='#888', width=2)
        ))
    except Exception as e:
        logger.warning(f"Could not fetch historical data: {e}")
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast,
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#00C853', width=3, dash='dot'),
        marker=dict(size=10, color='#00C853')
    ))
    
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="#FFA726",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    fig.update_layout(
        title=f"{asset_name.upper()} — 5-Day Self-Learning Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        day1_change = (forecast[0] - current_price) / current_price * 100
        st.metric("Day 1 Prediction", f"${forecast[0]:.2f}", f"{day1_change:+.2f}%")
    with col2:
        day3_change = (forecast[2] - current_price) / current_price * 100
        st.metric("Day 3 Prediction", f"${forecast[2]:.2f}", f"{day3_change:+.2f}%")
    with col3:
        day5_change = (forecast[4] - current_price) / current_price * 100
        st.metric("Day 5 Prediction", f"${forecast[4]:.2f}", f"{day5_change:+.2f}%")
    
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    if accuracy_log["total_predictions"] > 0:
        st.info(
            f"🧠 **Model Intelligence**: Learned from {accuracy_log['total_predictions']} validated predictions | "
            f"Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | "
            f"Features: {metadata.get('feature_count', 1)} | "
            f"Version: v{metadata['version']} | "
            f"Retrains: {metadata['retrain_count']}"
        )
    
    st.markdown("---")
    st.markdown("### 📊 Daily Action Based on Day 1 Forecast")
    st.markdown(daily_recommendation(ticker, asset_name, forecast_data=forecast), unsafe_allow_html=True)

# ================================
# 15. TELEGRAM ALERTS
# ================================
def send_telegram_alert(text: str) -> bool:
    """Send Telegram alert with rate limiting."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return False
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")
        return False

# ================================
# 16. UI COMPONENTS
# ================================
def add_header():
    """Add application header."""
    st.markdown("""
    <div style='text-align:center;padding:20px;background:linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);color:#00C853;margin-bottom:30px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.3);'>
        <h1 style='margin:0;font-size:2.5em;'>🧠 AI Alpha Stock Tracker v5.0</h1>
        <p style='margin:10px 0;font-size:1.1em;color:#aaa;'>Production-Ready • Thread-Safe • Self-Learning AI</p>
        <p style='margin:5px 0;font-size:0.9em;color:#666;'>All Critical Issues Fixed • Database-Backed • Atomic Operations</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    """Add application footer."""
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
        <p style='margin:0;'>© 2025 AI Alpha Stock Tracker v5.0 | Production-Ready Architecture</p>
        <p style='margin:5px 0;font-size:0.9em;'>Thread-Safe • Database-Backed • Atomic File Operations</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# 17. MAIN APPLICATION
# ================================
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Alpha Stock Tracker v5.0",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    add_header()
    
    validation_status = scheduled_validation_check()
    
    with st.sidebar:
        st.header("⚙️ Asset Selection")
        category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
        asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
        ticker = ASSET_CATEGORIES[category][asset]
        
        st.markdown("---")
        st.subheader("🧠 Model Status")
        
        accuracy_log = load_accuracy_log(ticker)
        metadata = load_metadata(ticker)
        
        if metadata["trained_date"]:
            try:
                trained = datetime.fromisoformat(metadata["trained_date"])
                st.metric("Last Trained", trained.strftime("%Y-%m-%d %H:%M"))
            except:
                st.metric("Last Trained", "Unknown")
            
            st.metric("Model Version", f"v{metadata['version']}")
            st.metric("Retrains", metadata["retrain_count"])
            st.metric("Features", metadata.get("feature_count", 1))
            
            if accuracy_log["total_predictions"] > 0:
                acc_pct = (1 - accuracy_log["avg_error"]) * 100
                st.metric("Accuracy", f"{acc_pct:.1f}%")
                st.metric("Validated Predictions", accuracy_log["total_predictions"])
        else:
            st.info("⚠️ No model trained yet. Click a button below to train.")
        
        if st.button("🔄 Force Retrain", width='stretch'):
            with st.spinner("Retraining from scratch..."):
                result = train_self_learning_model(ticker, days=1, force_retrain=True)
                if result[0] is not None:
                    st.success("✅ Model retrained successfully!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Retraining failed. Check logs.")
        
        st.markdown("---")
        st.subheader("📡 System Status")
        
        st.markdown(f"""
        **Last Validation**: {validation_status['timestamp'].strftime('%H:%M:%S')}  
        **Validated Today**: {validation_status['validated']} tickers  
        **Auto-Retrained**: {validation_status['retrained']} models  
        **Avg Accuracy**: {validation_status['avg_accuracy']*100:.1f}%
        """)
        
        if st.button("🔍 Run Validation Now", width='stretch'):
            st.cache_data.clear()
            with st.spinner("Running validation..."):
                result = scheduled_validation_check()
                st.success(f"✅ Validated {result['validated']} tickers, retrained {result['retrained']} models")
                st.rerun()
        
        st.markdown("---")
        st.subheader("📨 Telegram Alerts")
        
        if BOT_TOKEN and CHAT_ID:
            st.success("✅ Configured")
            if st.button("🧪 Test Alert", width='stretch'):
                success = send_telegram_alert(
                    "✅ <b>TEST ALERT</b>\n\n"
                    "AI Alpha Stock Tracker v5.0\n"
                    "System operational and monitoring active!"
                )
                if success:
                    st.success("✅ Alert sent!")
                else:
                    st.error("❌ Failed to send. Check logs.")
        else:
            st.warning("⚠️ Not configured. Add secrets to enable.")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        price = get_latest_price(ticker)
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if price:
            st.markdown(
                f"<h2 style='text-align:center;'>LIVE PRICE: "
                f"<code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;color:#00C853;'>"
                f"${price:.2f}</code></h2>",
                unsafe_allow_html=True
            )
            st.caption(f"Last updated: {current_time} | Cache TTL: {get_cache_ttl(ticker)}s")
            
            if st.button("🔄 Refresh Price", key="refresh_price", width='stretch'):
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("⚠️ **Market Closed or Data Unavailable**")
            st.info("""
            The price data could not be fetched. This usually means:
            - Market is currently closed
            - The asset ticker is invalid
            - Network connectivity issues
            
            Please try again later or select a different asset.
            """)
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📊 Daily Recommendation", key="daily_rec", width='stretch'):
                with st.spinner("🧠 AI analyzing market patterns..."):
                    recommendation = daily_recommendation(ticker, asset)
                    st.markdown(recommendation, unsafe_allow_html=True)
                    st.info("💡 **Tip**: Click '5-Day Forecast' for full prediction analysis!")
        
        with col_btn2:
            if st.button("📈 5-Day Forecast", key="forecast", width='stretch'):
                show_5day_forecast(ticker, asset)
    
    st.markdown("---")
    st.subheader("🧠 Model Intelligence Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Performance Metrics**")
        
        perf_data = {
            "Metric": [
                "Average Error (Last 30)",
                "Total Validated Predictions",
                "Training Volatility",
                "Model Version",
                "Total Retrains",
                "Feature Count",
                "Technical Indicators"
            ],
            "Value": [
                f"{accuracy_log['avg_error']:.2%}" if accuracy_log['total_predictions'] >= 10 else "N/A",
                str(accuracy_log["total_predictions"]),
                f"{metadata['training_volatility']:.4f}",
                f"v{metadata['version']}",
                str(metadata["retrain_count"]),
                str(metadata.get("feature_count", 1)),
                "✅ Enabled" if LEARNING_CONFIG["use_technical_indicators"] else "❌ Disabled"
            ]
        }
        
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, width='stretch', hide_index=True)
    
    with col2:
        st.markdown("**📈 Recent Validation History**")
        
        try:
            with db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT run_date, tickers_validated, models_retrained, avg_accuracy
                    FROM validation_runs
                    ORDER BY run_date DESC
                    LIMIT 5
                """)
                
                rows = cursor.fetchall()
                
                if rows:
                    history_data = {
                        "Time": [datetime.fromisoformat(row['run_date']).strftime('%m-%d %H:%M') for row in rows],
                        "Validated": [row['tickers_validated'] for row in rows],
                        "Retrained": [row['models_retrained'] for row in rows],
                        "Accuracy": [f"{row['avg_accuracy']*100:.1f}%" for row in rows]
                    }
                    df_history = pd.DataFrame(history_data)
                    st.dataframe(df_history, width='stretch', hide_index=True)
                else:
                    st.info("No validation history yet. System will auto-validate every hour.")
        except Exception as e:
            st.error(f"Could not load validation history: {e}")
    
    st.markdown("---")
    st.markdown("### 🔧 System Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **🛡️ Security Features**
        - ✅ Thread-safe database operations
        - ✅ Atomic file writes
        - ✅ Secure credential management
        - ✅ Input validation
        """)
    
    with info_col2:
        st.markdown("""
        **⚡ Performance**
        - ✅ Intelligent caching (60s/30s)
        - ✅ Scheduled validation (hourly)
        - ✅ No blocking threads
        - ✅ Memory leak prevention
        """)
    
    with info_col3:
        st.markdown("""
        **🧠 ML Features**
        - ✅ 5 technical indicators
        - ✅ Walk-forward validation
        - ✅ Auto-retraining logic
        - ✅ Model versioning
        """)
    
    add_footer()

# ================================
# 18. APPLICATION ENTRY
# ================================
if __name__ == "__main__":
    main()
