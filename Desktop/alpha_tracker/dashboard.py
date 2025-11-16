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

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ================================
# 1. CONFIG & KEYS (✅ SECURED)
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")
except:
    BOT_TOKEN = None
    CHAT_ID = None
    ALPHA_VANTAGE_KEY = None

# ================================
# 2. ASSETS (✅ UPDATED LIST)
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
# 3. DIRECTORIES & PATHS
# ================================
MODEL_DIR = Path("models")
SCALER_DIR = Path("scalers")
ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata")
PREDICTIONS_DIR = Path("predictions")
CONFIG_DIR = Path("config")

for dir_path in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# ================================
# 4. PERSISTENT CONFIG PATHS
# ================================
DAEMON_CONFIG_PATH = CONFIG_DIR / "daemon_config.json"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring_config.json"

# ================================
# 5. SELF-LEARNING CONFIG
# ================================
LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,
    "min_predictions_for_eval": 10,
    "retrain_interval_days": 30,
    "volatility_change_threshold": 0.5,
    "fine_tune_epochs": 5,
    "full_retrain_epochs": 25,
    "lookback_window": 60,
    "use_technical_indicators": True,  # ✅ NEW: Enable technical analysis
    "feature_count": 5  # ✅ Price + RSI + MACD + Volume + Support/Resistance
}

# ================================
# 6. THREAD-SAFE LOCKS
# ================================
model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()
config_lock = threading.Lock()

# ================================
# 7. PERSISTENT CONFIG MANAGEMENT
# ================================
def load_daemon_config():
    """Load daemon configuration from file."""
    if DAEMON_CONFIG_PATH.exists():
        try:
            with config_lock:
                with open(DAEMON_CONFIG_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
    return {"enabled": False, "last_started": None}

def save_daemon_config(enabled):
    """Save daemon configuration to file."""
    config = {
        "enabled": enabled,
        "last_started": datetime.now().isoformat() if enabled else None
    }
    with config_lock:
        with open(DAEMON_CONFIG_PATH, 'w') as f:
            json.dump(config, f)

def load_monitoring_config():
    """Load monitoring configuration from file."""
    if MONITORING_CONFIG_PATH.exists():
        try:
            with config_lock:
                with open(MONITORING_CONFIG_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
    return {"enabled": False, "last_started": None}

def save_monitoring_config(enabled):
    """Save monitoring configuration to file."""
    config = {
        "enabled": enabled,
        "last_started": datetime.now().isoformat() if enabled else None
    }
    with config_lock:
        with open(MONITORING_CONFIG_PATH, 'w') as f:
            json.dump(config, f)

# ================================
# 8. HELPER FUNCTIONS
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
# 9. PRICE FETCHING (✅ COMPLETELY REDESIGNED)
# ================================
@st.cache_data(ttl=30, show_spinner=False)  # Reduced cache time
def get_latest_price(ticker):
    """Fetch latest price with smart fallback methods optimized for commodities."""
    try:
        # Method 1: Try Ticker.info first (most reliable for commodities)
        try:
            tick = yf.Ticker(ticker)
            info = tick.info
            
            # Try current market price first
            if 'currentPrice' in info and info['currentPrice'] and info['currentPrice'] > 0:
                price = float(info['currentPrice'])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            
            # Try regular market price
            if 'regularMarketPrice' in info and info['regularMarketPrice'] and info['regularMarketPrice'] > 0:
                price = float(info['regularMarketPrice'])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
            
            # Try previous close as last resort from info
            if 'previousClose' in info and info['previousClose'] and info['previousClose'] > 0:
                price = float(info['previousClose'])
                return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
        except:
            pass
        
        # Method 2: Try recent history with 5-day period (more reliable)
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if not hist.empty and len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
        except:
            pass
        
        # Method 3: Try 1-hour interval with 2-day period
        try:
            data = yf.download(ticker, period="2d", interval="1h", progress=False, auto_adjust=True)
            if not data.empty and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                if price > 0:
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
        except:
            pass
        
        # Method 4: Try 15-minute interval
        try:
            data = yf.download(ticker, period="1d", interval="15m", progress=False, auto_adjust=True)
            if not data.empty and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                if price > 0:
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
        except:
            pass
        
        # Method 5: Last resort - try history() method
        try:
            tick = yf.Ticker(ticker)
            hist = tick.history(period="1d")
            if not hist.empty and len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
        except:
            pass
        
        return None
        
    except Exception as e:
        # Log detailed error
        error_msg = f"All methods failed for {ticker}: {str(e)[:100]}"
        st.session_state.setdefault('errors', []).append(error_msg)
        return None

# ================================
# 10. TECHNICAL INDICATORS (✅ OPTION A - CONSERVATIVE)
# ================================
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # Return MACD histogram

def calculate_volume_change(volume):
    """Calculate volume rate of change."""
    return volume.pct_change()

def find_support_resistance(prices, window=20):
    """Find nearest support and resistance levels."""
    # Find local minima (support) and maxima (resistance)
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(prices) - window):
        # Check if it's a local minimum (support)
        if prices.iloc[i] == prices.iloc[i-window:i+window].min():
            support_levels.append(prices.iloc[i])
        # Check if it's a local maximum (resistance)
        if prices.iloc[i] == prices.iloc[i-window:i+window].max():
            resistance_levels.append(prices.iloc[i])
    
    # Get most recent levels
    nearest_support = max(support_levels[-3:]) if support_levels else prices.min()
    nearest_resistance = min(resistance_levels[-3:]) if resistance_levels else prices.max()
    
    return nearest_support, nearest_resistance

def calculate_distance_to_levels(current_price, support, resistance):
    """Calculate normalized distance to support/resistance."""
    distance_to_support = (current_price - support) / current_price
    distance_to_resistance = (resistance - current_price) / current_price
    return distance_to_support, distance_to_resistance

def add_technical_indicators(df):
    """Add technical indicators to dataframe - OPTION A (Conservative)."""
    df = df.copy()
    
    # 1. RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 2. MACD Histogram
    df['MACD'] = calculate_macd(df['Close'])
    
    # 3. Volume Change
    df['Volume_Change'] = calculate_volume_change(df['Volume'])
    
    # 4. Support/Resistance Distance
    support, resistance = find_support_resistance(df['Close'])
    df['Distance_Support'] = (df['Close'] - support) / df['Close']
    df['Distance_Resistance'] = (resistance - df['Close']) / df['Close']
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# ================================
# 11. ACCURACY TRACKING SYSTEM
# ================================
def load_accuracy_log(ticker):
    """Load accuracy history for a ticker."""
    path = get_accuracy_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "predictions": [],
        "errors": [],
        "dates": [],
        "avg_error": 0.0,
        "total_predictions": 0
    }

def save_accuracy_log(ticker, log_data):
    """Save accuracy history."""
    path = get_accuracy_path(ticker)
    with accuracy_lock:
        with open(path, 'w') as f:
            json.dump(log_data, f)

def record_prediction(ticker, predicted_price, prediction_date):
    """Record a prediction for future validation."""
    pred_path = get_prediction_path(ticker, prediction_date)
    pred_data = {
        "ticker": ticker,
        "predicted_price": float(predicted_price),
        "prediction_date": prediction_date,
        "timestamp": datetime.now().isoformat()
    }
    with open(pred_path, 'w') as f:
        json.dump(pred_data, f)

def validate_predictions(ticker):
    """Check past predictions against actual prices and update accuracy log."""
    accuracy_log = load_accuracy_log(ticker)
    updated = False
    
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
                
                pred_path.unlink()
                
        except Exception as e:
            pass
    
    return updated, accuracy_log

# ================================
# 12. MODEL METADATA SYSTEM
# ================================
def load_metadata(ticker):
    """Load model metadata."""
    path = get_metadata_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "trained_date": None,
        "training_samples": 0,
        "training_volatility": 0.0,
        "version": 1,
        "retrain_count": 0,
        "last_accuracy": 0.0
    }

def save_metadata(ticker, metadata):
    """Save model metadata."""
    path = get_metadata_path(ticker)
    with open(path, 'w') as f:
        json.dump(metadata, f)

# ================================
# 12. RETRAINING DECISION ENGINE
# ================================
def should_retrain(ticker, accuracy_log, metadata):
    """Decide if model needs retraining based on multiple factors."""
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
    except:
        pass
    
    return False, reasons

# ================================
# 14. LSTM MODEL BUILDER (✅ UPDATED FOR MULTI-FEATURE)
# ================================
def build_lstm_model(feature_count=5):
    """Build LSTM model - supports multiple features."""
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

# ================================
# 15. SELF-LEARNING TRAINING SYSTEM (✅ ENHANCED WITH INDICATORS)
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    """Fully autonomous self-learning training system with technical indicators."""
    
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    
    updated, accuracy_log = validate_predictions(ticker)
    if updated:
        st.session_state.setdefault('learning_log', []).append(
            f"✅ Validated prediction for {ticker}"
        )
    
    metadata = load_metadata(ticker)
    
    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
    
    if not needs_retrain and not force_retrain:
        training_type = "fine-tune"
    else:
        training_type = "full-retrain"
        if reasons:
            st.session_state.setdefault('learning_log', []).append(
                f"🔄 Retraining {ticker}: {', '.join(reasons)}"
            )
    
    try:
        df = yf.download(ticker, period="1y", progress=False) 
        if len(df) < 100:
            return None, None, None
    except:
        return None, None, None

    # ✅ ADD TECHNICAL INDICATORS
    if LEARNING_CONFIG["use_technical_indicators"]:
        df = add_technical_indicators(df)
        feature_columns = ['Close', 'RSI', 'MACD', 'Volume_Change', 'Distance_Support']
    else:
        feature_columns = ['Close']
    
    df = df[feature_columns].copy()
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Scale features
    if training_type == "full-retrain" or not scaler_path.exists():
        scaler = MinMaxScaler()
        scaler.fit(df)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
    
    scaled = scaler.transform(df)
    
    # Prepare sequences
    X, y = [], []
    lookback = LEARNING_CONFIG["lookback_window"]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 0])  # Predict Close price (first column)
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        return None, None, None
    
    feature_count = len(feature_columns)
    
    with model_cache_lock:
        if training_type == "full-retrain":
            model = build_lstm_model(feature_count=feature_count)
            epochs = LEARNING_CONFIG["full_retrain_epochs"]
            
            model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, 
                      validation_split=0.1,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            
            metadata["retrain_count"] += 1
            st.session_state.setdefault('learning_log', []).append(
                f"🧠 Full retrain #{metadata['retrain_count']} for {ticker} with {feature_count} features"
            )
        else:
            try:
                model = tf.keras.models.load_model(str(model_path))
                epochs = LEARNING_CONFIG["fine_tune_epochs"]
                
                recent_size = int(len(X) * 0.3)
                model.fit(X[-recent_size:], y[-recent_size:], 
                          epochs=epochs, batch_size=32, verbose=0)
                
                st.session_state.setdefault('learning_log', []).append(
                    f"⚡ Fine-tuned {ticker} on recent data"
                )
            except:
                model = build_lstm_model(feature_count=feature_count)
                model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], 
                          batch_size=32, verbose=0, validation_split=0.1)
        
        try:
            model.save(str(model_path))
        except Exception as e:
            st.session_state.setdefault('errors', []).append(f"Model save failed {ticker}")
        
        metadata["trained_date"] = datetime.now().isoformat()
        metadata["training_samples"] = len(X)
        metadata["training_volatility"] = float(df['Close'].pct_change().std())
        metadata["version"] += 1
        metadata["last_accuracy"] = accuracy_log["avg_error"]
        metadata["feature_count"] = feature_count  # ✅ Track feature count
        save_metadata(ticker, metadata)
    
    # Make predictions
    last = scaled[-lookback:].reshape(1, lookback, feature_count)
    preds = []
    for _ in range(days):
        pred = model.predict(last, verbose=0)
        preds.append(pred[0, 0])
        
        # Create next input with predicted price + last known indicators
        next_input = np.zeros((1, 1, feature_count))
        next_input[0, 0, 0] = pred[0, 0]  # Predicted close
        # Copy last known indicator values for other features
        for j in range(1, feature_count):
            next_input[0, 0, j] = last[0, -1, j]
        
        last = np.append(last[:, 1:, :], next_input, axis=1)
    
    # Inverse transform predictions (only the Close price column)
    forecast_scaled = np.zeros((len(preds), feature_count))
    forecast_scaled[:, 0] = preds
    for j in range(1, feature_count):
        forecast_scaled[:, j] = scaled[-1, j]
    
    forecast_full = scaler.inverse_transform(forecast_scaled)
    forecast = forecast_full[:, 0]  # Extract Close prices
    
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    record_prediction(ticker, forecast[0], tomorrow)
    
    dates = []
    i = 1
    while len(dates) < days:
        next_date = datetime.now().date() + timedelta(days=i)
        if next_date.weekday() < 5:
            dates.append(next_date)
        i += 1
    
    tf.keras.backend.clear_session()
    
    return forecast, dates, model

# ================================
# 15. DAILY RECOMMENDATION
# ================================
def daily_recommendation(ticker, asset):
    price = get_latest_price(ticker)
    if not price:  
        return "<span style='color:orange'>Market closed or no data</span>"
    
    forecast, _, _ = train_self_learning_model(ticker, 1)
    if forecast is None or len(forecast) == 0:
        return "<span style='color:orange'>Unable to generate forecast</span>"
    
    pred_price = round(forecast[0], 2)
    change = (pred_price - price) / price * 100
    action = "BUY" if change >= 1.5 else "SELL" if change <= -1.5 else "HOLD"
    color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
    
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    learning_status = ""
    if accuracy_log["total_predictions"] > 0:
        learning_status = f"<p><small>Model Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | Predictions: {accuracy_log['total_predictions']} | Version: {metadata['version']}</small></p>"
    
    return f"""
    <div style="background:#1a1a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;margin:15px 0;">
    <h3 style="margin:0;color:{color};">{asset.upper()} — DAILY RECOMMENDATION</h3>
    <p><strong>Live:</strong> ${price:.2f} → <strong>AI Predicts:</strong> ${pred_price:.2f} ({change:+.2f}%)</p>
    <p><strong>Action:</strong> <span style="font-size:1.3em;color:{color};">{action}</span></p>
    {learning_status}
    </div>
    """

# ================================
# 16. 5-DAY FORECAST
# ================================
def show_5day_forecast(ticker, asset_name):
    forecast, dates, _ = train_self_learning_model(ticker, days=5)
    if forecast is None:
        st.error("Failed to generate forecast.")
        return

    current_price = get_latest_price(ticker)
    if not current_price:
        current_price = forecast[0] * 0.99

    fig = go.Figure()
    
    try:
        hist = yf.download(ticker, period="30d", progress=False)['Close'] 
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values, 
            mode='lines', name='Historical', 
            line=dict(color='#888')
        ))
    except:
        pass
    
    fig.add_trace(go.Scatter(
        x=dates, y=forecast,
        mode='lines+markers',
        name='Self-Learning AI Forecast',
        line=dict(color='#00C853', width=3, dash='dot'),
        marker=dict(size=10)
    ))
    
    fig.add_hline(y=current_price, line_dash="dash", line_color="#FFA726", 
                  annotation_text=f"Live: ${current_price:.2f}")
    
    fig.update_layout(
        title=f"{asset_name.upper()} — 5-Day Self-Learning Forecast",
        xaxis_title="Date", yaxis_title="Price (USD)", 
        template="plotly_dark", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        day1_change = (forecast[0] - current_price) / current_price * 100
        st.metric("Day 1", f"${forecast[0]:.2f}", f"{day1_change:+.2f}%")
    with col2:
        day3_change = (forecast[2] - current_price) / current_price * 100
        st.metric("Day 3", f"${forecast[2]:.2f}", f"{day3_change:+.2f}%")
    with col3:
        day5_change = (forecast[4] - current_price) / current_price * 100
        st.metric("Day 5", f"${forecast[4]:.2f}", f"{day5_change:+.2f}%")
    
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    if accuracy_log["total_predictions"] > 0:
        st.info(f"🧠 Model learns from {accuracy_log['total_predictions']} validated predictions | "
                f"Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | "
                f"Features: {metadata.get('feature_count', 1)} | "
                f"Version: {metadata['version']} | "
                f"Retrains: {metadata['retrain_count']}")

# ================================
# 17. BACKGROUND LEARNING DAEMON
# ================================
def continuous_learning_daemon():
    """Background thread that continuously validates and improves models."""
    while True:
        daemon_config = load_daemon_config()
        if not daemon_config.get("enabled", False):
            break
            
        try:
            all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
            
            for ticker in all_tickers:
                daemon_config = load_daemon_config()
                if not daemon_config.get("enabled", False):
                    break
                
                updated, accuracy_log = validate_predictions(ticker)
                
                if updated:
                    metadata = load_metadata(ticker)
                    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                    
                    if needs_retrain:
                        st.session_state.setdefault('learning_log', []).append(
                            f"🔄 Auto-retraining {ticker}: {', '.join(reasons)}"
                        )
                        train_self_learning_model(ticker, days=1, force_retrain=True)
            
            time.sleep(3600)
            
        except Exception as e:
            st.session_state.setdefault('errors', []).append(f"Learning daemon error: {str(e)[:50]}")
            time.sleep(600)

# ================================
# 18. 6%+ DETECTION
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
            return {
                "asset": name,
                "direction": direction,
                "confidence": confidence
            }
    except:
        pass
    return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def monitor_6percent_pre_move():
    """Background monitoring thread for 6%+ moves."""
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    
    while True:
        monitoring_config = load_monitoring_config()
        if not monitoring_config.get("enabled", False):
            break
            
        for name, ticker in all_assets.items():
            monitoring_config = load_monitoring_config()
            if not monitoring_config.get("enabled", False):
                break
                
            alert = detect_pre_move_6percent(ticker, name)
            
            if alert and alert["asset"] not in st.session_state['alert_history']:
                text = f"🚨 6%+ MOVE INCOMING\n{alert['asset'].upper()} {alert['direction']}\nCONFIDENCE: {alert['confidence']}%"
                
                if send_telegram_alert(text):
                    st.session_state['alert_history'][alert["asset"]] = {
                        "direction": alert["direction"],
                        "timestamp": datetime.now().isoformat()
                    }
                    time.sleep(2)
        
        time.sleep(60)

# ================================
# 19. AUTO-RESTART THREADS ON APP LOAD
# ================================
def initialize_background_threads():
    """Auto-start background threads based on persistent config."""
    
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        
        daemon_config = load_daemon_config()
        if daemon_config.get("enabled", False):
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.session_state.setdefault('learning_log', []).append(
                "✅ Learning Daemon auto-started on app load"
            )
        
        monitoring_config = load_monitoring_config()
        if monitoring_config.get("enabled", False):
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.session_state.setdefault('learning_log', []).append(
                "✅ 6%+ Pre-Move Monitoring auto-started on app load"
            )

# ================================
# 20. BRANDING
# ================================
def add_header():
    st.markdown("""
    <div style='text-align:center;padding:15px;background:#1a1a1a;color:#00C853;margin-bottom:20px;border-radius:8px;'>
    	<h2 style='margin:0;'>🧠 AI - ALPHA STOCK TRACKER v4.0</h2>
    	<p style='margin:5px 0;'>True Self-Learning • Persistent 24/7 • Autonomous</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
    	<p style='margin:0;'>© 2025 AI - Alpha Stock Tracker | Truly Self-Learning AI with Persistent Threads</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# 21. MAIN APP
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.0", layout="wide")

# ================================
# STREAMLIT APP INITIALIZATION
# ================================
# Initialize session state for alert tracking if it doesn't exist
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = {} 

# 🚀 AUTO-START BACKGROUND THREADS ON APP LOAD
initialize_background_threads()

add_header()

# Initialize session state
for key in ["learning_log", "errors"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    st.markdown("---")
    st.subheader("🧠 Self-Learning Status")
    
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    if metadata["trained_date"]:
        trained = datetime.fromisoformat(metadata["trained_date"])
        st.metric("Last Trained", trained.strftime("%Y-%m-%d"))
        st.metric("Model Version", f"v{metadata['version']}")
        st.metric("Retrains", metadata["retrain_count"])
        
        if accuracy_log["total_predictions"] > 0:
            acc_pct = (1 - accuracy_log["avg_error"]) * 100
            st.metric("Accuracy", f"{acc_pct:.1f}%")
            st.metric("Predictions", accuracy_log["total_predictions"])
    else:
        st.info("No model trained yet")
    
    if st.button("🔄 Force Retrain", width='stretch'):
        with st.spinner("Retraining from scratch..."):
            train_self_learning_model(ticker, days=1, force_retrain=True)
        st.success("✅ Retrained!")
        st.rerun()

    st.markdown("---")
    st.subheader("🤖 Learning Daemon")
    
    daemon_config = load_daemon_config()
    daemon_status = "🟢 RUNNING" if daemon_config.get("enabled", False) else "🔴 STOPPED"
    st.markdown(f"**Status:** {daemon_status}")
    
    if daemon_config.get("last_started"):
        try:
            started = datetime.fromisoformat(daemon_config["last_started"])
            st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", width='stretch'):
            save_daemon_config(True)
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.success("🧠 Started!")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", width='stretch'):
            save_daemon_config(False)
            st.success("Stopped!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.subheader("📡 Alert Systems")
    
    monitoring_config = load_monitoring_config()
    monitoring_status = "🟢 RUNNING" if monitoring_config.get("enabled", False) else "🔴 STOPPED"
    st.markdown(f"**6%+ Alerts:** {monitoring_status}")
    
    if monitoring_config.get("last_started"):
        try:
            started = datetime.fromisoformat(monitoring_config["last_started"])
            st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    if st.button("🧪 Test Telegram", width='stretch'):
        success = send_telegram_alert("✅ TEST ALERT\nAI - Alpha Stock Tracker v4.0\nPersistent Threads Active")
        if success:
            st.success("✅ Sent!")
        else:
            st.error("❌ Check keys")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Alerts", width='stretch'):
            save_monitoring_config(True)
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.success("Started!")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Alerts", width='stretch'):
            save_monitoring_config(False)
            st.success("Stopped!")
            time.sleep(0.5)
            st.rerun()

# Main content
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    if price:
        # Display price with timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"<h2 style='text-align:center;'>LIVE: <code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;'>${price:.2f}</code></h2>",
            unsafe_allow_html=True
        )
        st.caption(f"Last updated: {current_time}")
    else:
        st.warning("⚠️ Market closed or no data available")
        # Show errors if any
        if st.session_state.get('errors'):
            with st.expander("🔍 Debug Info"):
                for error in st.session_state['errors'][-5:]:
                    st.text(error)
    
    if st.button("📊 Daily Recommendation", width='stretch'):
        with st.spinner("AI analyzing with self-learning..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)
    
    if st.button("📈 5-Day Self-Learning Forecast", width='stretch'):
        with st.spinner("Self-learning model adapting..."):
            show_5day_forecast(ticker, asset)

st.markdown("---")

# Learning Activity Log
st.subheader("🧠 Self-Learning Activity Log")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Recent Learning Events:**")
    if st.session_state.learning_log:
        for log_entry in st.session_state.learning_log[-10:]:
            st.text(log_entry)
    else:
        st.info("No learning activity yet. Start making predictions!")

with col2:
    st.markdown("**Model Performance:**")
    perf_data = [
        {"Metric": "Average Error (last 30)", "Value": f"{accuracy_log['avg_error']:.2%}" if accuracy_log['total_predictions'] >= LEARNING_CONFIG["min_predictions_for_eval"] else "N/A"},
        {"Metric": "Total Validated Predictions", "Value": str(accuracy_log["total_predictions"])},
        {"Metric": "Training Volatility", "Value": f"{metadata['training_volatility']:.4f}"},
        {"Metric": "Model Version", "Value": str(metadata["version"])},
        {"Metric": "Retrain Count", "Value": str(metadata["retrain_count"])},
        {"Metric": "Feature Count", "Value": str(metadata.get("feature_count", 1))},
        {"Metric": "Technical Indicators", "Value": "✅ Enabled" if LEARNING_CONFIG["use_technical_indicators"] else "❌ Disabled"}
    ]

    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf.set_index('Metric'), width='stretch')
    else:
        st.info("Performance data is not yet available.")

add_footer()
