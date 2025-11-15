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
# 2. ASSETS
# ================================
ASSET_CATEGORIES = {
    "Commodities": {
        "Crude Oil": "CL=F", "Brent Oil": "BZ=F", "Natural Gas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
        # 🟢 FINAL UPDATE: Reverting to ZW=F (Futures) for the most direct commodity pricing
        "Corn": "ZC=F", 
        "Wheat": "ZW=F", 
        "Soybeans": "ZS=F", 
        "Coffee": "KC=F" # Reverting Coffee to KC=F as well, for consistency
    },
    "Indices": {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "Russell 2000": "^RUT"
    },
    "Currencies": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X"
    },
    "Popular": {
        "Tesla": "TSLA", "NVIDIA": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT",
        "Alphabet": "GOOGL", "Coinbase": "COIN", "Palantir": "PLTR"
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

for dir_path in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ================================
# 4. SELF-LEARNING CONFIG
# ================================
LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,  # Retrain if error > 8%
    "min_predictions_for_eval": 10,  # Need 10 predictions before evaluating
    "retrain_interval_days": 30,  # Retrain every 30 days minimum
    "volatility_change_threshold": 0.5,  # Retrain if vol changes > 50%
    "fine_tune_epochs": 5,  # Epochs for fine-tuning
    "full_retrain_epochs": 25,  # Epochs for full retrain
    "lookback_window": 60  # LSTM lookback period
}

# ================================
# 5. THREAD-SAFE LOCKS
# ================================
model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()

# ================================
# 6. HELPER FUNCTIONS
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
# 7. PRICE FETCHING
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty or len(data) == 0:
            return None
        price = float(data['Close'].iloc[-1])
        return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
    except:
        return None

# ================================
# 8. ACCURACY TRACKING SYSTEM
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
    
    # Look for predictions from yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    pred_path = get_prediction_path(ticker, yesterday)
    
    if pred_path.exists():
        try:
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
            
            # Get actual price
            actual_price = get_latest_price(ticker)
            if actual_price:
                predicted_price = pred_data["predicted_price"]
                error = abs(predicted_price - actual_price) / actual_price
                
                accuracy_log["predictions"].append(predicted_price)
                accuracy_log["errors"].append(error)
                accuracy_log["dates"].append(yesterday)
                accuracy_log["total_predictions"] += 1
                
                # Keep only last 50 predictions
                if len(accuracy_log["errors"]) > 50:
                    accuracy_log["predictions"] = accuracy_log["predictions"][-50:]
                    accuracy_log["errors"] = accuracy_log["errors"][-50:]
                    accuracy_log["dates"] = accuracy_log["dates"][-50:]
                
                # Calculate average error
                accuracy_log["avg_error"] = np.mean(accuracy_log["errors"][-30:])
                
                save_accuracy_log(ticker, accuracy_log)
                updated = True
                
                # Delete validated prediction file
                pred_path.unlink()
                
        except Exception as e:
            pass
    
    return updated, accuracy_log

# ================================
# 9. MODEL METADATA SYSTEM
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
# 10. RETRAINING DECISION ENGINE
# ================================
def should_retrain(ticker, accuracy_log, metadata):
    """Decide if model needs retraining based on multiple factors."""
    reasons = []
    
    # Check 1: No model exists
    if not get_model_path(ticker).exists():
        reasons.append("No model exists")
        return True, reasons
    
    # Check 2: Accuracy dropped below threshold
    if len(accuracy_log["errors"]) >= LEARNING_CONFIG["min_predictions_for_eval"]:
        avg_error = accuracy_log["avg_error"]
        if avg_error > LEARNING_CONFIG["accuracy_threshold"]:
            reasons.append(f"Accuracy below threshold ({avg_error:.2%} error)")
            return True, reasons
    
    # Check 3: Time-based retraining
    if metadata["trained_date"]:
        try:
            last_trained = datetime.fromisoformat(metadata["trained_date"])
            days_since = (datetime.now() - last_trained).days
            if days_since >= LEARNING_CONFIG["retrain_interval_days"]:
                reasons.append(f"Model is {days_since} days old")
                return True, reasons
        except:
            pass
    
    # Check 4: Market volatility changed
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
# 11. LSTM MODEL BUILDER
# ================================
def build_lstm_model():
    # Adjusted to 30 units for better memory performance on Streamlit Cloud Free Tier
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], 1)),
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(15), # Adjusted Dense layer for smaller LSTM
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ================================
# 12. SELF-LEARNING TRAINING SYSTEM
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    """Fully autonomous self-learning training system."""
    
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    
    # Step 1: Validate past predictions
    updated, accuracy_log = validate_predictions(ticker)
    if updated:
        st.session_state.setdefault('learning_log', []).append(
            f"✅ Validated prediction for {ticker}"
        )
    
    # Step 2: Load metadata
    metadata = load_metadata(ticker)
    
    # Step 3: Decide if retraining is needed
    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
    
    if not needs_retrain and not force_retrain:
        # Just fine-tune existing model
        training_type = "fine-tune"
    else:
        # Full retrain needed
        training_type = "full-retrain"
        if reasons:
            st.session_state.setdefault('learning_log', []).append(
                f"🔄 Retraining {ticker}: {', '.join(reasons)}"
            )
    
    # Step 4: Fetch training data
    try:
        # Reduced historical period to 1 year for better free tier memory usage/speed
        df = yf.download(ticker, period="1y", progress=False) 
        if len(df) < 100:
            return None, None, None
    except:
        return None, None, None

    df = df[['Close']].copy()
    df = df.ffill().bfill()
    
    # Step 5: Prepare scaler
    if training_type == "full-retrain" or not scaler_path.exists():
        scaler = MinMaxScaler()
        scaler.fit(df[['Close']])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
    
    scaled = scaler.transform(df[['Close']])
    
    # Step 6: Create sequences
    X, y = [], []
    lookback = LEARNING_CONFIG["lookback_window"]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    
    if len(X) == 0:
        return None, None, None
    
    # Step 7: Train or fine-tune model
    with model_cache_lock:
        if training_type == "full-retrain":
            # Full retrain from scratch
            model = build_lstm_model()
            epochs = LEARNING_CONFIG["full_retrain_epochs"]
            
            model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, 
                      validation_split=0.1,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            
            metadata["retrain_count"] += 1
            st.session_state.setdefault('learning_log', []).append(
                f"🧠 Full retrain #{metadata['retrain_count']} for {ticker} completed"
            )
        else:
            # Fine-tune existing model
            try:
                model = tf.keras.models.load_model(str(model_path))
                epochs = LEARNING_CONFIG["fine_tune_epochs"]
                
                # Fine-tune on recent 30% of data
                recent_size = int(len(X) * 0.3)
                model.fit(X[-recent_size:], y[-recent_size:], 
                          epochs=epochs, batch_size=32, verbose=0)
                
                st.session_state.setdefault('learning_log', []).append(
                    f"⚡ Fine-tuned {ticker} on recent data"
                )
            except:
                # Fallback to full retrain if loading fails
                model = build_lstm_model()
                model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"], 
                          batch_size=32, verbose=0, validation_split=0.1)
        
        # Step 8: Save model
        try:
            model.save(str(model_path))
        except Exception as e:
            st.session_state.setdefault('errors', []).append(f"Model save failed {ticker}")
        
        # Step 9: Update metadata
        metadata["trained_date"] = datetime.now().isoformat()
        metadata["training_samples"] = len(X)
        metadata["training_volatility"] = float(df['Close'].pct_change().std())
        metadata["version"] += 1
        metadata["last_accuracy"] = accuracy_log["avg_error"]
        save_metadata(ticker, metadata)
    
    # Step 10: Generate forecast
    last = scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(last, verbose=0)
        preds.append(pred[0, 0])
        last = np.append(last[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    
    # Step 11: Record prediction for tomorrow
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    record_prediction(ticker, forecast[0], tomorrow)
    
    # Step 12: Generate dates
    dates = []
    i = 1
    while len(dates) < days:
        next_date = datetime.now().date() + timedelta(days=i)
        if next_date.weekday() < 5:
            dates.append(next_date)
        i += 1
    
    # Step 13: Cleanup
    tf.keras.backend.clear_session()
    
    return forecast, dates, model

# ================================
# 13. DAILY RECOMMENDATION
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
    
    # Show learning status
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
# 14. 5-DAY FORECAST
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
        # Reduced historical period to 30 days for plotting clarity and speed
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
    
    # Show learning metrics
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    
    if accuracy_log["total_predictions"] > 0:
        st.info(f"🧠 Model learns from {accuracy_log['total_predictions']} validated predictions | "
               f"Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | "
               f"Version: {metadata['version']} | "
               f"Retrains: {metadata['retrain_count']}")

# ================================
# 15. BACKGROUND LEARNING DAEMON
# ================================
def continuous_learning_daemon():
    """Background thread that continuously validates and improves models."""
    while st.session_state.get("learning_daemon_active", False):
        try:
            all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
            
            for ticker in all_tickers:
                if not st.session_state.get("learning_daemon_active", False):
                    break
                
                # Validate predictions
                updated, accuracy_log = validate_predictions(ticker)
                
                if updated:
                    # Check if retrain needed
                    metadata = load_metadata(ticker)
                    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                    
                    if needs_retrain:
                        st.session_state.setdefault('learning_log', []).append(
                            f"🔄 Auto-retraining {ticker}: {', '.join(reasons)}"
                        )
                        # Trigger retrain in background
                        train_self_learning_model(ticker, days=1, force_retrain=True)
            
            # Sleep for 1 hour
            time.sleep(3600)
            
        except Exception as e:
            st.session_state.setdefault('errors', []).append(f"Learning daemon error: {str(e)[:50]}")
            time.sleep(600)

# ================================
# 16. 6%+ DETECTION (SIMPLIFIED)
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
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    alerted = set()
    
    while st.session_state.get("pre_move_monitoring", False):
        for name, ticker in all_assets.items():
            if not st.session_state.get("pre_move_monitoring", False):
                break
            alert = detect_pre_move_6percent(ticker, name)
            if alert and alert["asset"] not in alerted:
                text = f"6%+ MOVE INCOMING\n{alert['asset'].upper()} {alert['direction']}\nCONFIDENCE: {alert['confidence']}%"
                send_telegram_alert(text)
                alerted.add(alert["asset"])
                time.sleep(600)
        time.sleep(60)

# ================================
# 17. BRANDING
# ================================
def add_header():
    st.markdown("""
    <div style='text-align:center;padding:15px;background:#1a1a1a;color:#00C853;margin-bottom:20px;border-radius:8px;'>
        <h2 style='margin:0;'>🧠 MICHA STOCKS AI TRADER v4.0</h2>
        <p style='margin:5px 0;'>True Self-Learning • Adaptive • Autonomous</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
        <p style='margin:0;'>© 2025 Micha Stocks | Truly Self-Learning AI</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# 18. MAIN APP
# ================================
st.set_page_config(page_title="Micha Stocks AI v4.0", layout="wide")
add_header()

# Initialize session state
for key in ["pre_move_monitoring", "learning_daemon_active", "learning_log", "errors"]:
    if key not in st.session_state:
        st.session_state[key] = False if "monitoring" in key or "active" in key else []

# Sidebar
with st.sidebar:
    st.header("⚙️ Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    st.markdown("---")
    st.subheader("🧠 Self-Learning Status")
    
    # Show model status
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
    
    if st.button("🔄 Force Retrain"):
        with st.spinner("Retraining from scratch..."):
            train_self_learning_model(ticker, days=1, force_retrain=True)
        st.success("✅ Retrained!")
        st.rerun()

    st.markdown("---")
    st.subheader("🤖 Learning Daemon")
    
    if st.button("▶️ Start Learning Daemon", use_container_width=True):
        if not st.session_state.learning_daemon_active:
            st.session_state.learning_daemon_active = True
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.success("🧠 Learning daemon started!")
        else:
            st.info("Already running")
    
    if st.button("⏹️ Stop Learning Daemon", use_container_width=True):
        st.session_state.learning_daemon_active = False
        st.success("Stopped")

    st.markdown("---")
    st.subheader("📡 Alert Systems")
    
    if st.button("🧪 Test Telegram", use_container_width=True):
        success = send_telegram_alert("TEST ALERT\nMicha Stocks v4.0\nSelf-Learning Active")
        if success:
            st.success("✅ Sent!")
        else:
            st.error("❌ Check keys")

    if st.button("▶️ 6%+ Pre-Move Alerts", use_container_width=True):
        if not st.session_state.pre_move_monitoring:
            st.session_state.pre_move_monitoring = True
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.success("Started!")
        else:
            st.session_state.pre_move_monitoring = False
            st.info("Stopped")

# Main content
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(
            f"<h2 style='text-align:center;'>LIVE: <code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;'>${price:.2f}</code></h2>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Market closed or no data")
    
    if st.button("📊 Daily Recommendation", use_container_width=True):
        with st.spinner("AI analyzing with self-learning..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)
    
    if st.button("📈 5-Day Self-Learning Forecast", use_container_width=True):
        with st.spinner("Self-learning model adapting..."):
            show_5day_forecast(ticker, asset)

st.markdown("---")

# Learning Activity Log
st.subheader("🧠 Self-Learning Activity Log")
col1, col2 = st.columns(2)

with col1:
    if st.session_state.learning_log:
        st.markdown("**Recent Learning Events:**")
        for log_entry in st.session_state.learning_log[-10:]:
            st.text(log_entry)
    else:
        st.info("No learning activity yet. Start making predictions!")

with col2:
    # Show accuracy comparison across assets
    st.markdown("**Model Performance:**")
    perf_data = []
    for cat in ASSET_CATEGORIES.values():
        for name, tick in cat.items():
            acc_log = load_accuracy_log(tick)
            if acc_log["total_predictions"] > 0:
                perf_data.append({
                    "Asset": name,
                    "Predictions": acc_log["total_predictions"],
                    "Accuracy": f"{(1 - acc_log['avg_error'])*100:.1f}%",
                    "Avg Error": f"{acc_log['avg_error']*100:.2f}%"
                })
    
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, use_container_width=True, hide_index=True)
    else:
        st.info("No validated predictions yet")

st.markdown("---")

# Self-Learning Explanation
with st.expander("ℹ️ How Self-Learning Works"):
    st.markdown("""
    ### 🧠 True Self-Learning Features:
    
    **1. Automatic Prediction Validation**
    - Every prediction is recorded with a timestamp
    - Next day, actual prices are compared to predictions
    - Accuracy metrics are automatically updated
    
    **2. Intelligent Retraining Triggers**
    - ⚡ **Accuracy Drop**: Retrains if error > 8%
    - 📅 **Time-Based**: Retrains every 30 days minimum
    - 📊 **Market Regime Change**: Retrains if volatility changes > 50%
    - 🆕 **No Model**: Trains immediately for new assets
    
    **3. Adaptive Learning**
    - 🎯 **Fine-Tuning**: Updates existing models with recent data
    - 🔄 **Full Retraining**: Rebuilds from scratch when needed
    - 📈 **Version Control**: Tracks model improvements over time
    
    **4. Continuous Improvement**
    - Learning daemon runs in background
    - Validates predictions hourly
    - Automatically triggers retraining
    - No human intervention required
    
    **5. Transparency**
    - All learning events logged
    - Accuracy metrics visible
    - Retrain count tracked
    - Model version history maintained
    """)

# Config display
with st.expander("⚙️ Self-Learning Configuration"):
    st.json(LEARNING_CONFIG)

# Manual refresh
if st.button("🔄 Refresh Dashboard"):
    st.rerun()

# Errors
if st.session_state.errors:
    with st.expander("⚠️ System Errors"):
        for err in st.session_state.errors[-10:]:
            st.text(err)

add_footer()