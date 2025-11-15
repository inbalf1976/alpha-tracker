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
warnings.filterwarnings("ignore")

# ================================
# 1. CONFIG & ASSETS
# ================================

BOT_TOKEN = "8447705352:AAFeuK6S01W94tm9dH0z5iatD5fFO8ETKC4"
CHAT_ID = "1500305017"

ASSET_CATEGORIES = {
    "Commodities": {
        "Crude Oil": "CL=F", "Brent Oil": "BZ=F", "Gasoline": "RB=F", "Natural Gas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F", "Platinum": "PL=F", "Palladium": "PA=F",
        "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F", "Sugar": "SB=F", "Coffee": "KC=F",
        "Cocoa": "CC=F", "Cotton": "CT=F", "Live Cattle": "LE=F"
    },
    "Indices": {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ 100": "^NDX", "Russell 2000": "^RUT",
        "Germany 40": "^GDAXI", "UK 100": "^FTSE", "France 40": "^FCHI", "Japan 225": "^N225",
        "Hong Kong 50": "^HSI"
    },
    "Currencies": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X", "NZD/USD": "NZDUSD=X", "USD/CNY": "USDCNY=X",
        "USD/MXN": "USDMXN=X", "USD/ZAR": "USDZAR=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X"
    },
    "Popular": {
        "Tesla": "TSLA", "NVIDIA": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
        "Meta": "META", "Alphabet": "GOOGL", "Netflix": "NFLX", "Coinbase": "COIN", "Palantir": "PLTR",
        "Airbnb": "ABNB", "Salesforce": "CRM", "Intel": "INTC", "PayPal": "PYPL", "Nike": "NKE",
        "Broadcom": "AVGO", "Visa": "V", "Mastercard": "MA", "JPMorgan": "JPM", "BlackRock": "BLK"
    }
}

# ================================
# 2. AUTO-INSTALL PACKAGES
# ================================

import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for pkg in ["yfinance", "plotly", "requests", "tensorflow", "scikit-learn", "pyttsx3"]:
    try:
        __import__(pkg.split("/")[-1].split(".")[0])
    except:
        install(pkg)

# ================================
# 3. PRICE & FORECAST FUNCTIONS
# ================================

def get_latest_price(ticker):
    try:
        price = yf.download(ticker, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        return round(float(price), 4) if ticker.endswith(("=F", "=X")) and "USD" not in ticker else round(float(price), 2)
    except:
        return None

@st.cache_data(ttl=3600)
def train_forecast(ticker, days=5):
    try:
        df = yf.download(ticker, period="2y", progress=False)
        if len(df) < 100: return None, None
        df = df[['Close']].copy()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        X = np.array([scaled[i-60:i] for i in range(60, len(scaled))])
        y = np.array([scaled[i] for i in range(60, len(scaled))])
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60,1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        last = scaled[-60:].reshape(1,60,1)
        preds = []
        for _ in range(days):
            pred = model.predict(last, verbose=0)
            preds.append(pred[0,0])
            last = np.append(last[:,1:,:], pred.reshape(1,1,1), axis=1)
        forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        dates = [(datetime.now().date() + timedelta(days=i+1)) for i in range(days)]
        return forecast, dates
    except:
        return None, None

# ================================
# 4. DAILY RECOMMENDATION (1% FROM CURRENT PRICE)
# ================================

def daily_recommendation(ticker, asset):
    price = get_latest_price(ticker)
    if not price:
        return "<span style='color:red'>No live data</span>"

    forecast, _ = train_forecast(ticker, 1)
    pred_price = round(forecast[0], 2) if forecast else price * 1.01
    change = (pred_price - price) / price

    # 1% THRESHOLD
    action = "BUY" if change >= 0.01 else "SELL" if change <= -0.01 else "HOLD"
    color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"

    return f"""
    <div style="background:#1a1a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;font-family:Arial;margin:15px 0;">
    <h3 style="margin:0;color:{color};">{asset.upper()} — DAILY RECOMMENDATION</h3>
    <p><strong>Live:</strong> <code>${price}</code> → <strong>AI Predicts:</strong> <code>${pred_price}</code> <strong>({change:+.2%})</strong></p>
    <p><strong>Action:</strong> <span style="font-size:1.3em;color:{color};">{action}</span></p>
    </div>
    """

# ================================
# 5. 6%+ VOICE ALERTS (ALL ASSETS)
# ================================

@st.cache_data(ttl=60)
def detect_6_percent_move(ticker, name):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if len(data) < 60: return None
        close = data['Close'].values
        volume = data['Volume'].values
        vol_spike = volume[-1] / np.mean(volume[-30:-1]) if np.mean(volume[-30:-1]) > 0 else 1
        momentum = np.mean(np.diff(close[-6:]) / close[-6:-1])
        if abs(momentum) >= 0.06 and vol_spike >= 2.0:
            direction = "UP" if momentum > 0 else "DOWN"
            return {
                "asset": name,
                "direction": direction,
                "action": "BUY" if direction == "UP" else "SELL",
                "change": abs(momentum) * 100,
                "confidence": min(95, int(60 + vol_spike * 10))
            }
    except:
        pass
    return None

def send_voice_alert(asset, direction, percent, action, confidence):
    text = f"ALERT! {asset.upper()} {direction} {percent:.1f}% in 5 min! {action} NOW! Confidence {confidence}%"
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        audio_file = "alert.mp3"
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        with open(audio_file, 'rb') as audio:
            files = {'voice': audio}
            data = {'chat_id': CHAT_ID, 'caption': f"*{text}*"}
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice", data=data, files=files)
        os.remove(audio_file)
    except:
        send_telegram_message(text)

def send_telegram_message(message):
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
    except:
        pass

# ================================
# 6. 5-DAY FORECAST (1% DAILY THRESHOLD)
# ================================

def show_5day_forecast(ticker, asset):
    forecast, dates = train_forecast(ticker, 5)
    if forecast is None:
        st.error("Not enough data for forecast")
        return

    current_price = get_latest_price(ticker)
    rows = []
    total_change = 0
    for i, (date, pred) in enumerate(zip(dates, forecast)):
        if i == 0:
            change = (pred - current_price) / current_price
        else:
            change = (pred - forecast[i-1]) / forecast[i-1]
        total_change += change
        # 1% THRESHOLD
        action = "BUY" if change >= 0.01 else "SELL" if change <= -0.01 else "HOLD"
        color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
        trend = "BULLISH" if total_change > 0.02 else "BEARISH" if total_change < -0.02 else "NEUTRAL"
        trend_color = "#00C853" if trend == "BULLISH" else "#D50000" if trend == "BEARISH" else "#FFA726"
        rows.append({
            "Date": date.strftime("%d/%m"),
            "AI Price": f"${pred:.2f}",
            "Daily": f"{change:+.2%}",
            "Action": f"<span style='color:{color};font-weight:bold;'>{action}</span>",
            "Trend": f"<span style='color:{trend_color};font-weight:bold;'>{trend}</span>"
        })
    df = pd.DataFrame(rows)
    st.markdown("### 5-Day AI Forecast — 1% Daily Threshold")
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown(f"**Overall 5-Day Move: {total_change:+.2%} → {trend}**")

# ================================
# 7. BACKTEST 6%+ STRATEGY
# ================================

def run_backtest():
    results = []
    progress = st.progress(0)
    status = st.empty()

    category = st.session_state.backtest_category
    asset = st.session_state.backtest_asset
    ticker = ASSET_CATEGORIES[category][asset]
    target_date = st.session_state.backtest_date

    status.text(f"Backtesting {asset} on {target_date}...")
    progress.progress(0.3)

    try:
        data = yf.download(ticker, period="60d", interval="5m", progress=False)
        if data.empty or len(data) < 50:
            st.error("No 5-min data.")
            return
        data = data[data.index.date == target_date]
        if data.empty or len(data) < 20:
            st.warning(f"No data on {target_date}.")
            return

        st.success(f"Found {len(data)} 5-min bars")

        position = entry_price = entry_time = None
        for j in range(12, len(data)):
            window = data.iloc[j-12:j]
            close = window['Close'].values
            volume = window['Volume'].values
            vol_mean = np.mean(volume[-6:-1]) if len(volume) > 6 else 1
            vol_spike = volume[-1] / vol_mean if vol_mean > 0 else 1
            momentum = np.mean(np.diff(close[-5:]) / close[-5:-1])
            current_price = data['Close'].iloc[j]
            current_time = data.index[j]

            if position and (current_time - entry_time).total_seconds() >= 1500:
                pnl = (current_price - entry_price) / entry_price if position == "LONG" else (entry_price - current_price) / entry_price
                results.append({
                    "Asset": asset,
                    "Direction": "UP" if position == "LONG" else "DOWN",
                    "Entry": f"${entry_price:.2f}",
                    "Exit": f"${current_price:.2f}",
                    "PnL": f"{pnl:+.2%}",
                    "Win": pnl > 0
                })
                position = None

            if abs(momentum) >= 0.06 and vol_spike >= 2.0 and not position:
                position = "LONG" if momentum > 0 else "SHORT"
                entry_price = close[-1]
                entry_time = window.index[-1]

        progress.progress(1.0)
        status.empty(); progress.empty()

        if not results:
            st.warning("No 6%+ signals.")
            return

        df = pd.DataFrame(results)
        win_rate = df['Win'].mean() * 100
        st.markdown(f"## {asset} — {target_date} | Win Rate: **{win_rate:.1f}%**")
        st.dataframe(df.style.apply(lambda x: ['background:#d4edda' if v else 'background:#f8d7da' for v in x['Win']], axis=1))

    except Exception as e:
        st.error(f"Error: {e}")

# ================================
# 8. 24/7 MONITORING (6%+ ALERTS)
# ================================

def monitor_6percent_alerts(placeholder):
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    alerted = set()
    while st.session_state.monitoring:
        for name, ticker in all_assets.items():
            if not st.session_state.monitoring: break
            alert = detect_6_percent_move(ticker, name)
            if alert and alert["asset"] not in alerted:
                send_voice_alert(alert["asset"], alert["direction"], alert["change"], alert["action"], alert["confidence"])
                color = "#00C853" if alert["direction"] == "UP" else "#D50000"
                placeholder.markdown(
                    f"<div style='background:{color};color:white;padding:20px;border-radius:12px;text-align:center;font-weight:bold;'>"
                    f"**{alert['asset'].upper()}**<br>**{alert['direction']} {alert['change']:.1f}%**<br>"
                    f"<span style='font-size:1.4em;'>{alert['action']} NOW</span> | {alert['confidence']}%</div>",
                    unsafe_allow_html=True
                )
                alerted.add(alert["asset"])
                time.sleep(300)
        time.sleep(60)

# ================================
# 9. APP LAYOUT
# ================================

st.set_page_config(page_title="Alpha Tracker AI", layout="wide")
st.markdown("<h1 style='text-align:center;color:#00C853;'>ALPHA TRACKER AI</h1>", unsafe_allow_html=True)

if "monitoring" not in st.session_state: st.session_state.monitoring = False
if "thread" not in st.session_state: st.session_state.thread = None

# === SIDEBAR ===
with st.sidebar:
    st.header("Manual Asset")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

# === MAIN PANEL ===
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    st.markdown(
        f"<h2 style='text-align:center;'>LIVE: <code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;'>${price}</code></h2>"
        if price else "<h2 style='text-align:center;color:red;'>NO DATA</h2>",
        unsafe_allow_html=True
    )

    alert_placeholder = st.empty()

    # === DAILY RECOMMENDATION ===
    if st.button("GET DAILY RECOMMENDATION", use_container_width=True):
        with st.spinner("AI Analyzing..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)

    # === 5-DAY FORECAST ===
    if st.button("5-DAY AI FORECAST", use_container_width=True):
        with st.spinner("Training AI..."):
            show_5day_forecast(ticker, asset)

    # === BACKTEST ===
    st.markdown("### Backtest 6%+ Strategy")
    col1, col2 = st.columns(2)
    with col1:
        backtest_category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()), key="backtest_cat")
        backtest_asset = st.selectbox("Asset", list(ASSET_CATEGORIES[backtest_category].keys()), key="backtest_asset")
    with col2:
        backtest_date = st.date_input("Event Date", value=datetime(2024, 10, 23), key="backtest_date")

    if st.button("RUN BACKTEST", use_container_width=True):
        st.session_state.backtest_category = backtest_category
        st.session_state.backtest_asset = backtest_asset
        st.session_state.backtest_date = backtest_date
        with st.spinner("Running..."): run_backtest()

    # === 24/7 6%+ ALERTS ===
    if st.button("START 24/7 6%+ ALERTS", use_container_width=True):
        if not st.session_state.monitoring:
            st.session_state.monitoring = True
            st.session_state.thread = threading.Thread(target=monitor_6percent_alerts, args=(alert_placeholder,), daemon=True)
            st.session_state.thread.start()
            st.success("6%+ ALERTS STARTED")
        else:
            st.info("Already running")