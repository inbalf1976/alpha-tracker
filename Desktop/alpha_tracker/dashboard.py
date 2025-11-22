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
import socket

# ================================
# SUPPRESS WARNINGS
# ================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ================================
# LOGGING
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

class ErrorSeverity(Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def reset_logs():
    for f in ["app.log", "errors.log"]:
        p = LOG_DIR / f
        if p.exists():
            try: p.unlink()
            except: pass
    if ERROR_LOG_PATH.exists():
        try: ERROR_LOG_PATH.unlink()
        except: pass
    with open(ERROR_LOG_PATH, "w") as f:
        json.dump([], f)

reset_logs()

logger = logging.getLogger("alpha_tracker")
logger.setLevel(logging.DEBUG)
for h in logger.handlers[:]: logger.removeHandler(h)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=10*1024*1024, backupCount=5)
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(funcName)s | %(message)s'))

logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== NEW SESSION STARTED ===")

# ================================
# ACCESS CONTROL
# ================================
def require_local_access(name):
    if os.getenv("STREAMLIT_SHARING_MODE") or "STREAMLIT_CLOUD" in os.environ:
        st.error(f"{name} is local-only")
        return False
    return True

# ================================
# TELEGRAM (defined early!)
# ================================
try:
    BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"] or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"] or os.getenv("TELEGRAM_CHAT_ID")
except:
    BOT_TOKEN = CHAT_ID = None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        return True
    except:
        return False

# ================================
# DIRECTORIES & CONFIG
# ================================
for d in ["models","scalers","accuracy_logs","metadata","predictions","config","logs"]:
    Path(d).mkdir(exist_ok=True)

DAEMON_CFG = Path("config/daemon_config.json")
MONITOR_CFG = Path("config/monitoring_config.json")

def load_cfg(path, default={"enabled": False}):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except: pass
    return default

def save_cfg(path, data):
    try: path.write_text(json.dumps(data, indent=2))
    except: pass

# ================================
# ASSETS
# ================================
ASSET_CATEGORIES = {
    "Tech Stocks": {"Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA", "Microsoft": "MSFT", "Google": "GOOGL"},
    "High Growth": {"Palantir": "PLTR", "MicroStrategy": "MSTR", "Coinbase": "COIN"},
    "Commodities": {"Corn": "ZC=F", "Gold": "GC=F", "Oil": "CL=F"},
    "ETFs": {"S&P 500": "SPY"}
}

# ================================
# PRICE FETCHING
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    time.sleep(0.2)
    try:
        data = yf.download(ticker, period="2d", interval="1m", progress=False)
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except: pass
    try:
        return round(yf.Ticker(ticker).info.get("regularMarketPrice", 0), 2)
    except: pass
    return None

# ================================
# BACKGROUND THREADS
# ================================
def continuous_learning_daemon():
    while load_cfg(DAEMON_CFG).get("enabled"):
        time.sleep(3600)  # placeholder
        # your real daemon logic here

def monitor_6percent_pre_move():
    while load_cfg(MONITOR_CFG).get("enabled"):
        time.sleep(60)
        # your real monitoring logic here

def thread_watchdog():
    while True:
        time.sleep(30)
        # watchdog logic

def initialize_background_threads():
    if st.session_state.get("threads_started"):
        return
    st.session_state.threads_started = True

    if load_cfg(DAEMON_CFG).get("enabled"):
        threading.Thread(target=continuous_learning_daemon, daemon=True).start()
    if load_cfg(MONITOR_CFG).get("enabled"):
        threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
    
    # FIXED LINE:
    if any([load_cfg(DAEMON_CFG).get("enabled"), load_cfg(MONITOR_CFG).get("enabled")]):
        threading.Thread(target=thread_watchdog, daemon=True).start()

# ================================
# MAIN APP
# ================================
st.set_page_config(page_title="Alpha Tracker v4.1", layout="wide")

for k in ["learning_log", "error_logs", "alert_history"]:
    st.session_state.setdefault(k, [])

initialize_background_threads()

st.markdown("""
<div style='text-align:center;padding:20px;background:#00C853;color:black;border-radius:12px;margin-bottom:20px;'>
<h1>AI - ALPHA TRACKER v4.1</h1>
<p>Self-Learning • Ultra-Confidence • Real-time Alerts</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("Asset Selection")
    cat = st.selectbox("Category", ASSET_CATEGORIES.keys())
    name = st.selectbox("Asset", ASSET_CATEGORIES[cat].keys())
    ticker = ASSET_CATEGORIES[cat][name]

    st.markdown("---")
    st.subheader("Learning Daemon")
    dc = load_cfg(DAEMON_CFG)
    st.write(f"Status: **{'RUNNING' if dc.get('enabled') else 'STOPPED'}**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", key="daemon_start"):
            if require_local_access("Daemon"):
                save_cfg(DAEMON_CFG, {"enabled": True, "last_started": datetime.now().isoformat()})
                threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                st.rerun()
    with c2:
        if st.button("Stop", key="daemon_stop"):
            save_cfg(DAEMON_CFG, {"enabled": False})
            st.rerun()

    st.markdown("---")
    st.subheader("6%+ Alerts")
    mc = load_cfg(MONITOR_CFG)
    st.write(f"Status: **{'RUNNING' if mc.get('enabled') else 'STOPPED'}**")

    if st.button("Test Telegram", key="test_telegram", use_container_width=True):
        if require_local_access("Telegram Test"):
            ok = send_telegram_alert("TEST ALERT\nAlpha Tracker v4.1 is alive!")
            if ok: st.success("Sent!")
            else: st.error("Failed – check keys")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Alerts", key="alerts_start"):
            if require_local_access("Alerts"):
                save_cfg(MONITOR_CFG, {"enabled": True, "last_started": datetime.now().isoformat()})
                threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                st.rerun()
    with c2:
        if st.button("Stop Alerts", key="alerts_stop"):
            save_cfg(MONITOR_CFG, {"enabled": False})
            st.rerun()

# ================================
# MAIN AREA
# ================================
price = get_latest_price(ticker)
if price:
    st.markdown(f"<h2 style='text-align:center;color:#00C853;'>LIVE PRICE: ${price}</h2>", unsafe_allow_html=True)
else:
    st.warning("No price data (market closed or ticker issue)")

col1, col2 = st.columns(2)
with col1:
    if st.button("Daily AI Recommendation", use_container_width=True):
        st.info("Feature coming soon – model training in progress")
with col2:
    if st.button("5-Day Forecast", use_container_width=True):
        st.info("Forecast engine loading...")

st.markdown("---")
tab1, tab2 = st.tabs(["Learning Log", "Errors"])

with tab1:
    for entry in st.session_state.learning_log[-20:]:
        st.write(entry)
with tab2:
    try:
        errs = json.loads(ERROR_LOG_PATH.read_text())
        st.json(errs[-20:], expanded=False)
    except:
        st.write("No errors yet")

st.markdown("<p style='text-align:center;color:#666'>© 2025 Alpha Tracker v4.1 – Local Only</p>", unsafe_allow_html=True)
