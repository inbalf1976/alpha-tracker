from fastapi import FastAPI, Query
from typing import List, Dict
import yfinance as yf
import pandas_ta as ta
import pandas as pd

app = FastAPI()

# --- Configurations ---
MODEL_CONFIG = {
    "model_a": {"enabled": True, "weight": 0.6},
    "model_b": {"enabled": True, "weight": 0.4}
}

RULE_SETS = {
    "default": lambda x: x if x["confidence"] > 0.7 else None,
    "strict": lambda x: x if x["confidence"] > 0.9 else None
}

SUPPORTED_ASSETS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]

# --- Data Fetch + Indicators ---
def get_live_data(symbol: str) -> Dict:
    if symbol not in SUPPORTED_ASSETS:
        return {"error": f"Asset '{symbol}' not supported."}

    df = yf.download(symbol, period="7d", interval="1h")
    if df.empty:
        return {"error": f"No data found for {symbol}."}

    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)

    latest = df.iloc[-1]
    return {
        "symbol": symbol,
        "price": round(latest["Close"], 2),
        "volume": int(latest["Volume"]),
        "rsi": round(latest["RSI_14"], 2),
        "macd": round(latest["MACD_12_26_9"], 2),
        "bollinger_upper": round(latest["BBU_20_2.0"], 2),
        "bollinger_lower": round(latest["BBL_20_2.0"], 2)
    }

# --- ML Models ---
def model_a_predict(data: Dict) -> Dict:
    # Dummy logic: bullish if RSI < 70 and MACD > 0
    bullish = data["rsi"] < 70 and data["macd"] > 0
    return {
        "model": "A",
        "prediction": "up" if bullish else "down",
        "confidence": 0.78 if bullish else 0.65
    }

def model_b_predict(data: Dict) -> Dict:
    # Dummy logic: bearish if price > Bollinger upper
    bearish = data["price"] > data["bollinger_upper"]
    return {
        "model": "B",
        "prediction": "down" if bearish else "up",
        "confidence": 0.72 if bearish else 0.60
    }

# --- Model Orchestration ---
def run_models(data: Dict) -> List[Dict]:
    results = []
    if MODEL_CONFIG["model_a"]["enabled"]:
        results.append(model_a_predict(data))
    if MODEL_CONFIG["model_b"]["enabled"]:
        results.append(model_b_predict(data))
    return results

# --- Rule Filtering ---
def apply_rules(predictions: List[Dict], logic_key: str) -> List[Dict]:
    rule = RULE_SETS.get(logic_key, RULE_SETS["default"])
    return [p for p in predictions if rule(p)]

# --- Output Formatting ---
def format_output(filtered: List[Dict], asset_data: Dict) -> Dict:
    if "error" in asset_data:
        return {"status": "error", "message": asset_data["error"]}

    return {
        "status": "success",
        "asset": {
            "symbol": asset_data["symbol"],
            "price": asset_data["price"],
            "rsi": asset_data["rsi"],
            "macd": asset_data["macd"]
        },
        "results": filtered,
        "message": f"{len(filtered)} predictions passed the filter"
    }

# --- API Endpoint ---
@app.get("/forecast")
def forecast(symbol: str = Query(...), user_logic: str = Query("default")):
    asset_data = get_live_data(symbol)
    if "error" in asset_data:
        return format_output([], asset_data)

    model_outputs = run_models(asset_data)
    filtered = apply_rules(model_outputs, user_logic)
    return format_output(filtered, asset_data)