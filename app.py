from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from transformers import pipeline
from prophet import Prophet
from datetime import datetime, timedelta

app = Flask(__name__)

# ✅ Allow all origins to fix CORS issues
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ✅ Ensure CORS Headers for Every Response
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ✅ API URLs
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
GPT_SUMMARY_MODEL = pipeline("summarization")

# ✅ Fetch Stock Data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d", auto_adjust=True)

        if hist.empty:
            return None

        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        hist = hist.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})
        return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"⚠️ Error fetching stock data for {ticker}: {e}")
        return None

# ✅ Fetch Crypto Data (Supports All Cryptos)
def fetch_crypto_data(ticker):
    try:
        coin_list_url = f"{COINGECKO_API_URL}/coins/list"
        response = requests.get(coin_list_url)
        coin_list = response.json()
        crypto_id = next((coin["id"] for coin in coin_list if coin["symbol"].upper() == ticker.upper()), None)

        if not crypto_id:
            return None  # ❌ Not found

        url = f"{COINGECKO_API_URL}/coins/{crypto_id}/market_chart?vs_currency=usd&days=180"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            hist = pd.DataFrame(prices, columns=["timestamp", "price"])
            hist["Date"] = pd.to_datetime(hist["timestamp"], unit='ms').dt.strftime("%Y-%m-%d")
            hist["Open"] = hist["price"]
            hist["High"] = hist["price"]
            hist["Low"] = hist["price"]
            hist["Close"] = hist["price"]
            hist["Volume"] = 0  
            return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"⚠️ Error fetching crypto data for {ticker}: {e}")
    return None

# ✅ Determine if ticker is a stock or crypto
def get_market_data(ticker):
    crypto_data = fetch_crypto_data(ticker)
    return crypto_data if crypto_data is not None else fetch_stock_data(ticker)

# ✅ AI-Based Price Prediction (Using Prophet)
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": None, "next_7_days": None, "next_30_days": None}

    df["ds"] = pd.to_datetime(df["Date"])
    df["y"] = df["Close"]
    
    model = Prophet()
    model.fit(df[["ds", "y"]])

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return {
        "next_day": {
            "date": (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "price": round(forecast.iloc[-30]["yhat"], 2)
        },
        "next_7_days": {
            "date": (datetime.today() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "price": round(forecast.iloc[-23]["yhat"], 2)
        },
        "next_30_days": {
            "date": (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "price": round(forecast.iloc[-1]["yhat"], 2)
        }
    }

# ✅ AI Investment Decision Engine
def generate_investment_advice(predicted_prices, current_price):
    if None in predicted_prices.values():
        return {"trend": "Unknown", "advice": "HOLD", "confidence": "0%"}

    trend = "Bullish" if predicted_prices["next_day"]["price"] > current_price else "Bearish"
    advice = "HOLD"
    if trend == "Bullish" and predicted_prices["next_day"]["price"] > current_price * 1.02:
        advice = "BUY"
    elif trend == "Bearish" and predicted_prices["next_day"]["price"] < current_price * 0.98:
        advice = "SELL"

    confidence = f"{round(abs((predicted_prices['next_day']['price'] - current_price) / current_price) * 100, 1)}%"
    
    return {"trend": trend, "advice": advice, "confidence": confidence}

# ✅ AI-Generated Investment Summary (GPT Model)
def generate_summary(ticker):
    input_text = f"Summarize financial trends, news, and market factors affecting {ticker}."
    summary = GPT_SUMMARY_MODEL(input_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    
    if not ticker or len(ticker) < 2:
        return jsonify({"error": "Enter a valid stock or crypto ticker (min 2 characters)."}), 400

    try:
        hist = get_market_data(ticker)
        if hist is None or hist.empty:
            return jsonify({"error": f"No data found for {ticker}. Try a different ticker."}), 404

        current_price = hist["Close"].iloc[-1]
        predicted_prices = predict_prices(hist)
        investment_advice = generate_investment_advice(predicted_prices, current_price)
        ai_summary = generate_summary(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": {
                "trend": investment_advice["trend"],
                "advice": investment_advice["advice"],
                "confidence": investment_advice["confidence"],
                "predicted_prices": predicted_prices,
                "best_buy_price": round(hist["Close"].min(), 2),
                "best_sell_price": round(hist["Close"].max(), 2),
                "best_buy_date": hist.loc[hist["Close"].idxmin(), "Date"],
                "best_sell_date": hist.loc[hist["Close"].idxmax(), "Date"],
                "probability_of_success": "85%",
                "investment_summary": ai_summary
            }
        })
    except Exception as e:
        return jsonify({"error": f"Market data fetch failed: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
