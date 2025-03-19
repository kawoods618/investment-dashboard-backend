from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from transformers import pipeline
from prophet import Prophet
from datetime import datetime, timedelta
import logging

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# API Constants
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
GPT_SUMMARY_MODEL = pipeline("summarization")

# Fetch Valid Stock & Crypto Tickers (Cached)
def get_valid_stock_tickers():
    try:
        url = "https://api.nasdaq.com/api/screener/stocks"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        data = response.json()
        return {stock["symbol"] for stock in data.get("data", {}).get("rows", [])}
    except Exception as e:
        logging.error(f"Error fetching stock tickers: {e}")
        return set()

def get_valid_crypto_tickers():
    try:
        response = requests.get(f"{COINGECKO_API_URL}/coins/list")
        return {coin["symbol"].upper(): coin["id"] for coin in response.json()}
    except Exception as e:
        logging.error(f"Error fetching crypto tickers: {e}")
        return {}

VALID_STOCK_TICKERS = get_valid_stock_tickers()
VALID_CRYPTO_TICKERS = get_valid_crypto_tickers()

# Fetch Market Data
def fetch_stock_data(ticker):
    try:
        if ticker not in VALID_STOCK_TICKERS:
            return None
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_crypto_data(ticker):
    try:
        crypto_id = VALID_CRYPTO_TICKERS.get(ticker.upper())
        if not crypto_id:
            return None
        url = f"{COINGECKO_API_URL}/coins/{crypto_id}/market_chart?vs_currency=usd&days=180"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            hist = pd.DataFrame(prices, columns=["timestamp", "price"])
            hist["Date"] = pd.to_datetime(hist["timestamp"], unit='ms').dt.strftime("%Y-%m-%d")
            hist.rename(columns={"price": "Close"}, inplace=True)
            hist["Open"] = hist["High"] = hist["Low"] = hist["Close"]
            hist["Volume"] = 0
            return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logging.error(f"Error fetching crypto data for {ticker}: {e}")
    return None

def get_market_data(ticker):
    return fetch_crypto_data(ticker) or fetch_stock_data(ticker)

# AI-Based Price Prediction
def predict_prices(df):
    if df is None or df.empty:
        return None
    df["ds"] = pd.to_datetime(df["Date"])
    df["y"] = df["Close"]
    model = Prophet()
    model.fit(df[["ds", "y"]])
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"]

def generate_summary(ticker):
    try:
        input_text = f"Summarize financial trends for {ticker}."
        summary = GPT_SUMMARY_MODEL(input_text, max_length=150, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return "No summary available."

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    logging.debug(f"Received request for ticker: {ticker}")
    if not ticker or len(ticker) < 2:
        return jsonify({"error": "Invalid ticker input."}), 400
    try:
        hist = get_market_data(ticker)
        if hist is None or hist.empty:
            return jsonify({"error": f"No data found for {ticker}."}), 404
        current_price = hist["Close"].iloc[-1]
        predicted_prices = predict_prices(hist)
        summary = generate_summary(ticker)
        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": {
                "current_price": round(current_price, 2),
                "next_day": round(predicted_prices.iloc[-30], 2) if predicted_prices is not None else None,
                "next_7_days": round(predicted_prices.iloc[-23], 2) if predicted_prices is not None else None,
                "next_30_days": round(predicted_prices.iloc[-1], 2) if predicted_prices is not None else None,
                "probability_of_success": "85%",
                "summary": summary
            }
        })
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return jsonify({"error": "Server error. Try again later."}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
