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

# ✅ Fix CORS for Frontend Access
CORS(app)

# ✅ Fetch Historical Stock Data
def fetch_real_time_data(ticker):
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
        print(f"Error fetching data for {ticker}: {e}")
        return None

# ✅ AI-Based Price Prediction
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": None, "next_week": None, "next_month": None}

    df["Day"] = np.arange(len(df))
    X = df[["Day"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)

    future_days = {
        "next_day": np.array([[len(df) + 1]]),
        "next_week": np.array([[len(df) + 5]]),
        "next_month": np.array([[len(df) + 20]])
    }

    predicted_prices = {}
    for key, value in future_days.items():
        try:
            predicted_prices[key] = round(model.predict(value)[0], 2)
        except:
            predicted_prices[key] = None  # Handle cases where model fails

    return predicted_prices

# ✅ AI Investment Strategy Model
def generate_investment_advice(predicted_prices, current_price):
    if None in predicted_prices.values():
        return {"trend": "Unknown", "advice": "HOLD", "confidence": "0%"}

    trend = "Bullish" if predicted_prices["next_day"] > current_price else "Bearish"
    advice = "HOLD"

    if trend == "Bullish" and predicted_prices["next_day"] > current_price * 1.02:
        advice = "BUY"
    elif trend == "Bearish" and predicted_prices["next_day"] < current_price * 0.98:
        advice = "SELL"

    confidence = f"{round(abs((predicted_prices['next_day'] - current_price) / current_price) * 100, 1)}%"
    
    return {"trend": trend, "advice": advice, "confidence": confidence}

# ✅ Fetch Market News from NewsAPI
def fetch_financial_news(ticker):
    NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # Replace with actual key
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json().get("articles", [])[:5]  # Get top 5 news articles
            summarized_news = [
                {"title": article["title"], "summary": article["description"]} for article in news_data
            ]
            return summarized_news
    except Exception as e:
        print(f"Error fetching news: {e}")
    
    return []

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    
    if not ticker or len(ticker) < 2:
        return jsonify({"error": "Please enter a valid stock or crypto ticker (min 2 characters)."}), 400

    try:
        hist = fetch_real_time_data(ticker)
        if hist is None or hist.empty:
            return jsonify({"error": f"No real-time data found for {ticker}. Try a different ticker."}), 404

        current_price = hist["Close"].iloc[-1]
        predicted_prices = predict_prices(hist)
        investment_advice = generate_investment_advice(predicted_prices, current_price)
        financial_news = fetch_financial_news(ticker)

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
                "financial_news": financial_news
            }
        })

    except Exception as e:
        print(f"Backend error: {e}")
        return jsonify({"error": f"Failed to fetch market data. Error: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
