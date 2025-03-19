from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ✅ Fetch Real-Time Data from Yahoo Finance API
def fetch_real_time_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d", interval="1h", auto_adjust=True)

    if hist.empty:
        return None

    hist = hist.reset_index()
    hist["Date"] = hist["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    hist = hist.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})
    
    return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

# ✅ AI Investment Strategy Based on Market Patterns
def generate_investment_advice(predicted_prices, current_price):
    trend = "Bullish" if predicted_prices[0] > current_price else "Bearish"
    advice = "HOLD"
    
    if trend == "Bullish" and predicted_prices[0] > current_price * 1.02:
        advice = "BUY"
    elif trend == "Bearish" and predicted_prices[0] < current_price * 0.98:
        advice = "SELL"

    return {
        "trend": trend,
        "advice": advice,
        "confidence": f"{round(abs((predicted_prices[0] - current_price) / current_price) * 100, 1)}%",
    }

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        print(f"Fetching real-time data for: {ticker}")
        hist = fetch_real_time_data(ticker)

        if hist is None or hist.empty:
            return jsonify({"error": f"No real-time data found for {ticker}"}), 404

        df = hist.copy()
        df["Day"] = np.arange(len(df))
        
        X = df[["Day"]]
        y = df["Close"]
        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([[len(df) + i] for i in range(1, 8)])
        predicted_prices = model.predict(future_days).tolist()
        predicted_prices = [round(price, 2) for price in predicted_prices]

        investment_advice = generate_investment_advice(predicted_prices, df["Close"].iloc[-1])

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": {
                "trend": investment_advice["trend"],
                "advice": investment_advice["advice"],
                "confidence": investment_advice["confidence"],
                "predicted_prices": predicted_prices,
                "best_buy_date": df.loc[df["Close"].idxmin(), "Date"],
                "best_sell_date": df.loc[df["Close"].idxmax(), "Date"]
            }
        })

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return jsonify({"error": f"Failed to fetch market data for {ticker}: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
