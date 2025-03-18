from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuantumVest AI Backend is Running!"})

@app.route("/api/health", methods=["GET"])
def get_health():
    return jsonify({"ok": True})

@app.route("/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", auto_adjust=True)

        # ✅ Ensure data exists
        if hist.empty:
            return jsonify({"error": f"No data found for ticker {ticker}"}), 404

        # ✅ Convert DataFrame index (Timestamp) into a column and remove timezone info
        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")  # ✅ Format: YYYY-MM-DD

        # ✅ Convert numerical values to standard JSON types
        hist = hist.astype({
            "Open": float,
            "High": float,
            "Low": float,
            "Close": float,
            "Volume": int
        })

        # ✅ Keep only necessary columns
        hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # ✅ AI Prediction Model (Using Linear Regression)
        df = hist.copy()
        df["Day"] = np.arange(len(df))  # Convert dates to numerical sequence

        # Train a simple linear regression model to predict future prices
        X = df[["Day"]]
        y = df["Close"]
        model = LinearRegression()
        model.fit(X, y)

        # Predict stock price for next trading day
        next_day = np.array([[len(df)]])
        predicted_price = round(model.predict(next_day)[0], 2)

        # Determine buy/sell dates based on past performance
        best_buy_date = df.loc[df["Close"].idxmin(), "Date"]
        best_sell_date = df.loc[df["Close"].idxmax(), "Date"]

        prediction_result = {
            "trend": "Bullish" if predicted_price > df["Close"].iloc[-1] else "Bearish",
            "confidence": f"{round(abs((predicted_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100, 1)}%",
            "predicted_price": predicted_price,
            "best_buy_date": best_buy_date,
            "best_sell_date": best_sell_date,
            "probability_of_success": "80%"  # Placeholder
        }

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": prediction_result
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch market data for {ticker}: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
