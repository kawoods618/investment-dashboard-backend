from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuantumVest AI Backend is Running!"})

@app.route("/api/health", methods=["GET"])
def get_health():
    return jsonify({"ok": True})

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        print(f"Fetching data for: {ticker}")

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", auto_adjust=True)

        if hist.empty:
            print(f"No data found for {ticker}")
            return jsonify({"error": f"No historical data available for {ticker}"}), 404

        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        hist = hist.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})
        hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

        df = hist.copy()
        df["Day"] = np.arange(len(df))

        X = df[["Day"]]
        y = df["Close"]
        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([[len(df) + i] for i in range(1, 8)])
        predicted_prices = model.predict(future_days).tolist()
        predicted_prices = [round(price, 2) for price in predicted_prices]

        best_buy_date = df.loc[df["Close"].idxmin(), "Date"]
        best_sell_date = df.loc[df["Close"].idxmax(), "Date"]

        prediction_result = {
            "trend": "Bullish" if predicted_prices[0] > df["Close"].iloc[-1] else "Bearish",
            "confidence": f"{round(abs((predicted_prices[0] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100, 1)}%",
            "predicted_prices": predicted_prices,
            "best_buy_date": best_buy_date,
            "best_sell_date": best_sell_date,
            "probability_of_success": "80%"
        }

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": prediction_result
        })

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return jsonify({"error": f"Failed to fetch market data for {ticker}: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
