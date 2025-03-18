from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# ✅ Root Route (Fixes 404)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuantumVest AI Backend is Running!"})

# ✅ Health Check Route
@app.route("/api/health", methods=["GET"])
def get_health():
    return jsonify({"ok": True})

# ✅ Fetch Market Data from Yahoo Finance with Predictions
@app.route("/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        # ✅ Convert timestamps to strings
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)  # Convert datetime to string

        # ✅ Basic AI Prediction (Replace with real model)
        prediction = {
            "trend": "Bullish" if hist["Close"].iloc[-1] > hist["Close"].iloc[-5] else "Bearish",
            "confidence": "80%"  # Placeholder value
        }

        # ✅ Investment Recommendation
        recommendation = {
            "rating": "Strong Buy" if prediction["trend"] == "Bullish" else "Hold",
            "reason": "Stock is showing positive momentum based on historical trends."
        }

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "prediction": prediction,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch market data: {str(e)}"}), 500

# ✅ Ensure Flask Runs on Correct Port (for Railway)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # Set default port to 8080
    app.run(host="0.0.0.0", port=port, debug=True)
