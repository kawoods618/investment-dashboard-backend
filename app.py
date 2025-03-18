from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from ai_model import predict_stock  # Import AI model

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
        hist = stock.history(period="1y")

        # ✅ Ensure there is historical data
        if hist.empty:
            return jsonify({"error": "No historical data found for this ticker"}), 404

        # ✅ Reset index and convert Timestamp index to a string
        hist = hist.reset_index()
        
        # ✅ Convert ALL columns with datetime values to string
        hist["Date"] = hist["Date"].astype(str)

        # ✅ Keep only necessary columns
        hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # ✅ Get AI-Based Predictions, Buy/Sell Dates & Probability
        prediction_result = predict_stock(hist)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),  # Convert DataFrame to list of dictionaries
            "prediction": prediction_result
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch market data: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
