from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from ai_model import predict_stock  # Import AI model

# ✅ Force Redeploy - Final Timestamp Fix
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

        # ✅ Convert DataFrame index (Timestamp) into a column and format correctly
        hist = hist.reset_index()

        # ✅ Convert ALL datetime columns to strings
        for col in hist.select_dtypes(include=["datetime64"]).columns:
            hist[col] = hist[col].astype(str)

        # ✅ Ensure all values are JSON serializable
        hist = hist.astype({
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Volume": "int"
        })

        # ✅ Keep only necessary columns
        hist = hist[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # ✅ Get AI-Based Predictions
        prediction_result = predict_stock(hist)

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
