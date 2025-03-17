from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask Backend is Running!"})

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
        hist_df = stock.history(period="1y")
        hist_data = hist_df.reset_index().to_dict(orient="records")
        return jsonify({"ticker": ticker, "market_data": hist_data})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch market data: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
