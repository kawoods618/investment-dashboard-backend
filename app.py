from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from ai_model import predict_stock  # Import AI model

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        # ✅ Convert timestamps to string for JSON serialization
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)  # Convert datetime to string

        # ✅ Get AI-Based Predictions, Buy/Sell Dates & Probability
        prediction_result = predict_stock(hist)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),  # Convert entire DataFrame to list of dictionaries
            "prediction": prediction_result
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch market data: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
