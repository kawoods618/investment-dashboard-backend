from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import requests
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np

app = Flask(__name__)

# ✅ FIXED CORS ISSUE: Allow all requests from frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ✅ Fetch Real-Time Stock Data
def fetch_real_time_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print("Error in fetch_real_time_data:", e)
        return None

# ✅ AI Prediction Model
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A"}

    try:
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["y"].values

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        future_days = [len(df) + i for i in [1, 7, 30]]
        predictions = model.predict(np.array(future_days).reshape(-1, 1))

        return {
            "next_day": round(predictions[0], 2),
            "next_week": round(predictions[1], 2),
            "next_month": round(predictions[2], 2),
        }

    except Exception as e:
        print("Error in predict_prices:", e)
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A"}

# ✅ Fetch and Summarize Financial News
def fetch_financial_news(ticker):
    API_KEY = "YOUR_NEWSAPI_KEY"
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if not articles:
                return "No significant news found."

            # Summarizing top 5 news articles
            summary = "\n".join([f"- {article['title']}: {article['description']}" for article in articles[:5]])
            return summary

    except Exception as e:
        print("Error in fetch_financial_news:", e)

    return "No financial news available."

# ✅ API Route
@app.route("/api/analyze", methods=["GET"])
def analyze():
    try:
        ticker = request.args.get("ticker", "").upper()
        if not ticker or len(ticker) < 2:
            return jsonify({"error": "Enter a valid stock or crypto ticker."}), 400

        hist = fetch_real_time_data(ticker)
        if hist is None:
            return jsonify({"error": f"No data for {ticker}."}), 404

        predicted_prices = predict_prices(hist)
        news_summary = fetch_financial_news(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "predictions": predicted_prices,
            "news_summary": news_summary
        })
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
