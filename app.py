from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import requests
import pandas as pd
from prophet import Prophet
import traceback

app = Flask(__name__)

# ✅ Allow all frontend requests (Fixes CORS)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

@app.errorhandler(Exception)
def handle_exception(e):
    print("ERROR:", traceback.format_exc())  # Debugging
    response = jsonify({"error": str(e)})
    response.status_code = 500
    return response

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

# ✅ AI Price Prediction using Prophet
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A"}
    try:
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        return {
            "next_day": round(forecast.iloc[-30]["yhat"], 2),
            "next_week": round(forecast.iloc[-7]["yhat"], 2),
            "next_month": round(forecast.iloc[-1]["yhat"], 2),
        }
    except Exception as e:
        print("Error in predict_prices:", e)
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A"}

# ✅ Summarize News for AI Insights
def summarize_news(ticker):
    API_KEY = "YOUR_NEWSAPI_KEY"
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])[:5]
            summary = " ".join([article["description"] for article in articles if article["description"]])
            return summary if summary else "No financial news available."
    except Exception as e:
        print("Error in summarize_news:", e)
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
        news_summary = summarize_news(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "predictions": predicted_prices,
            "news_summary": news_summary,
        })
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
