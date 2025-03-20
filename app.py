from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import requests
import pandas as pd
from prophet import Prophet
import traceback

app = Flask(__name__)

# ✅ Corrected CORS Policy: Allow All Requests from Frontend
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

@app.after_request
def add_cors_headers(response):
    """Ensure CORS headers are added to every response."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    print("ERROR:", traceback.format_exc())  # Debugging
    response = jsonify({"error": str(e)})
    response.status_code = 500
    return response

# ✅ Fetch Historical Stock Data
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

# ✅ AI-Based Market News Summary (Instead of Linking)
def fetch_news_summary(ticker):
    API_KEY = "YOUR_NEWSAPI_KEY"
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])[:5]
            if not articles:
                return "No relevant financial news found."
            
            # Generate a basic summary
            summary = f"Recent financial news for {ticker}: "
            for article in articles:
                summary += f"{article['title']}. {article['description']} "

            return summary.strip()
    except Exception as e:
        print("Error in fetch_news_summary:", e)
    return "No financial news available."

# ✅ Fetch Congress Trading Data
def fetch_congress_trading(ticker):
    API_URL = "https://api.quiverquant.com/beta/live/housetrading"
    headers = {"Authorization": "Bearer YOUR_QUIVERQUANT_API_KEY"}
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            return [trade for trade in response.json() if trade.get("Ticker") == ticker]
    except Exception as e:
        print("Error in fetch_congress_trading:", e)
    return []

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
        news_summary = fetch_news_summary(ticker)
        congress_trades = fetch_congress_trading(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "predictions": predicted_prices,
            "news_summary": news_summary,
            "congress_trades": congress_trades
        })
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
