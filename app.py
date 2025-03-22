from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from prophet import Prophet
from bs4 import BeautifulSoup
import feedparser
from transformers import pipeline
import traceback

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Load local AI summarizer model (lightweight + accurate)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.errorhandler(Exception)
def handle_exception(e):
    print("ERROR:", traceback.format_exc())
    return jsonify({"error": str(e)}), 500

# ✅ Get 6 months of historical stock data
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

# ✅ Forecast prices using Prophet
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A", "probability": "N/A"}
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
            "probability": 80.0  # Static for now, can improve later
        }
    except Exception as e:
        print("Error in predict_prices:", e)
        return {"next_day": "N/A", "next_week": "N/A", "next_month": "N/A", "probability": "N/A"}

# ✅ Get and summarize news using RSS + Transformers
def summarize_news(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        descriptions = []

        for entry in feed.entries[:5]:
            soup = BeautifulSoup(entry.description, "html.parser")
            text = soup.get_text()
            descriptions.append(text)

        combined_text = " ".join(descriptions)
        if not combined_text.strip():
            return "No financial news available."

        summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print("Error in summarize_news:", e)
        return "No financial news available."

# ✅ API endpoint
@app.route("/api/analyze", methods=["GET"])
def analyze():
    try:
        ticker = request.args.get("ticker", "").upper()
        if not ticker or len(ticker) < 2:
            return jsonify({"error": "Enter a valid stock or crypto ticker."}), 400

        hist = fetch_real_time_data(ticker)
        if hist is None:
            return jsonify({"error": f"No data found for {ticker}."}), 404

        predictions = predict_prices(hist)
        news_summary = summarize_news(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "predictions": predictions,
            "news_summary": news_summary
        })
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
