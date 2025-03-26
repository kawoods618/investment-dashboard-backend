# app.py — Enhanced with CORS, Forecast Logic, and Ticker-Based News
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app)

# Simulated backend logic for investment predictions and strategy
def predict_prices(df):
    try:
        current_price = df['y'].iloc[-1]

        # Generate dummy forecast based on simple growth assumptions
        next_day = round(current_price * (1 + random.uniform(-0.01, 0.02)), 2)
        next_week = round(current_price * (1 + random.uniform(-0.03, 0.05)), 2)
        next_month = round(current_price * (1 + random.uniform(-0.05, 0.12)), 2)

        # Simulated investment recommendation logic
        buy_price = round(current_price * random.uniform(0.97, 0.99), 2)
        sell_price = round(current_price * random.uniform(1.06, 1.12), 2)

        buy_date = (datetime.now() + timedelta(days=random.randint(1, 3))).strftime('%Y-%m-%d')
        sell_date = (datetime.now() + timedelta(days=random.randint(20, 35))).strftime('%Y-%m-%d')

        expected_return = round(((sell_price - buy_price) / buy_price) * 100, 2)
        probability = round(85 - abs(expected_return * 0.5), 2)

        return {
            "next_day": next_day,
            "next_week": next_week,
            "next_month": next_month,
            "buy_price": buy_price,
            "buy_date": buy_date,
            "sell_price": sell_price,
            "sell_date": sell_date,
            "probability": probability,
        }

    except Exception as e:
        print("❌ Error in predict_prices():", e)
        return {
            "next_day": "N/A",
            "next_week": "N/A",
            "next_month": "N/A",
            "buy_price": "N/A",
            "buy_date": "N/A",
            "sell_price": "N/A",
            "sell_date": "N/A",
            "probability": "N/A"
        }

# Enhanced simulated news generation (ticker-based)
def summarize_news(ticker):
    try:
        ticker = ticker.upper()
        examples = {
            "TSLA": [
                "Tesla is expanding production capacity in Europe and Asia.",
                "Recent earnings exceeded Wall Street expectations.",
                "CEO Elon Musk hinted at upcoming AI breakthroughs.",
                "Institutional investors have increased their holdings."
            ],
            "AAPL": [
                "Apple announces new AI-driven chips for iPhones.",
                "Record-breaking quarterly revenue reported.",
                "Expansion into financial services expected this year.",
                "Strong buy signals from hedge funds and ETFs."
            ],
            "MSFT": [
                "Microsoft strengthens its cloud market dominance.",
                "Partnership with OpenAI boosts innovation potential.",
                "Positive analyst sentiment in tech sector recovery.",
                "Solid dividend returns continue to attract investors."
            ]
        }
        fallback = [
            f"{ticker} shows growing investor confidence in recent sessions.",
            f"Analysts forecast a bullish trend for {ticker}.",
            f"Volume spikes indicate rising institutional activity in {ticker}.",
            f"Technical indicators suggest a breakout may be forming."
        ]

        bullet_points = examples.get(ticker, fallback)

        summary = f"\n📊 News Summary for {ticker}:\n" + "\n".join(f"- {point}" for point in bullet_points)
        summary += f"\n\n✅ Justification: Given the above developments, {ticker} is forecasted to trend upward in the short-term, supporting a buy signal and medium-term growth target."
        return summary

    except Exception as e:
        print("Error in summarize_news:", e)
        return "No financial news available."

# Simulated historical price generator
def fetch_real_time_data(ticker):
    dates = pd.date_range(end=datetime.now(), periods=90)
    base_price = 100 + (hash(ticker) % 100)  # semi-random base by ticker
    prices = [round(base_price + random.uniform(-3, 3) + i * 0.3, 2) for i in range(len(dates))]
    df = pd.DataFrame({"ds": dates, "y": prices})
    return df

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
