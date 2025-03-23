# app.py
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Dummy forecast function for demo purposes
def predict_prices(df):
    try:
        current_price = df['y'].iloc[-1]

        # Dummy forecast values
        buy_price = round(current_price * 0.98, 2)
        sell_price = round(current_price * 1.08, 2)

        buy_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        sell_date = (datetime.now() + timedelta(days=25)).strftime('%Y-%m-%d')

        probability = round(100 - abs((sell_price - buy_price) / buy_price) * 20, 2)

        return {
            "next_day": round(current_price * 1.01, 2),
            "next_week": round(current_price * 1.03, 2),
            "next_month": round(current_price * 1.07, 2),
            "probability": probability,
            "buy_price": buy_price,
            "buy_date": buy_date,
            "sell_price": sell_price,
            "sell_date": sell_date,
        }

    except Exception as e:
        print("❌ Error in predict_prices():", e)
        return {
            "next_day": "N/A",
            "next_week": "N/A",
            "next_month": "N/A",
            "probability": "N/A",
            "buy_price": "N/A",
            "buy_date": "N/A",
            "sell_price": "N/A",
            "sell_date": "N/A"
        }

# Simulated news summarizer
def summarize_news(ticker):
    try:
        ticker = ticker.upper()
        return f"""
        📈 Market Insights for {ticker}:
        - {ticker} has seen a rise in investor sentiment based on recent earnings projections.
        - Analysts are optimistic due to expansion into AI and cloud sectors.
        - Recent trading volume indicates institutional buying pressure.
        - Technical indicators show a potential breakout approaching resistance.

        Based on these factors, our AI forecasts a short-term upward trend, suggesting a timely buy-in followed by a targeted sell-off within the next 3–4 weeks.
        """
    except Exception as e:
        print("Error in summarize_news:", e)
        return "No financial news available."

# Dummy data fetcher

def fetch_real_time_data(ticker):
    dates = pd.date_range(end=datetime.now(), periods=90)
    prices = [100 + i * 0.1 + (i % 5) for i in range(len(dates))]
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
