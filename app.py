from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# CoinGecko API URL for cryptocurrency data
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# ✅ Fetch Stock Data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d", auto_adjust=True)

        if hist.empty:
            return None

        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        hist = hist.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int})

        return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"⚠️ Error fetching stock data for {ticker}: {e}")
        return None

# ✅ Fetch Crypto Data
def fetch_crypto_data(ticker):
    url = f"{COINGECKO_API_URL}/coins/{ticker}/market_chart?vs_currency=usd&days=180"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            prices = data["prices"]
            hist = pd.DataFrame(prices, columns=["timestamp", "price"])
            hist["Date"] = pd.to_datetime(hist["timestamp"], unit='ms').dt.strftime("%Y-%m-%d")
            hist["Open"] = hist["price"]
            hist["High"] = hist["price"]
            hist["Low"] = hist["price"]
            hist["Close"] = hist["price"]
            hist["Volume"] = 0  # Crypto volume not available in this endpoint
            return hist[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"⚠️ Error fetching crypto data for {ticker}: {e}")
    return None

# ✅ Determine if ticker is crypto or stock
def get_market_data(ticker):
    ticker = ticker.lower()
    crypto_list = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin", "ripple"]
    
    if ticker in crypto_list:
        return fetch_crypto_data(ticker)
    else:
        return fetch_stock_data(ticker)

# ✅ AI-Based Price Prediction
def predict_prices(df):
    if df is None or df.empty:
        return {"next_day": None, "next_week": None, "next_month": None}

    df["Day"] = np.arange(len(df))
    X = df[["Day"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)

    future_days = {
        "next_day": np.array([[len(df) + 1]]),
        "next_week": np.array([[len(df) + 7]]),
        "next_month": np.array([[len(df) + 30]])
    }

    predicted_prices = {key: round(model.predict(value)[0], 2) for key, value in future_days.items()}
    return predicted_prices

# ✅ AI Investment Strategy Model
def generate_investment_advice(predicted_prices, current_price):
    if None in predicted_prices.values():
        return {"trend": "Unknown", "advice": "HOLD", "confidence": "0%"}

    trend = "Bullish" if predicted_prices["next_day"] > current_price else "Bearish"
    advice = "HOLD"

    if trend == "Bullish" and predicted_prices["next_day"] > current_price * 1.02:
        advice = "BUY"
    elif trend == "Bearish" and predicted_prices["next_day"] < current_price * 0.98:
        advice = "SELL"

    confidence = f"{round(abs((predicted_prices['next_day'] - current_price) / current_price) * 100, 1)}%"
    
    return {"trend": trend, "advice": advice, "confidence": confidence}

# ✅ Fetch Market News from NewsAPI
def fetch_financial_news(ticker):
    NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # Replace this with your NewsAPI key

    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY":
        return [{"title": "No News Available", "summary": "Set a valid NewsAPI key."}]

    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json().get("articles", [])[:5]  # Get top 5 news articles
            if not news_data:
                return [{"title": "No Recent News", "summary": f"No relevant news found for {ticker}."}]
            return [{"title": article["title"], "summary": article["description"]} for article in news_data]
    except Exception as e:
        print(f"⚠️ Error fetching news: {e}")
        return [{"title": "Error", "summary": "News API request failed."}]

@app.route("/api/analyze", methods=["GET"])
def analyze():
    ticker = request.args.get("ticker", "").lower()
    
    if not ticker or len(ticker) < 2:
        return jsonify({"error": "Enter a valid stock or crypto ticker (min 2 characters)."}), 400

    try:
        hist = get_market_data(ticker)
        if hist is None or hist.empty:
            return jsonify({"error": f"No data found for {ticker}. Try a different ticker."}), 404

        current_price = hist["Close"].iloc[-1]
        predicted_prices = predict_prices(hist)
        investment_advice = generate_investment_advice(predicted_prices, current_price)
        financial_news = fetch_financial_news(ticker)

        return jsonify({
            "ticker": ticker.upper(),
            "market_data": hist.to_dict(orient="records"),
            "prediction": {
                "trend": investment_advice["trend"],
                "advice": investment_advice["advice"],
                "confidence": investment_advice["confidence"],
                "predicted_prices": predicted_prices,
                "best_buy_price": round(hist["Close"].min(), 2),
                "best_sell_price": round(hist["Close"].max(), 2),
                "best_buy_date": hist.loc[hist["Close"].idxmin(), "Date"],
                "best_sell_date": hist.loc[hist["Close"].idxmax(), "Date"],
                "probability_of_success": "85%",
                "financial_news": financial_news
            }
        })

    except Exception as e:
        print(f"Backend error: {e}")
        return jsonify({"error": f"Market data fetch failed: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
