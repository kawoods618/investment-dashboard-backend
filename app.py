from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import requests
import pandas as pd
from prophet import Prophet
import traceback

app = Flask(__name__)

# ✅ Allow frontend requests (CORS Fix)
CORS(app, resources={r"/api/*": {"origins": "https://investment-dashboard-frontend-production.up.railway.app"}}, supports_credentials=True)

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
        
        # ✅ Fix: Ensure correct frequency
        df["ds"] = pd.to_datetime(df["ds"])  # Convert to datetime format
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)
        
        return {
            "next_day": f"${round(forecast.iloc[-30]['yhat'], 2)}",
            "next_week": f"${round(forecast.iloc[-7]['yhat'], 2)}",
            "next_month": f"${round(forecast.iloc[-1]['yhat'], 2)}"
        }
    except Exception as e:
        print("Error in predict_prices:", e)
        return {"next_day": "Error", "next_week": "Error", "next_month": "Error"}

# ✅ Fetch Financial News
def fetch_financial_news(ticker):
    API_KEY = "YOUR_NEWSAPI_KEY"  # 🔥 Replace this with an actual key!
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [
                {"title": article["title"], "summary": article["description"]}
                for article in articles[:5]
            ]
        else:
            print("News API Error:", response.text)  # 🔥 Debugging output
            return [{"title": "No news available.", "summary": "No recent updates."}]
    
    except Exception as e:
        print("Error in fetch_financial_news:", e)
        return [{"title": "Error fetching news.", "summary": str(e)}]

# ✅ Fetch Congress Trading Data
def fetch_congress_trading(ticker):
    API_URL = "https://api.quiverquant.com/beta/live/housetrading"
    headers = {"Authorization": "Bearer YOUR_QUIVERQUANT_API_KEY"}  # 🔥 Replace with valid API Key
    
    try:
        response = requests.get(API_URL, headers=headers)
        if response.status_code == 200:
            trades = [trade for trade in response.json() if trade.get("Ticker") == ticker]
            if not trades:
                return [{"message": "No congress trades found for this ticker."}]
            return trades
        else:
            print("Congress Trading API Error:", response.text)  # 🔥 Debugging output
            return [{"message": "Error retrieving congress trades."}]
    
    except Exception as e:
        print("Error in fetch_congress_trading:", e)
        return [{"message": "Congress trading data not available."}]

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
        news = fetch_financial_news(ticker)
        congress_trades = fetch_congress_trading(ticker)

        return jsonify({
            "ticker": ticker,
            "market_data": hist.to_dict(orient="records"),
            "predictions": predicted_prices,
            "news": news,
            "congress_trades": congress_trades
        })
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
