from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import requests
import re
import os
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Serve frontend ──
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 🔑 Use your NewsAPI key (NOT GNews if using newsapi.org)
NEWS_API_KEY = "042f5843f5974cd9a80fadfda70c2eac"

# ✅ Load ML model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ✅ Text cleaning function
def clean(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# 🔹 MANUAL PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.json.get("text", "")
        text_clean = clean(text)

        vec = vectorizer.transform([text_clean])
        result = model.predict(vec)[0]

        label = "Real News" if int(result) == 1 else "Fake News"

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🔹 LIVE NEWS API
# =========================
@app.route("/live", methods=["POST"])
def live_news():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            query = "india"  # default fallback

        # ✅ Correct date format (last 2 days)
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

        url = f"https://newsapi.org/v2/everything?q={query}&from={two_days_ago}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"

        print("Fetching:", url)  # debug

        resp = requests.get(url, timeout=10)
        news_data = resp.json()

        # ❌ Handle API errors
        if news_data.get("status") != "ok":
            return jsonify({
                "error": news_data.get("message", "API error")
            }), 400

        articles = news_data.get("articles", [])[:5]

        if not articles:
            return jsonify({"error": "No news found"}), 200

        results = []

        for art in articles:
            title = art.get("title", "") or ""
            desc = art.get("description", "") or ""
            image = art.get("urlToImage", "") or ""
            url_link = art.get("url", "") or ""

            content = f"{title} {desc}"
            content_clean = clean(content)

            vec = vectorizer.transform([content_clean])
            pred = model.predict(vec)[0]

            label = "Real" if int(pred) == 1 else "Fake"

            results.append({
                "title": title,
                "description": desc,
                "image": image,
                "url": url_link,
                "prediction": label
            })

        return jsonify({"articles": results}), 200

    except Exception as e:
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500


# =========================
# 🔹 HOME ROUTE
# =========================
@app.route("/")
def home():
    return "✅ Fake News Detector API Running!"


# =========================
# 🔹 RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)




    
