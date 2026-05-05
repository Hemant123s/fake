import joblib
import json
import re

model = joblib.load('model.pkl')
vec = joblib.load('vectorizer.pkl')
arts = json.load(open('out.json'))['articles']

def clean(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for a in arts[:10]:
    title = a.get('title', '') or ''
    desc = a.get('description', '') or ''
    content = f"{title} {desc}"
    content_clean = clean(content)
    pred = model.predict(vec.transform([content_clean]))[0]
    print(pred)
