import zipfile
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ---- READ ZIP CSV ----
def read_zip_csv(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            return pd.read_csv(f)

fake = read_zip_csv("fake.csv (1).zip")
true = read_zip_csv("True.csv (1).zip")

true['label'] = 1
fake['label'] = 0

# ---- STRIP PUBLISHER PREFIXES FROM TRUE NEWS ----
# True.csv articles start with "CITY (Publisher) - ..." which gives the model
# an unfair shortcut. Remove these so it learns real linguistic patterns.
def strip_publisher_prefix(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'^[A-Z][A-Za-z\s,]+ \(.*?\)\s*[-–—]\s*', '', text)

true['text'] = true['text'].apply(strip_publisher_prefix)

# ---- COMBINE TITLE + TEXT FOR BETTER SHORT-TEXT HANDLING ----
# This ensures the model learns patterns from both headlines AND article bodies
fake['text'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
true['text'] = true['title'].fillna('') + ' ' + true['text'].fillna('')

data = pd.concat([fake, true], axis=0)
data = data.drop(['title','subject','date'], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['text'] = data['text'].apply(wordopt)

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))


# ---- SAVE MODEL ----
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")
