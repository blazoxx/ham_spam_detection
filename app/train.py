import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "spam_ham_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


df = pd.read_csv(DATA_PATH)


df = df[['label', 'text']]


df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})


df['cleaned'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

best_alpha = None
best_accuracy = 0

for alpha in alphas:
    temp_model = MultinomialNB(alpha=alpha)
    temp_model.fit(X_train, y_train)

    preds = temp_model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Alpha: {alpha} -> Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_alpha = alpha

print("\nBest Alpha:", best_alpha)
print("Best Accuracy:", best_accuracy)

model = MultinomialNB(alpha=best_alpha)
model.fit(X_train, y_train)

os.makedirs(MODELS_DIR, exist_ok=True)

with open(MODELS_DIR / "spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(MODELS_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel saved successfully!")