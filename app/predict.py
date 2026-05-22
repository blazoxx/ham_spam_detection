import pickle

from app.preprocess import preprocess_text


model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


def predict_message(message):
    cleaned = preprocess_text(message)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    confidence = model.predict_proba(vectorized)[0].max()

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return result, confidence