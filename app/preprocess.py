import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = str(text).lower()
    text = text.replace("subject", "")

    text = "".join([
        char for char in text
        if char not in string.punctuation
    ])

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)