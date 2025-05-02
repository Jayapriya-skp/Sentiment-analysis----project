import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

data = {
    'text': [
        'I love this product!',
        'This is the worst thing I bought.',
        'Absolutely fantastic experience.',
        'I hate how this works.',
        'Not bad, could be better.',
        'Really happy with the purchase.',
        'Worst service ever.',
        'Amazing quality and service.',
        'Disappointed and sad.',
        'Excellent performance!'
    ],
    'sentiment': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['sentiment']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

sample = ["I am not happy with this service"]
sample_clean = [clean_text(s) for s in sample]
sample_vec = vectorizer.transform(sample_clean)
print("Predicted Sentiment:", "Positive" if model.predict(sample_vec)[0] == 1 else "Negative")
