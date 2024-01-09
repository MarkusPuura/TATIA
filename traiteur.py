import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Remplacez 'votre_fichier.csv' par le chemin vers votre fichier CSV
df = pd.read_csv('ExtractedTweets.csv')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

df['ProcessedTweet'] = df['Tweet'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['ProcessedTweet'], df['Party'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

input_text = "I'm gonna build a wall between Mexico and China, and I want to abolish ObamaCare"
input_text_processed = preprocess_text(input_text)
input_vectorized = vectorizer.transform([input_text_processed])

prediction = classifier.predict(input_vectorized)
print(f"The input text is predicted to be from the {prediction[0]} party.")

