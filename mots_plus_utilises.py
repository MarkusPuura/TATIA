import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

# Chargement des données
df = pd.read_csv('ExtractedTweets.csv')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    
    # Stemming and remove stop words, exclude specific words and those starting with "@" or "#"
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in ['http', 'amp', 'rt'] and not (word.startswith('@') or word.startswith('#'))]
    
    return ' '.join(words)

# Prétraitement des tweets
df['ProcessedTweet'] = df['Tweet'].apply(preprocess_text)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['ProcessedTweet'], df['Party'], test_size=0.2, random_state=42)

# Classification avec un classificateur SVM linéaire
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
classifier = LinearSVC(dual=False, max_iter=10000)
classifier.fit(X_train_vectorized, y_train)

# Utiliser le CountVectorizer pour obtenir les mots originaux
feature_names = np.array(vectorizer.get_feature_names_out())

# Obtenir les indices des 5 premiers mots les plus fréquemment utilisés par les démocrates
democrat_indices = np.argsort(classifier.coef_[0])[-5:][::-1]
democrat_words = feature_names[democrat_indices]
print(f"Top 5 words used by Democrats: {democrat_words}")

# Obtenir les indices des 5 premiers mots les plus fréquemment utilisés par les républicains
republican_indices = np.argsort(classifier.coef_[0])[:5]
republican_words = feature_names[republican_indices]
print(f"Top 5 words used by Republicans: {republican_words}")
