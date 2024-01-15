import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

# Chargement des données
df = pd.read_csv('ExtractedTweets.csv')

def preprocess_text(text, exclude_words=None):
    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Exclude specific words and those starting with "@" or "#"
    filtered_words = []
    skip_next_word = False

    for word in words:
        # Check if the word starts with "@" or "#"
        if re.match(r'^[#]', word):
            skip_next_word = True
        elif skip_next_word:
            # Skip the next word if the previous word started with "@" or "#"
            skip_next_word = False
        elif exclude_words and word.lower() in exclude_words:
            # Exclude specific words
            continue
        else:
            # Perform stemming for non-excluded words
            filtered_words.append(stemmer.stem(word))

    # Join the processed words back into a string
    processed_text = ' '.join(filtered_words)

    return processed_text

# Prétraitement des tweets
df['ProcessedTweet'] = df['Tweet'].apply(preprocess_text)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['ProcessedTweet'], df['Party'], test_size=0.2, random_state=42)

# Classification avec un classificateur SVM linéaire
vectorizer = TfidfVectorizer(min_df=5)  # Vous pouvez ajuster min_df selon vos besoins
X_train_vectorized = vectorizer.fit_transform(X_train)

classifier = LinearSVC(dual=False, max_iter=10000, C=1.0)  # Vous pouvez ajuster C selon vos besoins
classifier.fit(X_train_vectorized, y_train)

# Utiliser le CountVectorizer pour obtenir les mots originaux
feature_names = np.array(vectorizer.get_feature_names_out())


# Obtenir les indices des 5 premiers mots les plus fréquemment utilisés par les démocrates
democrat_indices = np.argsort(classifier.coef_[0])[-5:][::-1]
democrat_words = feature_names[democrat_indices]
print(f"Top 5 at used by Republicans: {democrat_words}")

# Obtenir les indices des 5 premiers mots les plus fréquemment utilisés par les républicains
republican_indices = np.argsort(classifier.coef_[0])[:5]
republican_words = feature_names[republican_indices]
print(f"Top 5 at used by Democrats: {republican_words}")

# Afficher les coefficients associés à ces mots
print("Coefficients for Republican at:", classifier.coef_[0, democrat_indices])
print("Coefficients for Democrats at:", classifier.coef_[0, republican_indices])