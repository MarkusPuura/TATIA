import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import numpy as np

# Charger le modèle Word2Vec pré-entraîné sur le corpus Brown
model = gensim.models.Word2Vec.load('brown.embedding')

# Charger les données
df = pd.read_csv('ExtractedTweets.csv')

# Définir les stop words et le stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Fonction de tokenisation et stemming
def token_stem(text):
    words = word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in ['http', 'amp', 'rt']]
    return words

# Appliquer le prétraitement aux tweets
df['ProcessedTweet'] = df['Tweet'].apply(lambda x: ' '.join(token_stem(x)))

# Obtenir les embeddings des phrases
def get_sentence_embedding(sentence):
    words = word_tokenize(sentence.lower())
    valid_words = [word for word in words if word.isalpha() and word not in stop_words and word not in ['http', 'amp', 'rt']]
    word_embeddings = [model.wv[word] for word in valid_words if word in model.wv]
    if not word_embeddings:
        return np.zeros(model.vector_size)
    return np.mean(word_embeddings, axis=0)

df['TweetEmbedding'] = df['ProcessedTweet'].apply(get_sentence_embedding)

# Mapping des labels à des valeurs numériques
label_encoder = LabelEncoder()
df['PartyEncoded'] = label_encoder.fit_transform(df['Party'])

# Création des ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['TweetEmbedding'].to_numpy().tolist(), df['PartyEncoded'], test_size=0.2, random_state=42)

# Classification avec XGBoost
classifier = XGBClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Évaluation sur l'ensemble de test
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test: {accuracy}")

# Entrer une phrase à tester
input_text = "This includes relief for TX, LA, FL, the U.S. Virgin Islands and Puerto Rico, as well as CA as they work to suppress deadly wildfires."
input_text_processed = ' '.join(token_stem(input_text))
input_embedding = get_sentence_embedding(input_text_processed)

# Prédiction
prediction = classifier.predict([input_embedding])
print(f"Le texte est prédit comme étant de la partie {label_encoder.inverse_transform(prediction)[0]}")
