import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
from nltk.corpus import brown
import numpy as np


model = gensim.models.Word2Vec(brown.sents())   #Entrainement du modèle Word2Vec sur le corpus Brown

model.save('brown.embedding')

model = gensim.models.Word2Vec.load('brown.embedding')

df = pd.read_csv('ExtractedTweets.csv')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def tokenstem(text):
    # Tokenisation
    words = word_tokenize(text.lower())
    
    # Stemming
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in ['http', 'amp', 'rt']]
    
    return words

# pretraitement (tokenisation + stemming)
df['ProcessedTweet'] = df['Tweet'].apply(lambda x: ' '.join(tokenstem(x)))

# embeddings, représentation vectorielle est obtenue en calculant la moyenne des embeddings des mots
def get_sentence_embedding(sentence):
    words = word_tokenize(sentence.lower())
    valid_words = [word for word in words if word.isalpha() and word not in stop_words and word not in ['http', 'amp', 'rt']]
    word_embeddings = [model.wv[word] for word in valid_words if word in model.wv]
    if not word_embeddings:
        return np.zeros(model.vector_size)
    return np.mean(word_embeddings, axis=0)

df['TweetEmbedding'] = df['ProcessedTweet'].apply(get_sentence_embedding)

#Création d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['TweetEmbedding'].to_numpy().tolist(), df['Party'], test_size=0.2, random_state=42)

# Classification avec un classificat SVM lineaire
classifier = LinearSVC(dual=False, max_iter=10000)
classifier.fit(X_train, y_train)

# EValuation des tests+ imprimé
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"précision des tests: {accuracy}")

# Entrer phrase à tester
input_text = "I'm gonna build a wall between Mexico and China and we want to continue using guns !"
input_text_processed = ' '.join(tokenstem(input_text))
input_embedding = get_sentence_embedding(input_text_processed)

prediction = classifier.predict([input_embedding])
print(f"le text est prédit comme étant de la partie {prediction[0]}")

