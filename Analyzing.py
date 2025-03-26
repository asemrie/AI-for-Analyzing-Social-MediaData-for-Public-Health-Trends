import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import fasttext
# Load Amharic Dataset
data = pd.read_csv("amharic_health_dataset.csv")  # dataset
data = data.dropna()  # Remove missing values
# Preprocessing Function
def clean_amharic_text(text):
    text = re.sub(r'[^\u1200-\u137F\s]', '', text)  # Keep only Amharic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
data["cleaned_text"] = data["text"].apply(clean_amharic_text)
# Tokenization & Word Embedding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["cleaned_text"])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(data["cleaned_text"])
padded_sequences = pad_sequences(sequences, maxlen=100)
# Load Amharic Pretrained Word Embeddings (FastText)
amharic_embeddings = fasttext.load_model("cc.am.300.bin")  # Pretrained model
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in amharic_embeddings:
        embedding_matrix[i] = amharic_embeddings[word]
# Sentiment Analysis Model using LSTM
model = Sequential([
    Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=100, trainable=False),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(3, activation="softmax")  # 3 Classes: Positive, Neutral, Negative
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Train Sentiment Analysis Model
labels = data["sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})  # Encoding labels
model.fit(padded_sequences, labels, epochs=5, batch_size=32)
# Topic Modeling with LDA
vectorizer = CountVectorizer(max_features=5000)
text_matrix = vectorizer.fit_transform(data["cleaned_text"])
lda = LatentDirichletAllocation(n_components=3, random_state=42)
topics = lda.fit_transform(text_matrix)
# Visualizing Sentiment Trends
plt.figure(figsize=(10, 5))
sns.countplot(x=data["sentiment"], palette="coolwarm")
plt.title("Sentiment Distribution in Amharic Health Tweets")
plt.show()

# Word Cloud for Public Health Topics
all_text = " ".join(data["cleaned_text"])
wordcloud = WordCloud(font_path="NotoSansEthiopic-Regular.ttf", background_color="white").generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Amharic Public Health Discussions")
plt.show()
