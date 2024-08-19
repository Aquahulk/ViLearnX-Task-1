from google.colab import files
uploaded = files.upload()
import pandas as pd
import re
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
df = pd.read_csv('tweets.csv')
nlp = spacy.load('en_core_web_sm')
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)
df['cleaned_text'] = df['Text'].apply(preprocess_text)
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
plt.figure(figsize=(10, 6))
df.set_index('Timestamp').resample('D')['sentiment'].value_counts().unstack().plot(kind='line')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()
