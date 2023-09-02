import streamlit as st
import nltk
import random
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download the NLTK Twitter dataset
nltk.download('twitter_samples')
nltk.download('punkt')

# Load positive and negative tweets from the Twitter dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Define a function to extract features from text (BoW)
def extract_features(tweet):
    words = word_tokenize(tweet)
    return dict([(word, True) for word in words])

# Prepare labeled data with positive and negative tweets
positive_features = [(extract_features(tweet), 'positive') for tweet in positive_tweets]
negative_features = [(extract_features(tweet), 'negative') for tweet in negative_tweets]

# Shuffle and split the data into training and testing sets
random.shuffle(positive_features)
random.shuffle(negative_features)

split_ratio = 0.8
positive_split = int(len(positive_features) * split_ratio)
negative_split = int(len(negative_features) * split_ratio)

train_set = positive_features[:positive_split] + negative_features[:negative_split]
test_set = positive_features[positive_split:] + negative_features[negative_split:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a tweet and click the 'Analyze' button to determine its sentiment.")

# User Input
user_input = st.text_area("Enter a tweet:")

# Sentiment Analysis
if st.button("Analyze"):
    features = extract_features(user_input)
    sentiment = classifier.classify(features)

    if sentiment == 'positive':
        sentiment_label = "Positive"
    else:
        sentiment_label = "Negative"

    st.write(f"Sentiment: {sentiment_label}")
