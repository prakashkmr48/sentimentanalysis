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

# Evaluate the classifier on the test set
accuracy_score = accuracy(classifier, test_set)
print(f'Accuracy: {accuracy_score:.2%}')

# Classify new tweets
new_tweet = "I love this product! It's amazing!"
features = extract_features(new_tweet)
sentiment = classifier.classify(features)
print(f'Sentiment: {sentiment}')
