# import packages
import pandas as pd
import numpy as np
import matplotlib as plt
import string
import re
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('omw-1.4')

pd.set_option('display.max_colwidth', -1)

# functions to be used in the pipeline


def remove_punctuations(text):
    clean_text = "".join([i for i in text if i not in string.punctuation])
    return clean_text


def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# preprocessing functions


def preproc(dataset):
    # symbol removal
    dataset['tweet_text'] = dataset['tweet_text'].apply(
        lambda x: remove_punctuations(x))

    # change all to lowercase
    dataset['tweet_text'] = dataset['tweet_text'].apply(
        lambda x: x.lower())

    # tokenization
    dataset['tweet_text'] = dataset['tweet_text'].apply(
        lambda x: word_tokenize(x))

    # lemmatization
    dataset['tweet_text'] = dataset['tweet_text'].apply(
        lambda x: lemmatizer(x))

    return dataset


# dataset import
dataset = pd.read_csv(
    './dataset/cyberbullying_tweets.csv', encoding="ISO-8859-1")
cross_tab = dataset['cyberbullying_type'].value_counts()

# preprocessing the dataset
dataset = preproc(dataset)
print(dataset['tweet_text'].head(10))

# pipeline
# pipeline = Pipeline([])
