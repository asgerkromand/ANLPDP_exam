# Make a class for BM25 IR model
# Import necessary libraries
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import json
import pandas as pd
import string
import numpy as np

def load_csv(file_path):
    # Load the csv file
    return pd.read_csv(file_path)

def preprocess(text):
    # extract and preprocess text8
    stop_words = set(stopwords.words('danish'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

def init_bm25_corpus(corpus):
    # Initialize the BM25 corpus
    return BM25Okapi(corpus)

def get_ranked_scores(bm25_corpus, query):
    # Get scores
    scores = bm25_corpus.get_scores(query)
    # Rank the scores
    return np.argsort(scores)[::-1]

