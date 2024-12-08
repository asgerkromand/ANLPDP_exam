# Import
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import pandas as pd
import string
import numpy as np

# Functions
# Load the csv file
def load_csv(file_path):
    # Load the csv file
    return pd.read_csv(file_path)

# Preprocess the text
def preprocess(text):
    # extract and preprocess text8
    stop_words = set(stopwords.words('danish'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

# Initialize the BM25 corpus
def init_bm25_corpus(corpus):
    # Initialize the BM25 corpus
    return BM25Okapi(corpus)

# Get the ranked scores
def get_ranked_scores(bm25_corpus, query, n_ranked = None):
    # Get scores
    scores = bm25_corpus.get_scores(query)
    # Rank the scores
    if n_ranked is not None:
        return np.argsort(scores)[::-1][:n_ranked]
    else:
        return np.argsort(scores)[::-1]

