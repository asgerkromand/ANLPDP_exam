# Import
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import pandas as pd
import re
import numpy as np

# Functions
# Load the csv file
def load_csv(file_path):
    # Load the csv file
    return pd.read_csv(file_path)

# Preprocess the text
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-word characters, digits, and the section symbol
    text = re.sub('\\W|[0-9]|§', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub('\\s{2,}', ' ', text)

    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('danish'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Initialize the BM25 corpus
def init_bm25_corpus(corpus):
    # Initialize the 