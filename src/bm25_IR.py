# Make a class for BM25 IR model
# Import necessary libraries
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import json
import pandas as pd

def load_csv(file_path):
    # Load the csv file
    return pd.read_csv(file_path)

def preprocess(list):
    # extract and preprocess text
    corpus = list
    corpus = [re.sub('\\s{2,}', ' ', 
                     re.sub('\\W|[0-9]', ' ',
                           item.lower())) for item in corpus]

    # remove stopwords
    #nltk.download('punkt')
    stop_words = set(stopwords.words('danish'))
    corpus = [' '.join(word for word in text.split() 
                      if word not in stop_words) for text in tqdm(corpus)]
    
    return corpus

def get_bm25_score(query, bm25):
    # Get the BM25 score for the query
    return bm25.get_scores(query)

def get_top_k(query, bm25, corpus, k=5):
    # Get the top k documents for the query
    scores = get_bm25_score(query, bm25, corpus)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return top_k

# Define __main__ function
if __name__ == '__main__':
    # Load dev/dev_set.csv file.
    dev = pd.read_csv('../devset/dev_set.csv')

    # Get the question and answer data into lists. Column names are: 'question, str', 'answers, str'
    questions = [q for q in dev['question, str']]
    answers = [a for a in dev['answer, str']]
    text = [t for t in dev['text, str']]

    # Preprocess question and answer data
    questions = preprocess(questions)
    answers = preprocess(answers)
    text = preprocess(text)

    # Use BM25 to get the top k documents for the query
    bm25 = BM25Okapi(answers)
    query = questions[0]
    top_k = get_top_k(query, bm25, answers, k=5)
    print(top_k)

    # Print the top k documents
    for i in top_k:
        print(answers[i])
        print('---')

