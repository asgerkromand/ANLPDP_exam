import numpy as np
import pandas as pd
import json
import os
import random
import regex as re

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

import torch
from transformers import (
    AutoModel,
    AutoTokenizer, 
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    pipeline
)

from utils.data_functions import *


# load the devset for evaluation
devset = pd.read_csv("data/dev_set.csv").astype(str)

# generate the RAG list, i.e. the legal documents to be used for retrieval
rag_list = generate_rag_list("data/domsdatabasen.retsinformation_newer.json", "data/rag_list.csv")

# TF-IDF - get the vectorizer and tfidf matrix, then get context for each question (top 3 paragraphs)
vectorizer, tfidf_matrix = tfidf_vectorizer(rag_list)
devset['tfidf_context'] = devset['question'].apply(lambda x: tfidf_retrieval(x, tfidf_matrix, rag_list, vectorizer))

# BM25 - get the bm25 model and the corpus, then get context for each question (top 3 paragraphs)
bm25_model, corpus = bm25_vectorizer(rag_list)
devset['bm25_context'] = devset['question'].apply(lambda x: bm25_retrieval(x, bm25_model, corpus))

# load the bert tokenizer and model
bert_tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")
bert_model = AutoModel.from_pretrained("vesteinn/DanskBERT")

# load embeddings
cls_embeddings, max_embeddings, mean_embeddings = load_embeddings()

# get context for each question (top 3 paragraphs)
devset['bert_cls_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                embeddings=cls_embeddings, 
                                                                                corpus=rag_list, 
                                                                                tokenizer=bert_tokenizer, 
                                                                                model=bert_model, 
                                                                                pooling='cls'))

devset['bert_max_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                embeddings=max_embeddings, 
                                                                                corpus=rag_list, 
                                                                                tokenizer=bert_tokenizer, 
                                                                                model=bert_model, 
                                                                                pooling='max'))

devset['bert_mean_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                 embeddings=mean_embeddings, 
                                                                                 corpus=rag_list, 
                                                                                 tokenizer=bert_tokenizer, 
                                                                                 model=bert_model, 
                                                                                 pooling='mean'))

# save the devset with the contexts
devset.to_csv("data/devset_with_contexts.csv", index=False)