import pandas as pd
import numpy as np
import json
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import torch

def replace_nbsp(obj):
    """
    Recursively replaces non-breaking space characters (\xa0) with regular spaces
    in strings within nested data structures.
    
    Args:
        obj: Input object which can be a dictionary, list, string or other type
        
    Returns:
        Object of same structure as input but with \xa0 replaced with spaces in all strings
    """
    if isinstance(obj, dict):
        return {k: replace_nbsp(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nbsp(i) for i in obj]
    elif isinstance(obj, str):
        return obj.replace('\xa0', ' ')
    else:
        return obj

def generate_rag_list(input_file, output_file):
    """
    Generates a list of dictionaries containing paragraph information from legal documents.
    
    Args:
        input_file: Path to input JSON file containing legal document data
        output_file: Path to output file to write the processed data
        
    Returns:
        rag_list: List of dictionaries, where each dictionary contains:
            - paragraf_nr: Paragraph number
            - lovnavn: Name of the law
            - text: Combined text of all subsections in the paragraph
    """
    with open(input_file) as f:
        retsinfo = json.load(f)

    rag_list = []
    idx = 0
    for lov in retsinfo:
        for kapitel in lov['kapitler']:
            lov_navn = lov['shortName']
            for paragraffer in kapitel['paragraffer']:
                temp_paragraf_dict = {}
                temp_paragraf_dict['paragraf_nr'] = paragraffer['nummer']
                temp_paragraf_dict['lovnavn'] = lov_navn
                temp_paragraf_list = []
                for styk in paragraffer['stk']:
                    temp_paragraf_list.append(styk['tekst'])
                temp_paragraf_dict['text'] = ' '.join(temp_paragraf_list)
                rag_list.append(temp_paragraf_dict)

    rag_list = replace_nbsp(rag_list)

    with open(output_file, "w") as file:
        for item in rag_list:
            file.write(f"{item}\n")
    return rag_list

def generate_paragraphs(input_file):
    """
    Generates a list of dictionaries containing paragraph information from legal documents.
    
    Args:
        input_file: Path to input JSON file containing legal document data
        output_file: Path to output file to write the processed data
        
    Returns:
        rag_list: List of dictionaries, where each dictionary contains:
            - paragraf_nr: Paragraph number
            - lovnavn: Name of the law
            - text: Combined text of all subsections in the paragraph
    """
    with open(input_file) as f:
        retsinfo = json.load(f)

    paragraphs = []
    idx = 0
    for lov in retsinfo:
        for kapitel in lov['kapitler']:
            lov_navn = lov['shortName']
            for paragraffer in kapitel['paragraffer']:
                temp_paragraf_dict = {}
                temp_paragraf_dict['paragraf_nr'] = paragraffer['nummer']
                temp_paragraf_dict['lovnavn'] = lov_navn
                temp_paragraf_list = []
                for styk in paragraffer['stk']:
                    temp_paragraf_list.append(styk['tekst'])
                temp_paragraf_dict['text'] = ' '.join(temp_paragraf_list)
                paragraphs.append(temp_paragraf_dict)

    paragraphs = replace_nbsp(paragraphs)
    return paragraphs


def tfidf_vectorizer(rag_list):
    """
    Preprocess and vectorize a corpus of text using TF-IDF
    
    Args:
        rag_list: List of dictionaries containing text to vectorize
        
    Returns:
        vectorizer: Fitted TfidfVectorizer
        X: Sparse matrix of TF-IDF features
    """
    # extract and preprocess text
    corpus = [item['text'] for item in rag_list]
    corpus = [re.sub('\\s{2,}', ' ', 
                     re.sub('\\W|[0-9]|§', ' ',
                           item.lower())) for item in corpus]

    # remove stopwords
    #nltk.download('punkt')
    stop_words = set(stopwords.words('danish'))
    corpus = [' '.join(word for word in text.split() 
                      if word not in stop_words) for text in corpus]
    
    # vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def tfidf_retrieval(question, corpus_embeddings, corpus, vectorizer, k=3, max_tokens=800):
    """
    Function that takes a question and returns a list of paragraphs that are most relevant to the question
    
    Returns:
        List of k strings containing the most relevant paragraphs
    """

    # preprocess and vectorize question
    question_processed = [re.sub('\\s{2,}', ' ', 
                               re.sub('\\W|[0-9]|§', ' ',
                                     question.lower()))]
    
    # remove stopwords
    stop_words = set(stopwords.words('danish'))
    question_processed = [' '.join(word for word in text.split() 
                                 if word not in stop_words) for text in question_processed]
    
    # embed question
    question_vector = vectorizer.transform(question_processed)

    # calculate cosine similarity
    sparse_retrieval = corpus_embeddings.dot(question_vector.T).toarray()

    # get top k paragraphs
    top_k = np.argsort(sparse_retrieval.flatten())[-k:]

    # get k most relevant paragraphs as list
    context = [corpus[i]['text'] for i in reversed(top_k)]

    return context

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-word characters, digits, and the section symbol
    text = re.sub('\\W|[0-9]|§', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub('\\s{2,}', ' ', text)

    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('danish'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def bm25_vectorizer(rag_list):
    """
    Creates and returns a BM25 model fitted on the corpus
    
    Args:
        rag_list: List of dictionaries containing text data
        
    Returns:
        bm25_model: Fitted BM25Okapi model
        corpus: List of raw text documents
    """
    corpus = [elem['text'] for elem in rag_list]
    
    # Preprocess the data
    tokenized_corpus = [preprocess(text) for text in corpus]
    
    # Create and fit the BM25 model
    bm25_model = BM25Okapi(tokenized_corpus)
    
    return bm25_model, corpus

def bm25_retrieval(question, bm25_model, corpus, k=3):
    """
    Returns k most relevant paragraphs for a given question using BM25
    
    Args:
        question: Question text string
        bm25_model: Fitted BM25Okapi model
        corpus: List of raw text documents
        k: Number of paragraphs to return (default 3)
        
    Returns:
        context: List of k most relevant paragraphs as strings
    """
    # Preprocess the question
    tokenized_question = preprocess(question)
    
    # Get top k most relevant paragraphs
    top_k = bm25_model.get_top_n(tokenized_question, corpus, n=k)
    
    return top_k


def load_embeddings():
    """
    Load pre-computed BERT embeddings from disk
    
    Returns:
        cls_embeddings: CLS token embeddings
        max_embeddings: Max pooled embeddings  
        mean_embeddings: Mean pooled embeddings
    """
    cls_embeddings = torch.load('output/embeddings/cls_embeddings_DanskBERT.pt')
    max_embeddings = torch.load('output/embeddings/max_embeddings_DanskBERT.pt')
    mean_embeddings = torch.load('output/embeddings/mean_embeddings_DanskBERT.pt')
    return cls_embeddings, max_embeddings, mean_embeddings

def dense_retrieval(question, embeddings, corpus, tokenizer, model, pooling='cls', k=3):
    """
    Function that takes a question and returns a list of paragraphs that are most relevant to the question
    pooling = 'cls', 'max' or 'mean'
    """
    
    # encode the input
    input_ids = tokenizer.encode(question, return_tensors="pt")

    # pass the input through the model
    with torch.no_grad():  # no backprop needed
        outputs = model(input_ids)
    
    if pooling == 'cls':
        embedding_vector = outputs.last_hidden_state[:, 0, :]
    
    elif pooling == 'max':
        embedding_vector = torch.max(outputs.last_hidden_state, dim=1)[0]

    elif pooling == 'mean':
        embedding_vector = torch.mean(outputs.last_hidden_state, dim=1)
    
    # normalise embeddings (to get cosine similarity from dot product)
    embedding_vector_normalised = embedding_vector / torch.norm(embedding_vector, dim=-1, keepdim=True)
    embeddings_matrix_normalised = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)

    # get paragraphs with highest similarity scores
    dense_retrieval = embeddings_matrix_normalised @ torch.transpose(embedding_vector_normalised, 0, 1)
    sorted_retrieval = torch.sort(dense_retrieval, descending=True, stable=True, dim=0)
    fixed_retrieval_list = [(item, idx) for (item, idx) in zip(sorted_retrieval[0], sorted_retrieval[1]) if torch.isnan(item) == False]
    top_k_indices = [item[1] for item in fixed_retrieval_list[:k]]
    document = '\n'.join([corpus[i]['text'] for i in top_k_indices])

    return document