import pandas as pd
from transformers import (
    AutoModel,
    AutoTokenizer, 
)

from data_functions import *
import argparse
import os

def main():
    # Load in filepaths
    parser = argparse.ArgumentParser()
    parser.add_argument("devset_path", type=str, help="Path to the devset file") # data/dev_set.csv
    parser.add_argument("paragraphs_path", type=str, help="Path to the paragraphs file") # data/domsdatabasen.retsinformation_newer.json
    parser.add_argument("output_folder", type=str, help="Path to the output file") # output/devset
    args = parser.parse_args()

    # Assign the filepaths to variables
    devset_path = args.devset_path
    paragraphs_path = args.paragraphs_path
    output_folder = args.output_folder
    # Check if the output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"Output folder '{output_folder}' exists")

    # load the devset for evaluation
    devset = pd.read_csv(devset_path).astype(str)

    # generate the RAG list, i.e. the legal documents to be used for retrieval
    paragraphs = generate_paragraphs(paragraphs_path)
    print(f"Paragraphs generated. Number of paragraphs: {len(paragraphs)}")

    # TF-IDF - get the vectorizer and tfidf matrix, then get context for each question (top 3 paragraphs)
    vectorizer, tfidf_matrix = tfidf_vectorizer(paragraphs)
    devset['tfidf_context'] = devset['question'].apply(lambda x: tfidf_retrieval(x, tfidf_matrix, paragraphs, vectorizer))
    print("TF-IDF retrieval done")

    # BM25 - get the bm25 model and the corpus, then get context for each question (top 3 paragraphs)
    bm25_model, corpus = bm25_vectorizer(paragraphs)
    devset['bm25_context'] = devset['question'].apply(lambda x: bm25_retrieval(x, bm25_model, corpus))
    print("BM25 retrieval done")

    # load the bert tokenizer and model
    bert_tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")
    bert_model = AutoModel.from_pretrained("vesteinn/DanskBERT")

    # load embeddings
    cls_embeddings, max_embeddings, mean_embeddings = load_embeddings()

    # get context for each question (top 3 paragraphs)
    devset['bert_cls_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                    embeddings=cls_embeddings, 
                                                                                    corpus=paragraphs, 
                                                                                    tokenizer=bert_tokenizer, 
                                                                                    model=bert_model, 
                                                                                    pooling='cls'))
    print("BERT CLS retrieval done")

    devset['bert_max_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                    embeddings=max_embeddings, 
                                                                                    corpus=paragraphs, 
                                                                                    tokenizer=bert_tokenizer, 
                                                                                    model=bert_model, 
                                                                                    pooling='max'))
    print("BERT Max retrieval done")

    devset['bert_mean_context'] = devset['question'].apply(lambda x: dense_retrieval(question=x, 
                                                                                     embeddings=mean_embeddings, 
                                                                                     corpus=paragraphs, 
                                                                                     tokenizer=bert_tokenizer, 
                                                                                     model=bert_model, 
                                                                                     pooling='mean'))
    print("BERT Mean retrieval done")

    # Write to Parquet with gzip compression
    output_folder = args.output_folder
    file_path = f"{output_folder}/devset_with_contexts.parquet.gzip"

    with open(file_path, "wb") as f:
        devset.to_parquet(file_path, compression="gzip", engine='pyarrow')

    # Assert if file is made
    if os.path.exists(file_path):
        print(f"File saved at: {file_path}")
    else:
        print("File not saved")

if __name__ == "__main__":
    main()