#!/bin/bash

# Change directory to src/model_inference
cd src/model_inference || exit

# Run neo_generation.py for all retrievers
for retriever in tfidf bm25 bert_cls bert_mean bert_max; do
    echo "Running neo_generation.py with retriever: $retriever"
    python neo_generation.py --retriever "$retriever" --k_retrievals 2
done

# Run t5_generation.py for all retrievers
for retriever in tfidf bm25 bert_cls bert_mean bert_max; do
    echo "Running t5_generation.py with retriever: $retriever"
    python t5_generation.py --retriever "$retriever" --k_retrievals 2
done