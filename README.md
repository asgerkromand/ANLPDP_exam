# Exam Project: RAG in Danish Legal Q/A - Testing with GPT and T5

This repo contains the source code (scripts and code) for the authors exam project in the course ''Advanced Natural Language Processing and Deep Learning (Autumn 2024)''.

The project examines tests the performance of Retrieval Augmented Generation (RAG) on Danish legal data by using two Danish Language Models and comparing their results. The two models are:

- ```strombergnlp/dant5-large``` (T5 model)
and
- ```KennethTM/gpt-neo-1.3B-danish``` (GPT model)

The data is retrieved from the The Danish Legal Information Portal (retsinformation.dk). The dataset contains all current laws dating back to XXXX (***indsæt år***). It was provided to the project in a ```.json```format.

## Getting Started

To replicate this study, you are more than welcome to make fork.

### Prerequisites

Of particular note this source code relies on the following dependencies:

- pytorch ([github link](https://github.com/pytorch/pytorch)).
- NLTK (Natural Language Toolkit, [github link](https://github.com/nltk/nltk)).
- sklearn (TF-IDF vectorizer, ([github link](https://github.com/scikit-learn/scikit-learn)).
- rank_bm25 (BM25 model, [github link](https://github.com/dorianbrown/rank_bm25)).
- Transformers ([github link](https://github.com/huggingface/transformers)).

### Installation

No installation needed. Please create a fork to be able to run the code.

## Data use in this exam

***Skriv noget om hvordan vi bruger data, skriv noget om det data vi har genereret.***

## Usage

To reproduce this study the src codes has been provided which consists of a mix of scripts and code. Below we will provide examples of all the python commandline prompts which are needed to be able to reproduce the coding part of our exam study.

The codebase can be divided into three parts. Info on initial input data will be provided first.

1) Data generation
2) Model inference
3) Performance evaluation

### Data generation

Code:

- ```data_functions.py```
  - Functions to load in data and generate a list of paragraphs.
  - Functions to preprocess and tokenize paragraphs, and afterwards generate the sparse and dense matrices for the information retrieval (IR).
- ```generation_embeddings.py```
  - Script with to generate and save BERT embeddings with different pooling (*CLS*, *Max-pooling*, *Mean-pooling*).
  - Input: List of context documents, e.g. paragraphs.
  - Output: BERT embeddings with three different poolings.
- ```info_retrieval.py```
  - Script to perform information retrieval (IR) on the paragraphs based on sparse and dense retrieval:
    - *Sparse:* TF-IDF, BM25.
    - *Dense:* BERT CLS-pooling, BERT Max-pooling, BERT Mean-pooling.
  - Input: data/dev_set.csv, domsdatabasen.retsinformation_newer.json
  - Output: output/devset/devset_with_contexts_parquet.gzip

Running the code:

### Model inference

Code:

- ```neo_generation.py```
  - Script to generate answers to legal questions with the ```GPT-Neo Danish```-model aided by RAG done by the five different IR-systems.
- ```neo_generation_baseline.py```
  - Script to generate answers to legal questions with the ```GPT-Neo Danish```-model without RAG.
- ```t5_generation_baseline.py```
  - Script to generate answers to legal questions with the ```DanT5 Large```-model aided by RAG done by the five different IR-systems.
- ```t5_generation_baseline.py```
  - Script to generate the answers to legal questions with the ```DanT5 Large```-model without RAG.

Input (all scripts): Development dataset with retrieved paragrapghs aka. concext (output/devset_with_contexts.parquet.gzip)
Output (all scripts): Inferred outputs for the models T5 and GPT with variyng configurations.

Running the code:

### Performance evaluation

Code:

- ```eval_functions.py```
  - Functions to calculate and plotting the plot.
- ```eval.py```
  - Script to compute and produce the performance results and plots.
  - Input: data/dev_set.csv
  - Output: output/plots

Running the code:

## Additional Documentation and Acknowledgments

Tak til ham manden der gav os data og vejledning fra Rob. Tak til Copilot.
