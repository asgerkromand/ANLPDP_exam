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

- pytorch ([github link](https://github.com/pytorch/pytorch))
- NLTK (Natural Language Toolkit, [github link](https://github.com/nltk/nltk))
- sklearn (TF-IDF vectorizer, ([github link](https://github.com/scikit-learn/scikit-learn))
- rank_bm25 (BM25 model, [github link](https://github.com/dorianbrown/rank_bm25))
- Transformers ([github link](https://github.com/huggingface/transformers)) 

### Installation

No installation needed. Please create a fork to be able to run the code.

## Usage

To reproduce this study the src codes has been provided which consists of a mix of scripts and code. Below we will provide examples of all the python commandline prompts which are needed to be able to reproduce the coding part of our exam study.

The codebase can be divided into three parts. Info on initial input data will be provided first.

1) Data generation
2) Model inference
3) Performance evaluation

### Data generation

Code:

- ```data_functions.py```
  - Functions to generate the sparse and dense matrices for the information retrieval.
- ```generation_embeddings.py```
  - Functions to generate and save BERT embeddings with different pooling (*CLS*, *Max-pooling*, *Mean-pooling*)
- ```ìnfo_retrieval.py```
  - fdjkjkl

## Additional Documentation and Acknowledgments

* Project folder on server:
* Confluence link:
* Asana board:
* etc...
