# Exam Project: RAG in Danish Legal Q/A - Testing with GPT and T5

This repo contains the source code (scripts and code) for the authors exam project in the course ''Advanced Natural Language Processing and Deep Learning (Autumn 2024)'' offered at IT University of Copenhagen (ITU).

The project examines tests the performance of Retrieval Augmented Generation (RAG) on Danish legal data by using two Danish Language Models and comparing their results. The two models are:

- ```strombergnlp/dant5-large``` (T5 model)
and
- ```KennethTM/gpt-neo-1.3B-danish``` (GPT model)

The data is retrieved from The Danish Legal Information Portal (retsinformation.dk). The dataset contains all current laws dating back to 1840. It was provided to the project in a ```.json```format.

## Getting Started

To replicate this study, you are more invited to make fork.

### Prerequisites

Among others this source code relies on the following dependencies:

- pytorch ([github link](https://github.com/pytorch/pytorch)).
- NLTK (Natural Language Toolkit, [github link](https://github.com/nltk/nltk)).
- sklearn (TF-IDF vectorizer, [github link](https://github.com/scikit-learn/scikit-learn)).
- rank_bm25 (BM25 model, [github link](https://github.com/dorianbrown/rank_bm25)).
- Transformers ([github link](https://github.com/huggingface/transformers)).

### Installation

No installation needed. Create a fork to be able to run the code.

## Data

For the task of legal question answering, we have obtained a dataset of all applicable Danish laws with the oldest dating back to 1865 from Mads Henrichsen (founder of [Dansk GPT](https://www.danskgpt.dk/)). These 1637 laws have been collected from The Danish Legal Information Portal (https://www.retsinformation.dk/). Each law is structured such that it consists of a number of chapters, which in turn contain the related paragraphs with subsections. In this project, our level of analysis is on the paragraphs, and our corpus consists of a list of $42,593$ paragraphs with a mean length of $\approx656$ characters.

Only a test subset have been provided in this repo to be able to run the code due to storage constraints. Please write an email to [Adam Wagner Hoegh](mailto:wagnerhoegh.adam@gmail.com) to obtain the full dataset.

## Usage

To reproduce this study the src codes has been provided which consists of a mix of scripts and code. Below we will provide examples of all the python commandline prompts which are needed to be able to reproduce the coding part of our exam study.

The codebase can be divided into three parts:

1) Data generation
2) Model inference
3) Performance evaluation

### 1 Data generation

#### Code

- ```data_functions.py```
  - Functions to load in data and generate a list of paragraphs.
  - Functions to preprocess and tokenize paragraphs, and afterwards generate the sparse and dense matrices for the information retrieval (IR).
- ```generation_embeddings.py```
  - Script with to generate and save BERT embeddings with different pooling (*CLS*, *Max-pooling*, *Mean-pooling*).
  - Input: Path to Law Data in .json-format, e.g. ```domsdatabasen.retsinformation_newer.json```.
  - Output: BERT embeddings with three different poolings.
- ```info_retrieval.py```
  - Script to perform information retrieval (IR) on the paragraphs based on sparse and dense retrieval:
    - *Sparse:* TF-IDF, BM25.
    - *Dense:* BERT CLS-pooling, BERT Max-pooling, BERT Mean-pooling.
  - Input: ```data/dev_set.csv```, ```domsdatabasen.retsinformation_newer.json```
  - Output: ```output/devset/devset_with_contexts_parquet.gzip```

#### Run in terminal

**Generate embeddings with CLS pooling method:**

```[python]
python generate_embeddings.py <input_filepath> <output_filepath> [--pooling_method <pooling methods> (default: CLS)] [--model_name <model_name> (default: vesteinn/DanskBERT)] [--batch_size <batch_size> (default: 32)]
```

*Example*:

```[python]
python generate_embeddings.py ../../data/retsinformation_subset.jsonl ../../output/embeddings_DanishBERT.pt --pooling_method {pooling method} --model_name vesteinn/DanskBERT --batch_size 32
```

**Perform information retrieval (IR):**

```[python]
python info_retrieval.py <devset_path> <paragraphs_path> <output_folder>
```

*Example*:

```[python]
python info_retrieval.py ../../data/dev_set.csv law_data_path ../../output/devset/devset_with_contexts.parquet.gzip
```

### 2 Model inference

#### Code

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

#### Run in terminal

Generate the answers based on NeoGPT and T5. To run the code locally, this will require approx 7-8 hours depending in your computer. To run all models with all configurations as in this project, you can run below shell script, by changing the path. Beware that the code was run on Mac M2 and M3 with 16MB ram.

Change wd to the folder with below shell file, and use these two commandline code in terminal to run the shell file: 

```[terminal]
chmod +x run_models.sh
```

```[terminal]
./run_models.sh
```

### 3 Performance evaluation

*Code*:

- ```eval_functions.py```
  - Functions to calculate and plotting the plot.
- ```eval.py```
  - Script to compute and produce the performance results and plots.
  - Input: data/dev_set.csv
  - Output: output/plots

#### Run in terminal

**Evaluating the models with different configurations:**

```[python]
python eval.py [--gold_answers <gold_answers> (default: ../../data/dev_set.csv)] [--inference-dir <inference-dir> (default: ../../output/inference)] <comparison-plot> <save-results> <metrics> [--titles <titles>] [--retrieve_order] [--retrievers]
```

*Example:*

```
python eval.py --comparison-plot results.svg --save-results results.tex --metrics BLEU ROUGE-L METEOR --retriever-order "tfidf" "bm25" "bert_cls" --retrievers "tfidf" "bm25" "bert_cls"
```

## Acknowledgments

Our gratitude goes to Mads Henrichsen ([LinkedIn](https://www.linkedin.com/in/mhenrichsen/)) for providing us with a dataset of Danish Laws from [retsinformation.dk](retsinformation.dk). We also want to thank Rob van der Goot ([website](https://robvanderg.github.io/)) for valuable guidance and feedback.
