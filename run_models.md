## How to run the inference scripts

### neo

Change directory to the folder:
$ cd src/model_inference
$ python neo_generation.py --retriever [insert retriever name] --k_retrievals [1 to 3 retrieved documents]



### T5

Change directory to the folder:
$ cd src/model_inference
$ python t5_generation.py --retriever [insert retriever name] --k_retrievals [1 to 3 retrieved documents]


Adam:
neo tfidf, bm25, bert_cls


Andreas:
neo bert_max, bert_mean

baseline models for both (k=0)


Asger: all T5 combinations