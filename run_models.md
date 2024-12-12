## How to run the inference scripts

### neo

Change directory to the folder:
```
cd [your_local_path/]src/model_inference
```

Generalised command:
```
python neo_generation.py --retriever [insert retriever name] --k_retrievals [1 to 3 retrieved documents]
```


### T5

Change directory to the folder:
```
cd [your_local_path/]src/model_inference
```

Generalised command:
```
python t5_generation.py --retriever [insert retriever name] --k_retrievals [1 to 3 retrieved documents]
```


Adam:

Before you run any other code, change directory to src/model_inference:

```
cd [your_local_path/]src/model_inference
```

Commands:
```
python neo_generation.py --retriever tfidf --k_retrievals 2 (done)
python neo_generation.py --retriever bm25 --k_retrievals 2 (done)
python neo_generation.py --retriever bert_cls --k_retrievals 2 (done)
python neo_generation.py --retriever bert_mean --k_retrievals 2 (done)
python neo_generation.py --retriever bert_max --k_retrievals 2 (done)
python t5_generation.py --retriever tfidf --k_retrievals 2 (done)
python t5_generation.py --retriever bm25 --k_retrievals 2 (done)
python t5_generation.py --retriever bert_cls --k_retrievals 2 (done)
python t5_generation.py --retriever bert_mean --k_retrievals 2 (done)
python t5_generation.py --retriever bert_max --k_retrievals 2 (done)
```

Andreas:

Before you run any other code, change directory to src/model_inference:
```
cd [your_local_path/]src/model_inference
```

Commands:
```
python neo_generation.py --retriever tfidf --k_retrievals 3
python neo_generation.py --retriever bm25 --k_retrievals 3
python neo_generation.py --retriever bert_cls --k_retrievals 3
python neo_generation.py --retriever bert_mean --k_retrievals 3
python neo_generation.py --retriever bert_max --k_retrievals 3
python t5_generation.py --retriever tfidf --k_retrievals 3
python t5_generation.py --retriever bm25 --k_retrievals 3
python t5_generation.py --retriever bert_cls --k_retrievals 3
python t5_generation.py --retriever bert_mean --k_retrievals 3
python t5_generation.py --retriever bert_max --k_retrievals 3
```

Asger:

Before you run any other code, change directory to src/model_inference:
```
cd [your_local_path/]src/model_inference
```

Commands:
```
python neo_generation.py --retriever tfidf --k_retrievals 1
python neo_generation.py --retriever bm25 --k_retrievals 1
python neo_generation.py --retriever bert_cls --k_retrievals 1
python neo_generation.py --retriever bert_mean --k_retrievals 1
python neo_generation.py --retriever bert_max --k_retrievals 1
python t5_generation.py --retriever tfidf --k_retrievals 1
python t5_generation.py --retriever bm25 --k_retrievals 1
python t5_generation.py --retriever bert_cls --k_retrievals 1
python t5_generation.py --retriever bert_mean --k_retrievals 1
python t5_generation.py --retriever bert_max --k_retrievals 1
```

