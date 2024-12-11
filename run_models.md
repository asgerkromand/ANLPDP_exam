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
all k=2

Andreas:
all k=3

Asger:
all k=1

