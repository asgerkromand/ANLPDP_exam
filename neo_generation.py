import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration

# load dev_set
dev_set = pd.read_csv('output/devset/dev_set_w_IR.csv')

# load the model and tokenizer
MODEL_NAME = "KennethTM/gpt-neo-1.3B-danish"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# set the device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # set up for mac here, change to cuda if needed
model.to(DEVICE)

neo_answers_tf_idf_k1 = []

# evaluating tf-idf
for question, documents in tqdm(zip(dev_set['question'], dev_set['tf_idf_k1']), desc='Answering questions'):

    # assemble a prompt from the documents, question and prompting an answer
    prompt = f"Relevante paragraffer: {documents} Spørgsmål: {question} Indsæt svar her baseret på de relevante paragraffer:"

    # tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    max_length = len(input_ids[0]) + 100

    # generate an answer within torch.no_grad() to save compute
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=7,
            # generation set to stop at '.' as it otherwise just repeats itself (think it's because we don't sample)
            eos_token_id=tokenizer.encode(' Spørgsmål')[0]
        )

    # decode generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip(' Spørgsmål')

    # append answer to list
    neo_answers_tf_idf_k1.append(answer[len(prompt):].strip())  # strip the prompt to leave just the answer

