import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration
import ast
import sys


k_retrievals = # find ud af hvordan du kan linke det her til argparse

# load dev_set
dev_set = pd.read_csv('output/devset/dev_set_w_IR.csv')

# converting list columns back to list columns as they are loaded as strings
columns_to_convert = ['tf_idf', 'bm25', 'dense_cls', 'dense_max', 'dense_mean']

for col in columns_to_convert:
    dev_set[col] = dev_set[col].apply(ast.literal_eval)


# load the model and tokenizer
MODEL_NAME = "KennethTM/gpt-neo-1.3B-danish"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# set the device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # set up for mac here, change to cuda if needed
model.to(DEVICE)

def generate_answers(retriever, k_retrievals, output_path):

    if retriever == 'dense_cls':
         embeddings_matrix = torch.load('output/embeddings/cls_embeddings_DanskBERT.pt')
    elif retriever == 'dense_max':
         embeddings_matrix = torch.load('output/embeddings/max_embeddings_DanskBERT.pt')
    elif retriever == 'dense_mean':
         embeddings_matrix = torch.load('output/embeddings/mean_embeddings_DanskBERT.pt')
    else:
         pass

    neo_answers = []

    # evaluating tf-idf
    for question, retrieval_column in tqdm(zip(dev_set['question'], dev_set[retriever]), desc='Answering questions'):

        # assemble documents into a single string with newlines between each paragraph
        documents = '\n'.join([item for item in retrieval_column][:k_retrievals])

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
                # generation set to stop at '.' as it otherwise just repeats itself (think it's because we don't sample)
                eos_token_id=tokenizer.encode(' Spørgsmål')[0]
            )

        # decode generated answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip(' Spørgsmål')

        # append answer to list
        neo_answers.append(answer[len(prompt):].strip())  # strip the prompt to leave just the answer

    with open(f'{output_path}/neo_gen_{retriever}_k{k_retrievals}', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in neo_answers))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Set the 'KennethTM/gpt-neo-1.3B-danish' model to generate answers")
    parser.add_argument("output_path", type=str, default="output/devset", help="Path to the output file.")
    parser.add_argument("--retriever", type=str, help="Retrieval model (options: 'tf-idf', 'bm25' or 'dense')")
    parser.add_argument("--k_retrievals", type=int, default=1, help="Number of retrievals, ranging from 1 to 3")
    args = parser.parse_args()

    output_path = args.output_path
    retriever = args.retriever
    k_retrievals = args.k_retrievals

    generate_answers(retriever, k_retrievals, output_path)
    
    print(f"Embeddings saved to {output_path}")