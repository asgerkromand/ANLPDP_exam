import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration
import ast
import sys

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

def generate_answers(retriever, k_retrievals):
    """
    Generates answers to legal questions with the huggingface-model 'KennethTM/gpt-neo-1.3B-danish',
    aided by retrieved legal paragraphs.

    Returns a list of abovementioned answers.

    Args:
        retriever: 'tf_idf', 'bm25', 'dense_cls', 'dense_max' or 'dense_mean'
        k_retrievals: integer between 1 to 3 denoting the amount of retrieved documents (paragraphs)
    """

    neo_answers = []

    # iterating through questions and retrieved documents
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

    return neo_answers


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Set the 'KennethTM/gpt-neo-1.3B-danish' model to generate answers")
    parser.add_argument("--output_directory", type=str, default="output/devset", help="Path to the output file.")
    parser.add_argument("--retriever", type=str, help="Retrieval model (options: 'tf-idf', 'bm25' or 'dense')")
    parser.add_argument("--k_retrievals", type=int, default=1, help="Number of retrievals, ranging from 1 to 3")
    args = parser.parse_args()

    output_directory = args.output_directory
    retriever = args.retriever
    k_retrievals = args.k_retrievals

    answers = generate_answers(retriever, k_retrievals)

    with open(f'{output_directory}/t5_gen_{retriever}_k{k_retrievals}.txt', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in answers))

    print(f"Generated answers saved to saved to f'{output_directory}/neo_gen_{retriever}_k{k_retrievals}'")