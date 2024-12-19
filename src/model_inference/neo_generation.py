import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

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

    Saves all generated answers to a .txt file after processing.

    Args:
        retriever: 'tfidf', 'bm25', 'bert_cls', 'bert_max' or 'bert_mean'
        k_retrievals: integer between 1 to 3 denoting the amount of retrieved documents (paragraphs)
    """

    neo_answers_list = []  # List to collect all answers

    # Iterating through questions and retrieved documents
    for question, retrieval_column in tqdm(zip(dev_set['question'], dev_set[str(retriever+'_context')]), desc='Answering questions'):
        # Assemble documents into a single string with newlines between each paragraph
        documents = '\n'.join([item for item in retrieval_column][:k_retrievals])

        # Assemble a prompt from the documents, question, and prompting an answer
        prompt = f"Relevante paragraffer: {documents} Spørgsmål: {question} Indsæt svar her baseret på de relevante paragraffer:"

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=924, truncation=True).input_ids.to(DEVICE)
        max_length = len(input_ids[0]) + 100

        # Generate an answer within torch.no_grad() to save compute
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode(' Spørgsmål')[0]
            )

        # Decode generated answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip(' Spørgsmål')

        # Strip the prompt to leave just the answer
        final_answer = answer[len(prompt):].strip()

        final_answer = re.sub('\n', ' ', final_answer)
        # Append the question and answer as a dictionary to the list
        neo_answers_list.append(final_answer)

    return neo_answers_list

if __name__ == "__main__":

    # adding arguments
    import argparse
    parser = argparse.ArgumentParser(description="Set the 'KennethTM/gpt-neo-1.3B-danish' model to generate answers")
    parser.add_argument("--retriever", type=str, help="Retrieval model (options: 'tfidf', 'bm25', 'bert_cls', 'bert_max' and 'bert_mean')")
    parser.add_argument("--k_retrievals", type=int, default=1, help="Number of retrievals, ranging from 1 to 3")
    args = parser.parse_args()

    retriever = args.retriever
    k_retrievals = args.k_retrievals

    # load dev_set
    file_path = "../../output/devset/devset_with_contexts.parquet.gzip"
    with open(file_path, "rb") as f:
        dev_set = pd.read_parquet(f)

    # run function
    answers = generate_answers(retriever, k_retrievals)

    # save results
    with open(f'../../output/inference/neo_gen_{retriever}_k{k_retrievals}.txt', 'w') as file:
        for answer in answers:
            file.write(answer + '\n')

    print(f"Answers saved to output/inference/neo_gen_{retriever}_k{k_retrievals}.txt")