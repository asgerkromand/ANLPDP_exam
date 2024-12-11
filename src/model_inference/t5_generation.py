import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import ast

# load dev_set
file_path = "output/devset/devset_with_contexts.parquet.gzip"
with open(file_path, "rb") as f:
    dev_set = pd.read_parquet(f)

# converting list columns back to list columns as they are loaded as strings
columns_to_convert = ['tf_idf', 'bm25', 'dense_cls', 'dense_max', 'dense_mean']

for col in columns_to_convert:
    dev_set[col] = dev_set[col].apply(ast.literal_eval)


# load model and tokenizer and set device
model_name = "strombergnlp/dant5-large"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # set up for mac here, change to cuda if needed
model.to(DEVICE)

def generate_answers(retriever, k_retrievals, output_directory):
    """
    Generates answers to legal questions with the huggingface-model 'strombergnlp/dant5-large',
    aided by retrieved legal paragraphs.

    Returns a list of abovementioned answers.

    Args:
        retriever: 'tf_idf', 'bm25', 'dense_cls', 'dense_max' or 'dense_mean'
        k_retrievals: integer between 1 to 3 denoting the amount of retrieved documents (paragraphs)
    """

    t5_answers = []

    # Example question and context

    for question, documents in tqdm(zip(dev_set['question'], dev_set[retriever]), desc='Answering questions'):
        
        # Format the input for T5
        input_text = f"Relevante paragraffer: {documents}\nSpørgsmål: {question}\nIndsæt svar her baseret på de relevante paragraffer:"

        # Tokenize the input and generate an answer
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

        max_length = len(input_ids[0]) + 100

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                # generation set to stop at ' Spørgsmål' as it otherwise just repeats itself (think it's because we don't sample)
                eos_token_id=tokenizer.encode(' Spørgsmål')[0]
            )

        # Decode and print the generated answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip(' Spørgsmål')
        t5_answers.append(answer)

    return t5_answers


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

    answers = generate_answers(retriever, k_retrievals, output_directory)

    with open(f'{output_directory}/neo_gen_{retriever}_k{k_retrievals}.txt', 'w') as outfile:
            outfile.write('\n'.join(str(i) for i in answers))

    print(f"Generated answers saved to saved to f'{output_directory}/neo_gen_{retriever}_k{k_retrievals}'")