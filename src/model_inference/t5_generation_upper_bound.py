import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re

# load model and tokenizer and set device
model_name = "strombergnlp/dant5-large"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # set up for mac here, change to cuda if needed
model.to(DEVICE)

def generate_answers():
    """
    Generates answers to legal questions with the huggingface-model 'strombergnlp/dant5-large',
    aided by retrieved legal paragraphs.

    Returns a list of abovementioned answers.
    """

    t5_answers_list = []

    # Example question and context

    for question, answer_paragraph in tqdm(zip(dev_set['question'], dev_set['text']), desc='Answering questions'):
        # Assemble documents into a single string with newlines between each paragraph

        # Format the input for T5
        input_text = f"Relevante paragraffer: {answer_paragraph}\nSpørgsmål: {question}\nIndsæt svar her baseret på de relevante paragraffer:"

        # Tokenize the input and generate an answer
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=412, truncation=True).input_ids.to(DEVICE)

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
        answer = re.sub('\n', ' ', answer)
        t5_answers_list.append(answer)

    return t5_answers_list


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Set the 'strombergnlp/dant5-large")
    args = parser.parse_args()

    # load dev_set
    file_path = "../../output/devset/devset_with_contexts.parquet.gzip"
    with open(file_path, "rb") as f:
        dev_set = pd.read_parquet(f)

    # run function
    answers = generate_answers()

    # save results
    with open(f'../../output/inference/t5_gen_upper_bound.txt', 'w') as file:
        for answer in answers:
            file.write(answer + '\n')

    print(f"Answers saved to output/inference/t5_gen_upper_bound.txt")