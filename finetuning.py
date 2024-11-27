from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments
import random

# save model name
model_name = "strombergnlp/dant5-large"

# initiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# create tokenizer function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# open list of text examples 
with open("finetune_list.txt", "r") as file:
    dataset = [line.strip() for line in file]

# create masking function
def mask_text(text, mask_probability=0.15):
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    num_masks = int(token_count * mask_probability)
    mask_indices = random.sample(range(token_count), num_masks)
    
    masked_tokens = []
    target_tokens = []
    current_mask_id = 0

    for i, token in enumerate(tokens):
        if i in mask_indices:
            if len(masked_tokens) == 0 or masked_tokens[-1] != f"<extra_id_{current_mask_id}>":
                masked_tokens.append(f"<extra_id_{current_mask_id}>")
                target_tokens.append(f"<extra_id_{current_mask_id}>")
                current_mask_id += 1
            target_tokens.append(token)
        else:
            masked_tokens.append(token)

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    target_text = tokenizer.convert_tokens_to_string(target_tokens)

    return {"input": masked_text, "target": target_text}

# mask data
masked_data = []
for example in dataset:
    result = mask_text(example)
    masked_data.append({
        "input_text": result["input"],  # Masked input text
        "target_text": result["target"]  # Target text to reconstruct
    })


# tokenize the masked dataset
tokenized_data = [
    tokenizer(
        data["input_text"],
        text_target=data["target_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ) for data in masked_data
]

# initiate model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# initiate training_args
training_args = TrainingArguments(output_dir="test_trainer")