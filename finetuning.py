from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch
from evaluate import load
import random
import numpy as np
from tqdm import tqdm

# Model and tokenizer setup
model_name = "strombergnlp/dant5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset
with open("finetune_list.txt", "r") as file:
    dataset = [line.strip() for line in file]

# Masking function
# Works for one list element in a list
def mask_text(text, mask_probability=0.15):
    # Tokenize string element
    tokens = tokenizer.tokenize(text)
    # Count the number of tokens
    token_count = len(tokens)
    # Determine k number of masks by multiplying no. of tokens with probability, round with int()
    num_masks = int(token_count * mask_probability)
    # Create k number of indices randomly in the range of indices for the number of tokens
    mask_indices = random.sample(range(token_count), num_masks)

    # Create list of masked tokens (input) and target tokens (output) for fine-tuning
    # Also set current mask id to 0, increment while iterating through the masking procedure
    masked_tokens = []
    target_tokens = []
    current_mask_id = 0

    # Iterate through tuples of indices and tokens with enumerate
    for i, token in enumerate(tokens):
        # If the index matches an index marked for masking:
        if i in mask_indices:
            # If there are no masked tokens yet or the previous masked token isn't the current mask id
            if len(masked_tokens) == 0 or masked_tokens[-1] != f"<extra_id_{current_mask_id}>":
                # Append the masked tokens and the target tokens to respective lists
                masked_tokens.append(f"<extra_id_{current_mask_id}>")
                target_tokens.append(f"<extra_id_{current_mask_id}>")
                # Increment current mask id index
                current_mask_id += 1
            # Also append the masked token to target tokens to signify which one it was
            target_tokens.append(token)
        else:
            # If the index doesn't match one marked for masking, simply append the token to the sequence of tokens
            masked_tokens.append(token)

    # Convert from tokens to string (not sure if this is necessary)
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    target_text = tokenizer.convert_tokens_to_string(target_tokens)

    # Return dictionary. This way the masked dataset will be a list of dictionary of lists
    return {"input": masked_text, "target": target_text}

# Mask and tokenize dataset
masked_data = []
# Apply above function to every string element in list
for example in tqdm(dataset, desc='Masking data'):
    result = mask_text(example)
    masked_data.append({
        "input_text": result["input"],
        "target_text": result["target"]
    })

# Apply the tokenizer to the dicts of masked text
tokenized_data = [
    tokenizer(
        data["input_text"],
        text_target=data["target_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ) for data in tqdm(masked_data, desc='Tokenizing masked data, training follows after this')
]

# Create a class for the dataset to fit with torch
class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = [item["input_ids"] for item in tokenized_data]
        self.attention_mask = [item["attention_mask"] for item in tokenized_data]
        self.labels = [item["labels"] for item in tokenized_data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

# instantiating the masked tokenized dataset as an instance of the class
train_dataset = FineTuningDataset(tokenized_data)

# Metrics, using bleu
bleu = load("bleu")

# Define metrics computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return bleu.compute(predictions=decoded_preds, references=decoded_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    # Starting with a relatively low learning rate
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

# Train and save
trainer.train()
model.save_pretrained("finetuned_model")
tokenizer.save_pretrained("finetuned_model")
print("Model saved!")
