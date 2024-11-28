from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch
from evaluate import load
import random

# Model and tokenizer setup
model_name = "strombergnlp/dant5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset
with open("finetune_list.txt", "r") as file:
    dataset = [line.strip() for line in file]

# Masking function
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

# Mask and tokenize dataset
masked_data = []
for example in dataset:
    result = mask_text(example)
    masked_data.append({
        "input_text": result["input"],
        "target_text": result["target"]
    })

tokenized_data = [
    tokenizer(
        data["input_text"],
        text_target=data["target_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    ) for data in masked_data
]

# PyTorch dataset
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

train_dataset = FineTuningDataset(tokenized_data)

# Metrics
bleu = load("bleu")

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
