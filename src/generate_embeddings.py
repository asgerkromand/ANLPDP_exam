import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def create_embeddings_matrix(
    text_list,
    model_name='vesteinn/DanskBERT',
    pooling_method='CLS',
    batch_size=16
):
    """
    Create embeddings matrix using a specified pooling method: CLS, max-pooling, or mean-pooling.

    Args:
        text_list (list): List of dictionaries, each containing a 'text' key with the input text.
        model_name (str): Name of the transformer model.
        pooling_method (str): Pooling method to use ('CLS', 'max', 'mean').
        batch_size (int): Batch size for processing.

    Returns:
        torch.Tensor: Tensor containing the embeddings.
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Determine device (mps for Apple Silicon GPU or fallback to cpu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Get model embedding size dynamically
    hidden_size = model.config.hidden_size

    # Initialize outputs
    embeddings = []
    error_count = 0

    # Validate input
    if not all(isinstance(item, dict) and 'text' in item for item in text_list):
        raise ValueError("Each item in text_list must be a dictionary with a 'text' key.")

    # Process data in batches
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing batches"):
        batch = text_list[i:i + batch_size]
        texts = [item['text'] for item in batch]

        try:
            # Tokenize and encode batch
            encoded = tokenizer(
                texts, return_tensors