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
                texts, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                # Extract outputs from model
                outputs = model(**encoded)

            # Apply pooling method
            if pooling_method == 'CLS':
                # CLS token embeddings (index 0 of last_hidden_state)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling_method == 'max':
                # Max pooling over sequence dimension
                batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
            elif pooling_method == 'mean':
                # Mean pooling over sequence dimension
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                raise ValueError("Invalid pooling_method. Choose from 'CLS', 'max', or 'mean'.")

            embeddings.append(batch_embeddings.cpu())
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            # Add zero-vectors for the whole batch in case of failure
            embeddings.append(torch.zeros(len(batch), hidden_size))
            error_count += len(batch)

    print(f"{error_count} errors encountered.")
    return torch.cat(embeddings, dim=0)

def save_tensor(tensor, output_path, pooling_method='CLS'):
    """Save a PyTorch tensor to a file."""
    # Add the pooling method to the filename
    output_path = Path(output_path)
    output_path = output_path.with_name(f"{output_path.stem}_{pooling_method}.pt")
    # Save the tensor to the file
    torch.save(tensor, output_path)
    print(f"Embeddings saved to {output_path}")

def load_text_list(input_path):
    """Load a list of dictionaries with 'text' keys from a JSONL file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from a list of texts.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file containing text data.")
    parser.add_argument("output_path", type=str, help="Path to the output file for saving embeddings.")
    parser.add_argument("--pooling_method", type=str, default="CLS", help="Pooling method: CLS, max, or mean.")
    parser.add_argument("--model_name", type=str, default="vesteinn/DanskBERT", help="Hugging Face model name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    
    args = parser.parse_args()

    # Load input data
    print(f"Loading text data from {args.input_path}")
    text_list = load_text_list(args.input_path)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = create_embeddings_matrix(
        text_list,
        model_name=args.model_name,
        pooling_method=args.pooling_method,
        batch_size=args.batch_size
    )

    # Save embeddings to file
    save_tensor(embeddings, args.output_path, pooling_method=args.pooling_method)

if __name__ == "__main__":
    # Example commandline usage: python generate_embeddings.py ../rag_list.jsonl output/embeddings_DanishBERT.pt --pooling_method CLS --model_name vesteinn/DanskBERT --batch_size 32
    main()

    