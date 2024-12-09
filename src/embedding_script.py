
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def create_CLS_embedding_matrix(text_list, model_name='vesteinn/DanskBERT', pooling='CLS', device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.to(device)
    
    cls_embeddings = []
    error_count = 0

    for item in tqdm(text_list):
        try:
            input_ids = tokenizer.encode(item['text'], return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(input_ids)
            cls_vector = outputs.last_hidden_state[:, 0, :]
            cls_embeddings.append(cls_vector.cpu())
        except Exception as e:
            cls_embeddings.append(torch.zeros(768))
            error_count += 1
    
    print(f"{error_count} errors encountered.")
    return torch.cat(cls_embeddings, dim=0)

def create_max_pool_embedding_matrix(text_list, model_name='vesteinn/DanskBERT', pooling='max', device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.to(device)
    
    max_embeddings = []
    error_count = 0

    for item in tqdm(text_list):
        try:
            input_ids = tokenizer.encode(item['text'], return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(input_ids)
            max_pool_vector = torch.max(outputs.last_hidden_state, dim=1)[0]
            max_embeddings.append(max_pooled_embedding)
        except Exception as e:
            max_pool_embeddings.append(torch.zeros(768))
            error_count += 1
    
    print(f"{error_count} errors encountered.")
    return torch.cat(max_pool_embeddings, dim=0)

def save_tensor(tensor, path):
    torch.save(tensor, path)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create embeddings for a list of texts.")
    parser.add_argument("input_path", type=str, help="Path to the input file.")
    parser.add_argument("output_path", type=str, help="Path to the output file.")
    parser.add_argument("--embedding_type", type=str, default="CLS", help="Type of embedding to create.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    embedding_type = args.embedding_type



    if embedding_type == "CLS":
        embeddings = create_cls_embedding_matrix(rag_list)
    elif embedding_type == "max-pool":
         embeddings = create_max_pool_embedding_matrix(rag_list)
    elif embedding_type == "mean-pool":
         embeddings = create_mean_pool_embedding_matrix(rag_list)

    save_tensor(embeddings, output_path)
    print(f"Embeddings saved to {output_path}")
