import faiss
import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

def get_embedding(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        if hasattr(model, "encoder"):
            encoder_outputs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            embeddings = encoder_outputs.last_hidden_state[:,0]
        else:
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:,0]
    return embeddings.cpu().numpy().astype('float32')

@lru_cache()
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")
    return tokenizer, model

def query_faiss_index(query_texts, model_name, index_path, meta_path, top_k=5):
    # Load tokenizer and model
    print("loading model")
    tokenizer, model = load_model(model_name)
    model.eval()

    # Load FAISS index and metadata
    print("reading index...")
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print("computing query embedding...")
    # Compute embeddings for query texts
    embeddings = get_embedding(query_texts, tokenizer, model)

    # Search FAISS index
    print("index search...")
    D, I = index.search(embeddings, top_k)  # D = distances, I = indices

    print("mapping the results...")
    # Map FAISS int IDs to original string IDs
    results = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        query_results = []
        for d, idx in zip(distances, indices):
            original_id = metadata.get(str(idx), None)
            query_results.append((original_id, float(d)))
        results.append(query_results)

    return results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_index', type=str, default='index.faiss')
    parser.add_argument('--input_meta', type=str, default='meta.json')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    query_texts = ["What is the capital of France?", "Explain quantum entanglement."]
    model_name = args.model
    index_path = args.input_index
    meta_path = args.input_meta

    print("Querying...")
    results = query_faiss_index(query_texts, model_name, index_path, meta_path, top_k=args.k)

    for i, r in enumerate(results):
        print(f"\nQuery {i+1}: {query_texts[i]}")
        for original_id, score in r:
            print(f" - ID: {original_id} | Distance: {score:.4f}")
