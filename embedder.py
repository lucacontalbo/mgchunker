import argparse
import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import List
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def get_embedding(texts: List[str], tokenizer, model) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        if hasattr(model, "encoder"):
            encoder_outputs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            embeddings = encoder_outputs.last_hidden_state[:,0]
            embeddings = embeddings.cpu() / np.linalg.norm(embeddings.cpu(), axis=1, keepdims=True)
        else:
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:,0]
            embeddings = embeddings.cpu() / np.linalg.norm(embeddings.cpu(), axis=1, keepdims=True)

    return embeddings.cpu().numpy().astype('float32')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--output_index', type=str, default='index.faiss')
    parser.add_argument('--output_meta', type=str, default='meta.json')
    parser.add_argument('--nlist', type=int, default=4096)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_train_samples', type=int, default=1_000_000)

    args = parser.parse_args()

    print("Start!")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to("cuda")
    model.eval()

    dim = None
    train_embeddings = []

    print("🔄 First pass: Collect embeddings for training the FAISS index...")
    reader = pd.read_csv(args.datapath, chunksize=args.batch_size)
    total_seen = 0
    for chunk in tqdm(reader, desc="Training sample collection"):
        texts = chunk["contents"].tolist()
        embs = get_embedding(texts, tokenizer, model)
        if dim is None:
            dim = embs.shape[1]
        train_embeddings.append(embs)
        total_seen += len(texts)
        if total_seen >= args.max_train_samples:
            break

    train_matrix = np.concatenate(train_embeddings, axis=0)[:args.max_train_samples]

    print("🏗️ Creating and training FAISS index...")
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, 8)
    index.metric_type = faiss.METRIC_L2
    index.train(train_matrix)

    print("🔄 Second pass: Adding full dataset to the index...")
    metadata = {}
    reader = pd.read_csv(args.datapath, chunksize=args.batch_size)
    current_id = 0
    current_chunk = 0
    for chunk in tqdm(reader, desc="Adding to index"):
        texts = chunk["contents"].tolist()
        string_ids = chunk["id"].tolist()
        embs = get_embedding(texts, tokenizer, model)

        num_vectors = len(embs)
        faiss_ids = np.arange(current_id, current_id + num_vectors, dtype=np.int64)

        # Add vectors with specified FAISS integer IDs
        index.add_with_ids(embs, faiss_ids)

        # Store mapping from FAISS int ID to original string ID
        for fid, sid in zip(faiss_ids, string_ids):
            metadata[str(fid)] = sid  # key = FAISS ID, value = original string ID

        current_id += num_vectors
        current_chunk += 1
        if current_chunk % 10000 == 0:
            print("Current chunk")
            print(current_chunk)
            faiss.write_index(index, args.output_index)
            with open(args.output_meta, "w") as f:
                json.dump(metadata, f, indent=2)


    faiss.write_index(index, args.output_index)
    with open(args.output_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved FAISS index to {args.output_index}")
    print(f"✅ Saved metadata to {args.output_meta}")

if __name__ == "__main__":
    main()

