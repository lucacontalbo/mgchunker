import argparse
import json
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import List
from tqdm import tqdm
import os
import warnings
from multiprocessing import cpu_count
import time
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
print(f"Using {faiss.omp_get_max_threads()} instead of value given by cpu_count: {cpu_count()}")

pooler_output = ["dpr-ctx_encoder-multiset-base"]
cls = ["sup-simcse-bert-base-uncased"]
st = ["gtr-t5-base"] #sentencetransformer
mean_pooling = ["contriever", "multilingual-e5-large-instruct"]

def get_embedding(texts: List[str], model_name, tokenizer=None, model=None) -> np.ndarray:
    if tokenizer is not None:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")

    with torch.no_grad():
        if model_name in st:
            token_embs = model.encode(texts, convert_to_tensor=True)
        #elif hasattr(model, "encoder"):
        #    token_embs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        else:
            token_embs = model(**encoded)

        if model_name in mean_pooling:
            token_embs = token_embs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embs.dtype)
            sent_embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        elif model_name in pooler_output:
            sent_embs = token_embs.pooler_output
        elif model_name in st:
            sent_embs = token_embs
        else:
            token_embs = token_embs.last_hidden_state
            sent_embs = token_embs[:,0] #cls
        sent_embs = F.normalize(sent_embs, p=2, dim=1)

    print(sent_embs.shape)
    return sent_embs.cpu().numpy().astype('float32')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--output_index', type=str, default='index.faiss')
    parser.add_argument('--output_meta', type=str, default='meta.json')
    parser.add_argument('--nlist', type=int, default=1024)
    parser.add_argument('--m', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_train_samples', type=int, default=300_000)
    args = parser.parse_args()

    print("Start!")
    model_name = args.model.split("/")[-1]
    if model_name in st:
        tokenizer = None
        model = SentenceTransformer(args.model).to("cuda")
    else:
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
        embs = get_embedding(texts, model_name, tokenizer, model)
        if dim is None:
            dim = embs.shape[1]
        train_embeddings.append(embs)
        total_seen += len(texts)
        if total_seen >= args.max_train_samples:
            break

    print("Out of the train extraction. Concatenating...")
    t0 = time.time()
    train_matrix = np.concatenate(train_embeddings, axis=0)[:args.max_train_samples]
    print("Concatenate done. shape:", train_matrix.shape, "took", time.time()-t0, "s", flush=True)

    print("🏗️ Creating and training FAISS index...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, 16)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(train_matrix)

    print("🔄 Second pass: Adding full dataset to the index...")
    metadata = {}
    reader = pd.read_csv(args.datapath, chunksize=args.batch_size)
    current_id = 0
    current_chunk = 0
    for chunk in tqdm(reader, desc="Adding to index"):
        texts = chunk["contents"].tolist()
        string_ids = chunk["id"].tolist()
        embs = get_embedding(texts, model_name, tokenizer, model)

        num_vectors = len(embs)
        faiss_ids = np.arange(current_id, current_id + num_vectors, dtype=np.int64)

        # add vectors with specified FAISS integer IDs
        index.add_with_ids(embs, faiss_ids)

        # store mapping from FAISS int ID to original string ID
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

    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, args.output_index)
    with open(args.output_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved FAISS index to {args.output_index}")
    print(f"✅ Saved metadata to {args.output_meta}")

if __name__ == "__main__":
    main()

