import faiss
import argparse
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from functools import lru_cache
from typing import List
import warnings
from multiprocessing import cpu_count
from sentence_transformers import SentenceTransformer, models

from data_processor import NQProcessor, index_paths, index_root_path, index_paths_cluster

warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
print(f"Using {faiss.omp_get_max_threads()} instead of value given by cpu_count: {cpu_count()}")

cls = ["sup-simcse-bert-base-uncased"]

def get_embedding(texts: List[str], model=None) -> np.ndarray:
    with torch.no_grad():
        token_embs = model.encode(texts, convert_to_tensor=True)
        sent_embs = F.normalize(token_embs, p=2, dim=1)

    return sent_embs.cpu().numpy().astype('float32') #faiss requires float32

@lru_cache()
def load_model(model_name_full: str):
    model_name = model_name_full.split("/")[-1]
    word_emb_model = models.Transformer(model_name_full, max_seq_length=512)
    if model_name not in cls:
        pooling_model_mean = models.Pooling(
            word_emb_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
    else:
        pooling_model_mean = models.Pooling(
            word_emb_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False
        )
    model = SentenceTransformer(modules=[word_emb_model, pooling_model_mean]).to("cuda").half()
    model.eval()

    return model

def query_faiss_index(query_texts, model_name, index_path, meta_path, top_k=5):
    # Load tokenizer and model
    print("loading model")
    model = load_model(model_name)

    # Load FAISS index and metadata
    print("reading index...")
    index = faiss.read_index(index_path)
    index.nprobe = 1024

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print("computing query embedding...")
    # Compute embeddings for query texts
    embeddings = get_embedding(query_texts, model)

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
            query_results.append(([int(idx), original_id], float(d)))

        results.append(query_results)

    return results

def get_cluster_mask(I, metadata):
    mask = []
    for i in range(len(I)):
        row = []
        for j in range(len(I[i])):
            if metadata[I[i][j]]["type"] == "normal":
                row.append(1)
            else:
                row.append(float("inf"))
        mask.append(row)

    return np.array(mask)

def query_faiss_index_cluster(query_texts, model_name, index_path, meta_path, top_k=5, cluster_weight=0.1):
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
    cluster_lookup = {(v['prefix'], v['label']): k for k, v in metadata.items() if v.get('type') == 'cluster'}

    embeddings = get_embedding(query_texts, tokenizer, model)

    # Search FAISS index
    print("index search...")
    D, I = index.search(embeddings, 500)  # first retrieve the normal embeddings, then combine and re-rank them

    cluster_mask = get_cluster_mask(I, metadata)
    vecs = np.vstack([index.reconstruct(int(idx)) for idx in I.flatten()]).reshape(*I.shape, -1)
    prefix_matrix = np.array(
        [[f"{metadata[int(id)]['prefix']} {metadata[int(id)]['label']}" for id in row] for row in I]) # building prefix matrix
    cluster_ids = np.array([[cluster_lookup[tuple(s.split())] for s in row] for row in prefix_matrix])
    cluster_embeds = np.vstack([index.reconstruct(int(i)) for i in cluster_ids.flatten()])
    cluster_embeds = cluster_embeds.reshape(*cluster_ids.shape, -1)

    vecs = (vecs+cluster_weight*cluster_embeds) / (1+cluster_weight)

    D = np.sum((vecs - embeddings[:, None, :]) ** 2, axis=-1) * cluster_mask[:, None, :]
    reranked_I = np.argsort(D, axis=1)
    D = np.take_along_axis(D, reranked_I, axis=1)[:,:top_k]
    I = np.take_along_axis(I, reranked_I, axis=1)[:,:top_k]

    print("mapping the results...")
    # Map FAISS int IDs to original string IDs
    results = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        query_results = []
        for d, idx in zip(distances, indices):
            original_id = metadata.get(str(idx), None)
            query_results.append(([int(idx), original_id], float(d)))
        results.append(query_results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--cluster', action="store_true", default=False)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--cluster_weight', type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == "nq":
        processor = NQProcessor()
        questions, _ = processor.read_data()
    else:
        raise NotImplementedError()

    if not args.cluster:
        index_path = os.path.join(index_root_path,
                                  index_paths[args.dataset][args.model.split("/")[-1]][args.method]["index"])
        meta_path = os.path.join(index_root_path,
                                 index_paths[args.dataset][args.model.split("/")[-1]][args.method]["meta"])

        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.k}.csv")
    else:
        index_path = os.path.join(index_root_path,
                                  index_paths_cluster[args.dataset][args.model.split("/")[-1]][args.method]["index"])
        meta_path = os.path.join(index_root_path,
                                 index_paths_cluster[args.dataset][args.model.split("/")[-1]][args.method]["meta"])

        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method,
                                f"cluster_{args.cluster_weight}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.k}.csv")

    print("Querying...")
    if not args.cluster:
        predictions = query_faiss_index(
            questions,
            args.model,
            index_path,
            meta_path,
            top_k=args.k
        )
    else:
        predictions = query_faiss_index_cluster(
            questions,
            args.model,
            index_path,
            meta_path,
            top_k=args.k
        )

    results = []

    for i, r in enumerate(predictions):
        tmp = []
        scores = []

        tmp.append(questions[i])
        for original_id, score in r:
            tmp.append(original_id)
            scores.append(score)

        tmp.append(scores)
        results.append(tmp)

    columns = ["question"] + ["ID"+str(i+1) for i in range(args.k)] + ["scores"]
    df = pd.DataFrame(results, columns=columns)

    df.to_csv(save_path, index=False)
