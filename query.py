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
from tqdm import tqdm
import warnings
import shelve
from multiprocessing import cpu_count
from sentence_transformers import SentenceTransformer, models
from copy import deepcopy

import gc
import mmap
import pickle

from data_processor import NQProcessor, index_paths, index_root_path, index_paths_cluster, TriviaQAProcessor, \
    WebQProcessor, SquadProcessor, EQProcessor

warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
print(f"Using {faiss.omp_get_max_threads()} instead of value given by cpu_count: {cpu_count()}")

cls = ["sup-simcse-bert-base-uncased"]

class LazyDict:
    def __init__(self, jsonl_path, index_path):
        self.file = open(jsonl_path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

    def __getitem__(self, key):
        try:
            offset, length = self.index[key]
        except KeyError:
            raise KeyError(f"{key!r} not found in index")
        self.mm.seek(offset)
        line = self.mm.read(length)
        obj = json.loads(line)
        # each line looks like {"key": value}
        return obj[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.index.keys()

    def items(self):
        for k in self.index.keys():
            yield k, self[k]

    def __iter__(self):
        return iter(self.index)

    def __contains__(self, key):
        return key in self.index

    def close(self):
        self.mm.close()
        self.file.close()

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
            if str(I[i][j]) != "-1" and metadata[str(I[i][j])]["type"] == "normal":
                row.append(0)
            else:
                row.append(-float("inf"))
        mask.append(row)

    return np.array(mask)

def query_faiss_cluster(query_texts, model_name, index_path, meta_path, top_k=5, save_path=""):
    print("Loading model", flush=True)
    model = load_model(model_name)
    model.eval()

    print("Reading index...", flush=True)
    index = faiss.read_index(index_path)
    index.nprobe = 1024

    print("Compute query embedding...", flush=True)
    embeddings = get_embedding(query_texts, model)

    if "passage" in meta_path:
        method = "passages"
    elif "sentence" in meta_path:
        method = "sentences"
    elif "proposition" in meta_path:
        method = "proposition"

    fid_to_cid_path = os.path.join("/".join(meta_path.split("/")[:-1]), f"{method}_fid_to_cid.json")
    cid_to_fids_path = os.path.join("/".join(meta_path.split("/")[:-1]), f"{method}_cid_to_fids.json")
    with open(fid_to_cid_path, "r") as f:
        fid_to_cid = json.load(f)
    with open(cid_to_fids_path, "r") as f:
        cid_to_fids = json.load(f)

    print("Index search...", flush=True)
    D, I = index.search(embeddings, 200)
    print("Index searched.", flush=True)

    index.make_direct_map()
    print("Created index map. Reconstructing...", flush=True)

    example_vec = index.reconstruct(0)  # just to get the embedding size
    embedding_dim = example_vec.shape[0]

    def compute_mean(faiss_ids):
        embeddings = np.array([index.reconstruct(int(i)) for i in faiss_ids])
        return np.mean(embeddings, axis=0)

    cluster_ids = np.array([
        [
            fid_to_cid[str(id)] if id != -1 and str(id) in fid_to_cid else (-1 if id == -1 else -2)
            for id in row
        ]
        for row in I
    ])
    del fid_to_cid
    gc.collect()

    cluster_embeds = np.vstack([compute_mean(cid_to_fids[str(i)]) if i not in [-1, -2] else np.zeros(embedding_dim) for i in cluster_ids.flatten()]) # zero embeddings for not found indices
    del cid_to_fids
    gc.collect()
    cluster_embeds = cluster_embeds.reshape(*cluster_ids.shape, -1)
    cluster_embeds /= np.linalg.norm(cluster_embeds, axis=-1, keepdims=True) + 1e-12

    del index
    gc.collect()
    print("Cluster embeddings created.", flush=True)

    cluster_weight_mask = np.array([[0 if id in [-1,-2] else 1 for id in row] for row in cluster_ids])
    notfound_weight_mask = np.array([[-float("inf") if id == -1 else 0 for id in row] for row in cluster_ids]) #[:,:,None]
    del cluster_ids
    gc.collect()

    print("Ranking...", flush=True)
    D_original = deepcopy(D)
    I_original = deepcopy(I)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    D_cluster = np.sum(cluster_embeds * embeddings[:, None, :], axis=-1) # * cluster_mask * notfound_weight_mask
    for cluster_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        print(f"Reranking with cluster_weight={cluster_weight}...", flush=True)
        prefix = "/".join(save_path.split("/")[:-1]).format(cluster_weight=cluster_weight)
        os.makedirs(prefix, exist_ok=True)
        D = (D_original + D_cluster * cluster_weight * cluster_weight_mask) / (1+(cluster_weight*cluster_weight_mask))
        D = D + notfound_weight_mask

        reranked_I = np.argsort(-D, axis=1) # reverse sorting. The +10 is to make negative similarities positive, so that we can later negate to obtain the reverse sorting.
        D = np.take_along_axis(D, reranked_I, axis=1)[:,:top_k]
        I = np.take_along_axis(I_original, reranked_I, axis=1)[:,:top_k]

        print("mapping the results...", flush=True)
        # Map FAISS int IDs to original string IDs
        results = []
        for i, (distances, indices) in enumerate(zip(D, I)):
            query_results = []
            for d, idx in zip(distances, indices):
                original_id = metadata.get(str(idx), None)
                query_results.append(([int(idx), original_id], float(d)))
            results.append(query_results)
        save(results, query_texts, top_k, save_path.format(cluster_weight=cluster_weight))

    return results


def query_faiss_index_cluster(query_texts, model_name, index_path, meta_path, top_k=5, cluster_weight=0.1):
    # Load tokenizer and model
    print("loading model", flush=True)
    model = load_model(model_name)
    model.eval()

    # Load FAISS index and metadata
    print("reading index...", flush=True)
    index = faiss.read_index(index_path)
    index.nprobe = 1024

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print("computing query embedding...", flush=True)
    # Compute embeddings for query texts
    cluster_lookup = {(v['prefix'], str(v['cluster_label'])): k for k, v in metadata.items() if v.get('type') != 'normal'}

    embeddings = get_embedding(query_texts, model)
    del model
    gc.collect()

    # Search FAISS index
    print("index search...", flush=True)
    D, I = index.search(embeddings, 200)  # first retrieve the normal embeddings, then combine and re-rank them
    print("Index searched.", flush=True)
    print(D)
    print(I, flush=True)
    # cluster_mask = get_cluster_mask(I, metadata) #[:,:,None]
    index.make_direct_map()
    print("Created index map. Reconstructing...", flush=True)

    example_vec = index.reconstruct(0)  # just to get the embedding size
    embedding_dim = example_vec.shape[0]

    """vecs_list = [
        index.reconstruct(int(idx)) if int(idx) != -1 else np.zeros(embedding_dim)
        for idx in I.flatten()
    ]

    vecs = np.vstack(vecs_list).reshape(I.shape[0], I.shape[1], embedding_dim)
    del vecs_list
    gc.collect()"""

    print("Index reconstructed. Creating cluster embeddings...", flush=True)

    #vecs = np.vstack([index.reconstruct(int(idx)) for idx in I.flatten() if int(idx) != -1]).reshape(*I.shape, -1)
    #vecs = np.vstack([index.reconstruct(int(idx)) for idx in I.flatten() if int(idx) != -1])
    #vecs = vecs.reshape(-1, I.shape[1], vecs.shape[-1])
    prefetched_meta = {str(idx): metadata[str(idx)] for idx in tqdm(I.flatten()) if int(idx) != -1}
    prefix_matrix = np.array([[f"{'-'.join(prefetched_meta[str(id)]['prefix'].split('-')[:2])} {prefetched_meta[str(id)]['cluster_label']}" if int(id) != -1 else "" for id in row] for row in I]) # building prefix matrix
    cluster_ids = np.array([[cluster_lookup[tuple(s.strip().split())] if s != "" else -2 for s in row] for row in prefix_matrix]) # -2 for cluster embs of not found indices
    del prefix_matrix
    del cluster_lookup
    gc.collect()

    cluster_embeds = np.vstack([index.reconstruct(int(i)) if i != -2 else np.zeros(embedding_dim) for i in cluster_ids.flatten()]) # zero embeddings for not found indices
    cluster_embeds = cluster_embeds.reshape(*cluster_ids.shape, -1)
    cluster_embeds /= np.linalg.norm(cluster_embeds, axis=1, keepdims=True) + 1e-12

    del index
    gc.collect()
    print("Cluster embeddings created.", flush=True)

    cluster_weight_mask = np.array([[0 if id != -2 and metadata[str(id)]["cluster_label"] == -1 else 1 for id in row] for row in cluster_ids])
    notfound_weight_mask = np.array([[-float("inf") if id == -2 else 0 for id in row] for row in cluster_ids]) #[:,:,None]
    del cluster_ids
    gc.collect()

    # samples not clustered (-1 cluster label) have a cluster embedding of zeros. In that case, vecs+cluster_weight*cluster_embeds = vecs. The cluster_weight_mask is 0, so that
    # in those cases, the embedding is just original chunk embedding
    # for the notfound_weight_mask, index.search() returns -1 if no chunk is found. In those cases, I use notfound_weight_mask to obtain and +inf embedding, making the distance high
    
    #vecs = (vecs+cluster_weight*cluster_embeds) / (1+(cluster_weight*cluster_weight_mask))
    print("Ranking...", flush=True)

    cluster_mask = get_cluster_mask(I, prefetched_meta)
    #D = np.sum((vecs - embeddings[:, None, :]) ** 2, axis=-1) * cluster_mask * notfound_weight_mask
    #D_cluster = np.sum((cluster_embeds - embeddings[:,None,:])**2, axis=-1) * cluster_mask * notfound_weight_mask
    D_cluster = np.sum(cluster_embeds * embeddings[:, None, :], axis=-1) # * cluster_mask * notfound_weight_mask
    D = (D + D_cluster * cluster_weight * cluster_weight_mask) / (1+(cluster_weight*cluster_weight_mask))
    D = D + cluster_mask + notfound_weight_mask

    reranked_I = np.argsort(-D, axis=1) # reverse sorting. The +10 is to make negative similarities positive, so that we can later negate to obtain the reverse sorting.
    D = np.take_along_axis(D, reranked_I, axis=1)[:,:top_k]
    I = np.take_along_axis(I, reranked_I, axis=1)[:,:top_k]

    print("mapping the results...", flush=True)
    # Map FAISS int IDs to original string IDs
    results = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        query_results = []
        for d, idx in zip(distances, indices):
            original_id = prefetched_meta.get(str(idx), None)
            if original_id and original_id["type"] == "normal":
                original_id = original_id["prefix"]
            else:
                raise ValueError(f"insufficient number of 'normal' embeddings found in top@{top_k} setting")
            query_results.append(([int(idx), original_id], float(d)))
        results.append(query_results)

    return results

def query_faiss_index_cluster_shelved(query_texts, model_name, index_path, meta_path, top_k=5, cluster_weight=0.1):
    # Load tokenizer and model
    print("loading model", flush=True)
    model = load_model(model_name)
    model.eval()

    # Load FAISS index and metadata
    print("reading index...", flush=True)
    index = faiss.read_index(index_path)
    index.nprobe = 1024

    idx_path = ".".join(meta_path.split(".")[:-1])+".idx"

    def iter_metadata(jsonl_file):
        with open(jsonl_file, "rb") as f:
            for line in f:
                obj = json.loads(line)
                key = list(obj.keys())[0]      # extract the key
                value = obj[key]               # extract the value
                yield key, value

    metadata = LazyDict(meta_path, idx_path)

    print("computing query embedding...", flush=True)
    # Compute embeddings for query texts
    cluster_lookup = {}
    for k,v in iter_metadata(meta_path):
        if v.get('type') != 'normal':
            cluster_lookup[(v['prefix'], str(v['cluster_label']))] = k

    embeddings = get_embedding(query_texts, model)
    del model
    gc.collect()

    # Search FAISS index
    print("index search...", flush=True)
    D, I = index.search(embeddings, 200)  # first retrieve the normal embeddings, then combine and re-rank them
    print("Index searched.", flush=True)

    # cluster_mask = get_cluster_mask(I, metadata) #[:,:,None]
    index.make_direct_map()
    print("Created index map. Reconstructing...", flush=True)

    example_vec = index.reconstruct(0)  # just to get the embedding size
    embedding_dim = example_vec.shape[0]

    """vecs_list = [
        index.reconstruct(int(idx)) if int(idx) != -1 else np.zeros(embedding_dim)
        for idx in I.flatten()
    ]

    vecs = np.vstack(vecs_list).reshape(I.shape[0], I.shape[1], embedding_dim)"""
    print("Index reconstructed. Creating cluster embeddings...", flush=True)
    print(I.shape)
    prefetched_meta = {str(idx): metadata[str(idx)] for idx in tqdm(I.flatten()) if int(idx) != -1}
    print("Prefetched meta computed.")

    #vecs = np.vstack([index.reconstruct(int(idx)) for idx in I.flatten()]).reshape(*I.shape, -1)
    prefix_matrix = np.array([[f"{'-'.join(prefetched_meta[str(id)]['prefix'].split('-')[:2])} {prefetched_meta[str(id)]['cluster_label']}" if int(id) != -1 else "" for id in row] for row in I]) # building$
    cluster_ids = np.array([[cluster_lookup[tuple(s.strip().split())] if s != "" else -2 for s in row] for row in prefix_matrix]) # -2 for cluster embs of not found indices

    cluster_embeds = np.vstack([index.reconstruct(int(i)) if i != -2 else np.zeros(embedding_dim) for i in cluster_ids.flatten()]) # zero embeddings for not found indices
    cluster_embeds = cluster_embeds.reshape(*cluster_ids.shape, -1)
    cluster_embeds /= np.linalg.norm(cluster_embeds, axis=1, keepdims=True) + 1e-12

    del index
    gc.collect()
    print("Cluster embeddings created.", flush=True)

    cluster_weight_mask = np.array([[0 if id != -2 and metadata[str(id)]["cluster_label"] == -1 else 1 for id in row] for row in cluster_ids])        
    notfound_weight_mask = np.array([[-float("inf") if id == -2 else 1 for id in row] for row in cluster_ids]) #[:,:,None]

    # samples not clustered (-1 cluster label) have a cluster embedding of zeros. In that case, vecs+cluster_weight*cluster_embeds = vecs. The cluster_weight_mask is 0, so that
    # in those cases, the embedding is just original chunk embedding
    # for the notfound_weight_mask, index.search() returns -1 if no chunk is found. In those cases, I use notfound_weight_mask to obtain and +inf embedding, making the distance high
    #vecs = (vecs+cluster_weight*cluster_embeds) / (1+(cluster_weight*cluster_weight_mask))
    print("Score created. Ranking...", flush=True)

    cluster_mask = get_cluster_mask(I, prefetched_meta)
    #D = np.sum((vecs - embeddings[:, None, :]) ** 2, axis=-1) * cluster_mask * notfound_weight_mask
    D_cluster = np.sum(cluster_embeds * embeddings[:, None, :], axis=-1) * cluster_mask * notfound_weight_mask
    D = (D + D_cluster * cluster_weight * cluster_weight_mask) / (1+(cluster_weight*cluster_weight_mask))

    reranked_I = np.argsort(-(D+10), axis=1) # reverse sorting
    D = np.take_along_axis(D, reranked_I, axis=1)[:,:top_k]
    I = np.take_along_axis(I, reranked_I, axis=1)[:,:top_k]

    print("mapping the results...", flush=True)
    # Map FAISS int IDs to original string IDs
    results = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        query_results = []
        for d, idx in zip(distances, indices):
            original_id = prefetched_meta.get(str(idx), None)
            if original_id and original_id["type"] == "normal":
                original_id = original_id["prefix"]
            else:
                raise ValueError(f"insufficient number of 'normal' embeddings found in top@{top_k} setting")
            query_results.append(([int(idx), original_id], float(d)))
        results.append(query_results)

    return results

def save(predictions, questions, k, save_path):
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

    columns = ["question"] + ["ID"+str(i+1) for i in range(k)] + ["scores"]
    df = pd.DataFrame(results, columns=columns)

    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--cluster', action="store_true", default=False)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if args.dataset == "nq":
        processor = NQProcessor()
    elif args.dataset == "eq":
        processor = EQProcessor()
    elif args.dataset == "squad":
        processor = SquadProcessor()
    elif args.dataset == "triviaqa":
        processor = TriviaQAProcessor()
    elif args.dataset == "webq":
        processor = WebQProcessor()
    else:
        raise NotImplementedError()

    questions, _ = processor.read_data()

    if not args.cluster:
        index_path = os.path.join(index_root_path,
                                  index_paths[args.model.split("/")[-1]][args.method]["index"])
        meta_path = os.path.join(index_root_path,
                                 index_paths[args.model.split("/")[-1]][args.method]["meta"])

        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.k}.csv")
    else:
        index_path = os.path.join(index_root_path,
                                  index_paths[args.model.split("/")[-1]][args.method]["index"])
        meta_path = os.path.join(index_root_path,
                                 index_paths[args.model.split("/")[-1]][args.method]["meta"])

        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method,
                                "cluster_{cluster_weight}")
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
    elif args.method == "proposition":
        if "multilingual" in args.model and args.dataset == "eq":
            predictions1 = query_faiss_cluster(
                questions[:len(questions)//2],
                args.model,
                index_path,
                meta_path,
                top_k=args.k,
                save_path=save_path
            )
            predictions2 = query_faiss_cluster(
                questions[len(questions)//2:],
                args.model,
                index_path,
                meta_path,
                top_k=args.k,
                save_path=save_path
            )
            predictions = predictions1+predictions2
        else:
            predictions = query_faiss_cluster(
                questions,
                args.model,
                index_path,
                meta_path,
                top_k=args.k,
                save_path=save_path
            )
    elif args.method == "sentence":
        predictions = query_faiss_cluster(
            questions,
            args.model,
            index_path,
	    meta_path,
            top_k=args.k,
            save_path=save_path
        )
    else:
        predictions = query_faiss_cluster(
            questions,
            args.model,
            index_path,
            meta_path,
            top_k=args.k,
            save_path=save_path
        )

    """results = []

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

    df.to_csv(save_path, index=False)"""
