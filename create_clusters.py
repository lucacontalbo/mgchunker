import faiss
import argparse
import json
import numpy as np
from collections import defaultdict
import hdbscan
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil

import warnings
warnings.filterwarnings("ignore")

def reconstruct_vector(i_fid):
    i, fid = i_fid
    return i, index_for_reconstruct.reconstruct(int(fid))

task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

parser = argparse.ArgumentParser()
parser.add_argument('--input_index', type=str, default='index.faiss')
parser.add_argument('--input_meta', type=str, default='meta.json')
#parser.add_argument('--output_index', type=str, default='index.faiss')
parser.add_argument('--output_meta', type=str, default='./')

args = parser.parse_args()

if "multilingual-e5" in args.input_index:
    dim = 1024
else:
    dim = 768

# Load index and metadata
index = faiss.read_index(args.input_index)
index_for_reconstruct = faiss.read_index(args.input_index)
#new_index = faiss.clone_index(index)
fid_to_cid = {}
cid_to_fids = {}

with open(args.input_meta, 'r') as f:
    metadata = json.load(f)

current_max_id = max(map(int, metadata.keys()))
next_id = current_max_id + 1

# 1. Group FAISS IDs by prefix of original string ID
print("Grouping by prefix...")
prefix_to_faiss_ids = defaultdict(list)

for faiss_id_str, orig_id in metadata.items():
    prefix = "-".join(orig_id.split('-')[:2])  # prefix before first underscore
    prefix_to_faiss_ids[prefix].append(int(faiss_id_str))

del metadata
gc.collect()

print("Clustering...")
index_for_reconstruct.make_direct_map()
num_threads = 8
os.makedirs(f"tmp_{task_id}", exist_ok=True)
max_cluster_idx, new_max_value = 0, 0

for iter, (prefix, faiss_ids) in tqdm(enumerate(prefix_to_faiss_ids.items())):
    results = {}
    faiss_ids_np = np.array(faiss_ids, dtype=np.int64)
    
    # 3. Extract embeddings from the FAISS index
    mmap = np.memmap(f'./tmp_{task_id}/embeddings.dat', dtype='float32', mode='w+', shape=(len(faiss_ids_np), dim))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, vec in executor.map(reconstruct_vector, enumerate(faiss_ids_np)):
            mmap[i] = vec

    embeddings = mmap

    if len(embeddings) < 2:
        zero_array = np.zeros_like(embeddings[0])
        cluster_centroids = {
            0: zero_array,
        }
        cluster_labels = [-1]
    else:
        min_cluster_size = min(2, len(embeddings))
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        cluster_labels = hdb.fit_predict(embeddings)
    
        """cluster_to_embeddings = defaultdict(list)
        for label, emb in zip(cluster_labels, embeddings):
            if label != -1:  # -1 indicates noise in HDBSCAN
                cluster_to_embeddings[label].append(emb)
            else:
                cluster_to_embeddings[label].append(np.zeros_like(emb))

        # compute mean vectors for each cluster
        cluster_centroids = {
            label: np.mean(np.stack(emb_list), axis=0)
            for label, emb_list in cluster_to_embeddings.items()
        }"""
    
    # store clustering result per prefix
    results[prefix] = {
        'faiss_ids': faiss_ids_np,
        'cluster_labels': cluster_labels,
        #'cluster_centroids': cluster_centroids
    }
    

    for prefix, res in results.items():
        max_cluster_idx = new_max_value
        if max_cluster_idx != 0:
            max_cluster_idx += 1
        faiss_ids = res['faiss_ids']
        cluster_labels = res['cluster_labels']
        for fid, label in zip(faiss_ids, cluster_labels):
            if label == -1:
                continue
            
            """fid_to_cid[fid] = {
                'type': 'normal',
                'cluster_label': int(label),
                'prefix': prefix
            }"""
            cid = max_cluster_idx+label
            fid_to_cid[fid] = cid
            if cid not in cid_to_fids.keys():
                cid_to_fids[cid] = []

            cid_to_fids[cid].append(int(fid))
            
            if cid > new_max_value:
                new_max_value = cid

    #print("Adding new clusters to index...")
    """for prefix, res in results.items():
        for label, centroid in res["cluster_centroids"].items():
            # Add centroid to index
            centroid_id = next_id
            next_id += 1

            index.add_with_ids(np.expand_dims(centroid.astype(np.float32), axis=0), np.array([centroid_id], dtype=np.int64))
        
        # Save metadata
            fid_to_cid[centroid_id] = {
                'type': 'cluster_centroid',
                'cluster_label': int(label),
                'prefix': prefix
            }

    if iter % 2_000_000 == 0:
        output_meta = args.output_meta.split(".")[0]+f"_{iter}."+args.output_meta.split(".")[-1]
        with open(output_meta, 'w') as f:
            json.dump({str(k): v for k, v in fid_to_cid.items()}, f, indent=2)
        fid_to_cid = {}"""

# Save new FAISS index
#faiss.write_index(index, args.output_index)

# Save new metadata

fid_to_cid_path = args.output_meta+"_fid_to_cid.json"
with open(fid_to_cid_path, 'w') as f:
    json.dump({int(k): int(v) for k, v in fid_to_cid.items()}, f, indent=2)

cid_to_fids_path = args.output_meta+"_cid_to_fids.json"
with open(cid_to_fids_path, 'w') as f:
    json.dump({int(k): v for k, v in cid_to_fids.items()}, f, indent=2)

if os.path.exists(f"tmp_{task_id}") and os.path.isdir(f"tmp_{task_id}"):
    shutil.rmtree(f"tmp_{task_id}")
