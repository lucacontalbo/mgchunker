import faiss
import argparse
import json
import numpy as np
from collections import defaultdict
import hdbscan
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input_index', type=str, default='index.faiss')
parser.add_argument('--input_meta', type=str, default='meta.json')
parser.add_argument('--output_index', type=str, default='index.faiss')
parser.add_argument('--output_meta', type=str, default='meta.json')

args = parser.parse_args()


# Load index and metadata
index = faiss.read_index(args.input_index)
new_index = faiss.clone_index(index)
new_metadata = {}

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

# 2. For each prefix group, extract embeddings and cluster
results = {}

print("Clustering...")
index.make_direct_map()

for prefix, faiss_ids in tqdm(prefix_to_faiss_ids.items()):
    # Convert to np array
    faiss_ids_np = np.array(faiss_ids, dtype=np.int64)
    
    # 3. Extract embeddings from the FAISS index
    # FAISS stores embeddings internally — to get them, use reconstruct or reconstruct_n
    # reconstruct_n(start, n) reconstructs vectors with IDs from start to start+n-1
    # Since IDs may be non-continuous, use reconstruct for each id:
    
    embeddings = np.array([index.reconstruct(int(fid)) for fid in faiss_ids_np])
    
    # 4. Cluster embeddings (for example, k=3 clusters, tune as needed)
    if len(embeddings) < 2:
        cluster_centroids = {
            0: embeddings[0]
        }
        cluster_labels = [0]
    else:
        min_cluster_size = min(2, len(embeddings))
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        cluster_labels = hdb.fit_predict(embeddings)
    
        cluster_to_embeddings = defaultdict(list)
        for label, emb in zip(cluster_labels, embeddings):
            if label != -1:  # -1 indicates noise in HDBSCAN
                cluster_to_embeddings[label].append(emb)

        # Compute mean vectors for each cluster
        cluster_centroids = {
            label: np.mean(np.stack(emb_list), axis=0)
            for label, emb_list in cluster_to_embeddings.items()
        }
    
    # Store clustering result per prefix
    results[prefix] = {
        'faiss_ids': faiss_ids_np,
        'cluster_labels': cluster_labels,
        'cluster_centroids': cluster_centroids
    }
    
print("Inserting old values into new meta...")
for prefix, res in tqdm(results.items()):
    faiss_ids = res['faiss_ids']
    cluster_labels = res['cluster_labels']
    
    for fid, label in zip(faiss_ids, cluster_labels):
        new_metadata[fid] = {
            'type': 'normal',
            'cluster_label': int(label),
            'prefix': prefix
        }

print("Adding new clusters to index...")
for prefix, res in tqdm(results.items()):
    for label, centroid in tqdm(res["cluster_centroids"].items()):
        # Add centroid to index
        centroid_id = next_id
        next_id += 1

        new_index.add_with_ids(np.expand_dims(centroid.astype(np.float32), axis=0), np.array([centroid_id], dtype=np.int64))
        
        # Save metadata
        new_metadata[centroid_id] = {
            'type': 'cluster_centroid',
            'cluster_label': int(label),
            'prefix': prefix
        }

# Save new FAISS index
faiss.write_index(new_index, args.output_index)

# Save new metadata
with open(args.output_meta, 'w') as f:
    json.dump({str(k): v for k, v in new_metadata.items()}, f, indent=2)

