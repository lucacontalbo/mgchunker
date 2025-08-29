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
from sentence_transformers import SentenceTransformer, models

warnings.filterwarnings("ignore")
faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count())))
print(f"Using {faiss.omp_get_max_threads()} instead of value given by cpu_count: {cpu_count()}")

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

pooler_output = ["dpr-ctx_encoder-multiset-base"]
cls = ["sup-simcse-bert-base-uncased"]
st = ["gtr-t5-base","contriever", "multilingual-e5-large-instruct","sup-simcse-bert-base-uncased","dpr-ctx_encoder-multiset-base"] #sentencetransformer
mean_pooling = ["contriever", "multilingual-e5-large-instruct"]

def get_embedding(texts: List[str], model_name, tokenizer=None, model=None, return_gpu=False, verbose=False) -> np.ndarray:
    if tokenizer is not None:
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")

    with torch.no_grad():
        #with torch.cuda.amp.autocast(dtype=torch.float16):
        if model_name in st:
                token_embs = model.encode(texts, convert_to_tensor=True)
            #elif hasattr(model, "encoder"):
            #    token_embs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        else:
                token_embs = model(**encoded)

        if model_name in st:
            sent_embs = token_embs
        elif model_name in mean_pooling:
            token_embs = token_embs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embs.dtype)
            sent_embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        elif model_name in pooler_output:
            sent_embs = token_embs.pooler_output
        else:
            token_embs = token_embs.last_hidden_state
            sent_embs = token_embs[:,0] #cls
        sent_embs = F.normalize(sent_embs, p=2, dim=1)

    if verbose:
        print(sent_embs.shape, flush=True)
    #if return_gpu:
    #    return sent_embs.to(torch.float32)
    return sent_embs.cpu().numpy().astype('float32') #faiss requires float32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--output_index', type=str, default='index.faiss')
    parser.add_argument('--output_meta', type=str, default='meta.json')
    parser.add_argument('--nlist', type=int, default=1024)
    parser.add_argument('--m', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--max_train_samples', type=int, default=300_000)
    args = parser.parse_args()

    print("Start!")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    model_name = args.model.split("/")[-1]
    if model_name == "multilingual-e5-large-instruct":
        print("Overriding args.m parameter so that the vector dimension is divisible by args.m")
        args.m = 64 # multilingual-e5 has a vector dim of 1024. m must divide the vector dim, so we force the number of subquantizers
    if model_name in st:
        tokenizer = None
        word_emb_model = models.Transformer(args.model, max_seq_length=512)
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
        #model = SentenceTransformer(args.model).to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to("cuda")
    model.eval()

    dim = None
    train_embeddings = []
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print("🔄 First pass: Collect embeddings for training the FAISS index...")
    reader = pd.read_csv(args.datapath, chunksize=args.batch_size)
    total_seen = 0
    for i,chunk in tqdm(enumerate(reader), desc="Training sample collection"):
        texts = chunk["contents"].tolist()
        if i == 0:
            verbose = True
        else:
            verbose = False
        embs = get_embedding(texts, model_name, tokenizer, model, verbose=verbose)
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
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, 8)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    print("Training...")
    index.train(train_matrix)

    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print("🔄 Second pass: Adding full dataset to the index...")
    metadata = {}
    # capping batch_size to reduce gpu overload
    batch_size = 7168 #8192 #min(args.batch_size, 1024)
    reader = pd.read_csv(args.datapath, chunksize=batch_size)
    current_id = 0
    current_chunk = 0
    future = None
    acc_embs, acc_faiss = np.array([]), np.array([])
    acc = 3

    for chunk in tqdm(reader, desc="Adding to index"):
        """if current_chunk != 0 and current_chunk % acc == 0:
            if len(acc_embs) == 0:
                acc_embs = """

        texts = chunk["contents"].tolist()
        string_ids = chunk["id"].tolist()

        if future is None:
            future = executor.submit(get_embedding, texts, model_name, tokenizer, model, True)
            faiss_ids = np.arange(current_id, current_id + len(string_ids), dtype=np.int64)
            old_metadata = {str(fid): sid for fid, sid in zip(faiss_ids, string_ids)}
            continue

        embs = future.result()
        #embs = get_embedding(texts, model_name, tokenizer, model, return_gpu=True)

        future = executor.submit(get_embedding, texts, model_name, tokenizer, model, True)

        index.add_with_ids(embs, faiss_ids)

        metadata.update(old_metadata)
        current_id += len(faiss_ids)

        faiss_ids = np.arange(current_id, current_id + len(string_ids), dtype=np.int64)
        old_metadata = {str(fid): sid for fid, sid in zip(faiss_ids, string_ids)}

        # store mapping from FAISS int ID to original string ID
        #for fid, sid in zip(faiss_ids, string_ids):
        #    metadata[str(fid)] = sid  # key = FAISS ID, value = original string ID
        #metadata.update({str(fid): sid for fid, sid in zip(faiss_ids, string_ids)})

        #current_id += num_vectors
        current_chunk += 1
        if current_chunk % 1000 == 0:
            print("Current chunk", flush=True)
            print(current_chunk, flush=True)
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)

            #index2 = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, args.output_index)
            with open(args.output_meta, "w") as f:
                json.dump(metadata, f, indent=2)

    embs = future.result()
    index.add_with_ids(embs, faiss_ids)
    metadata.update(old_metadata)

    #index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, args.output_index)
    with open(args.output_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved FAISS index to {args.output_index}")
    print(f"✅ Saved metadata to {args.output_meta}")

if __name__ == "__main__":
    main()

