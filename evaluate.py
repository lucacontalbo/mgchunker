import faiss
import argparse
import json
import pandas as pd
import re
import string
import os
from tqdm import tqdm

from data_processor import index_paths, index_root_path, index_paths_cluster


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--cluster', action="store_true", default=False)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--cluster_weight', type=float, default=0.1)
    args = parser.parse_args()

    if not args.cluster:
        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method)
        file_path = os.path.join(save_dir, f"{args.k}.csv")
    else:
        save_dir = os.path.join("predictions", args.dataset, args.model.split("/")[-1], args.method,
                                f"cluster_{args.cluster_weight}")
        file_path = os.path.join(save_dir, f"{args.k}.csv")

    new_metric = os.path.join(save_dir, f"{args.k}_metric.csv")
    if args.dataset == "nq":
        test_path = "./data/nq/test.csv"
    else:
        raise NotImplementedError()

    """test_df = pd.read_csv(test_path, sep="\t")
    prediction_df = pd.read_csv(file_path)
    data_df = pd.read_csv(f"data/factoid/{args.method}.csv") # may fill the cpu

    assert len(test_df) == len(prediction_df)
    accuracy = []
    for i, row in tqdm(prediction_df.iterrows()):
        found = False
        result = test_df.iloc[i,1]
        if isinstance(result, str):
            result = eval(result)

        ids = [value[1] for j in range(len(prediction_df.columns) - 2) for value in eval(prediction_df.iloc[i, j + 1])]

        texts = []
        for id in ids:
            content = data_df[data_df["id"] == id]["contents"]
            texts.append(normalize_answer(content))

        for res in result:
            res = normalize_answer(res)
            for text in texts:
                if res in text:
                    accuracy.append(1)
                    found = True
                    continue

        if not found:
            accuracy.append(0)

    value = round((sum(accuracy) / len(accuracy))*100, 4)
    metric_df = pd.DataFrame([[value]], columns=[f"Recall@{args.k}"])
    metric_df.to_csv(new_metric, index=False)"""

    test_df = pd.read_csv(test_path, sep="\t")
    prediction_df = pd.read_csv(file_path)
    df = pd.DataFrame()
    for filename in os.listdir("data/factoid/"):
        if filename.split(".")[-1] != "parquet":
            continue
        if str(args.method) not in filename:
            continue

        print(f"Loading file {filename}...")
        df_tmp = pd.read_parquet(f"data/factoid/{filename}")
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.reset_index()
    print("Building index...")
    id2content = dict(zip(df["id"], df["contents"]))

    assert len(test_df) == len(prediction_df)

    accuracy = []

    for i in tqdm(range(len(prediction_df)), desc="Evaluating..."):
        found = False
        result = test_df.iloc[i, 1]
        query = test_df.iloc[i, 0]
        if isinstance(result, str):
            result = eval(result)

        ids = []
        for j in range(len(prediction_df.columns) - 2):
            pred_col = prediction_df.iloc[i, j+1]
            pred_vals = eval(pred_col) if isinstance(pred_col, str) else pred_col

            ids.append(pred_vals[1])
                # [value[1] for value in pred_vals])

        texts = []
        for id_ in ids:
            if id_ not in id2content.keys():
                print("error")
                print(a)
                continue
            texts.append(normalize_answer(id2content[id_]))

        for res in result:
            res = normalize_answer(res)
            if any(res in text for text in texts):
                found = True
                break

        if not found:
            accuracy.append(0)
        else:
            accuracy.append(1)

    # Calculate final metric
    value = round((sum(accuracy) / len(accuracy)) * 100, 4)
    metric_df = pd.DataFrame([[value]], columns=[f"Recall@{args.k}"])
    metric_df.to_csv(new_metric, index=False)
