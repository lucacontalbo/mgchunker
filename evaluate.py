import faiss
import argparse
import pandas as pd
import re
import string
import os
from tqdm import tqdm
import gc
from data_processor import NQProcessor, EQProcessor, SquadProcessor, TriviaQAProcessor, WebQProcessor

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

    questions, answers = processor.read_data()

    print("loading prediction df", flush=True)
    prediction_df = pd.read_csv(file_path)
    print("ended loading", flush=True)
    df = pd.DataFrame()
    for filename in os.listdir("data/factoid/"):
        if filename.split(".")[-1] != "parquet":
            continue
        # we only need passage and not sentence/proposition level data, since the retriever is only evaluated at passage level
        if "passage" not in filename:
            continue

        print(f"Loading file {filename}...", flush=True)
        df_tmp = pd.read_parquet(f"data/factoid/{filename}")
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.reset_index()
    del df_tmp
    gc.collect()
    print("Building index...", flush=True)
    if args.method != "proposition":
        id2content = dict(zip(df["id"], df["contents"]))
        id2content_ok = True
    else:
        df = df.set_index("id")
        id2content_ok = False

    assert len(questions) == len(answers) == len(prediction_df)

    accuracy = []

    for i in tqdm(range(len(prediction_df)), desc="Evaluating..."):
        found = False
        result = answers[i]
        query = questions[i]

        if isinstance(result, str):
            result = eval(result)

        ids = []
        for j in range(len(prediction_df.columns) - 2):
            pred_col = prediction_df.iloc[i, j+1]
            pred_vals = eval(pred_col) if isinstance(pred_col, str) else pred_col

            ids.append(pred_vals[1])

        texts = []
        print(f"Query: {query}", flush=True)
        print(ids, flush=True)
        for id_ in ids:
            if args.method in ["sentence", "proposition"]:
                id_ = "-".join(id_.split("-")[:-1])

            if not id2content_ok:
                content = df.at[id_, "contents"]
                texts.append(normalize_answer(content))
            else:
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
        print(f"Accuracy: {accuracy[-1]}", flush=True)

    # final metric
    value = round((sum(accuracy) / len(accuracy)) * 100, 4)
    metric_df = pd.DataFrame([[value]], columns=[f"Recall@{args.k}"])
    metric_df.to_csv(new_metric, index=False)
