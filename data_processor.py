import pandas as pd

index_root_path = "./indexes/"

index_paths = {
    "nq": {
        "gtr-t5-base": {
            "proposition": {
                "index": "gtr-t5-base_index.faiss",
                "meta": "gtr-t5-base_meta.json",
            },
            "sentence": {
                "index": "gtr-t5-base_index_sentences.faiss",
                "meta": "gtr-t5-base_meta_sentences.json"
            },
            "passage": {
                "index": "gtr-t5-base_index_passages.faiss",
                "meta": "gtr-t5-base_meta_passages.json",
            }
        }
    }
}

index_paths_cluster = {
    "nq": {
        "gtr-t5-base": {
            "proposition": {},
            "sentence": {
                "index": "gtr-t5-base_index_sentences_cluster.faiss",
                "meta": "gtr-t5-base_meta_sentences_cluster.json"
            },
            "passage": {
                "index": "gtr-t5-base_index_passages_cluster.faiss",
                "meta": "gtr-t5-base_meta_passages_cluster.json",
            }
        }
    }
}


class DataProcessor:
    def __init__(self):
        pass

class NQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/nq/test.csv"

    def read_data(self):
        df = pd.read_csv(self.data_path, sep="\t")
        questions = df.iloc[:,0].tolist()
        results = df.iloc[:,1].tolist()
        results = [eval(res) for res in results]

        return questions, results
