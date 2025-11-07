import pandas as pd

index_root_path = "./indexes/"

index_paths = {
        "gtr-t5-base": {
            "proposition": {
                "index": "gtr-t5-base/gtr-t5-base_proposition_index.faiss",
                "meta": "gtr-t5-base/gtr-t5-base_proposition_meta.json",
            },
            "sentence": {
                "index": "gtr-t5-base/gtr-t5-base_sentences_index.faiss",
                "meta": "gtr-t5-base/gtr-t5-base_sentences_meta.json"
            },
            "passage": {
                "index": "gtr-t5-base/gtr-t5-base_passages_index.faiss",
                "meta": "gtr-t5-base/gtr-t5-base_passages_meta.json",
            }
        },
        "contriever": {
            "proposition": {
                "index": "contriever/contriever_proposition_index.faiss",
                "meta": "contriever/contriever_proposition_meta.json",
            },
            "sentence": {
                "index": "contriever/contriever_sentences_index.faiss",
                "meta": "contriever/contriever_sentences_meta.json"
            },
            "passage": {
                "index": "contriever/contriever_passages_index.faiss",
                "meta": "contriever/contriever_passages_meta.json",
            }
        },
        "multilingual-e5-large-instruct": {
            "proposition": {
                "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_proposition_index.faiss",
                "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_proposition_meta.json",
            },
            "sentence": {
                "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_sentences_index.faiss",
                "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_sentences_meta.json"
            },
            "passage": {
                "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_passages_index.faiss",
                "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_passages_meta.json",
            }
        },
        "sup-simcse-bert-base-uncased": {
            "proposition": {
                "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_proposition_index.faiss",
                "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_proposition_meta.json",
            },
            "sentence": {
                "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_sentences_index.faiss",
                "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_sentences_meta.json"
            },
            "passage": {
                "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_passages_index.faiss",
                "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_passages_meta.json",
            }
        },
}


class DataProcessor:
    def __init__(self):
        self.data_path = None

    def read_data(self):
        if self.data_path is None:
            return None, None

        df = pd.read_csv(self.data_path, sep="\t")
        questions = df.iloc[:,0].tolist()
        results = df.iloc[:,1].tolist()
        results = [eval(res) for res in results]

        return questions, results

class NQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/nq/test.tsv"

class EQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/eq/test.tsv"

class SquadProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/squad/test.tsv"

class TriviaQAProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/triviaqa/test.tsv"

class WebQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/webq/test.tsv"
