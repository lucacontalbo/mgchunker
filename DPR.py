from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

class DPRWrapper(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)

    def forward(self, features):
        # features["input_ids"], features["attention_mask"] come from sentence-transformers
        out = self.model(input_ids=features["input_ids"],
                         attention_mask=features["attention_mask"])
        features["sentence_embedding"] = out.pooler_output  # DPR’s pooling
        return features
