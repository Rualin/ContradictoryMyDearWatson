import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification


MULTILANG_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
ENGLISH_MODEL_NAME = "roberta-large"
MAX_INPUT_LENGTH = 260
MULTIL_TOKENIZER = XLMRobertaTokenizer.from_pretrained(MULTILANG_MODEL_NAME)
ENGLISH_TOKENIZER = RobertaTokenizer.from_pretrained(ENGLISH_MODEL_NAME)
ENGLISH_PATH = os.path.join("results", "english", "checkpoint-3650")
MULTIL_PATH = os.path.join("results", "multil", "checkpoint-2790")

DEVICE = torch.device("cuda", 3)
torch.set_default_device(DEVICE)


def tokenize_bert(tokenizer, sent1, sent2=None):
    global MAX_INPUT_LENGTH
    if sent2 is None:
        tokens = tokenizer.encode_plus(sent1, return_tensors="pt", 
                                       padding="max_length", truncation=True, 
                                       max_length=MAX_INPUT_LENGTH) 
    else:
        tokens = tokenizer.encode_plus(sent1, sent2, return_tensors="pt", 
                                       padding="max_length", truncation=True, 
                                       max_length=MAX_INPUT_LENGTH)
    return tokens

class BertDataset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer):
        super().__init__()
        self.hyps = df["hypothesis"].to_list()
        self.prems = df["premise"].to_list()
        self.labels = df["label"].to_list()
        length = int(len(self.hyps) * 1)
        self.hyps = self.hyps[:length]
        self.prems = self.prems[:length]
        self.labels = self.labels[:length]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.hyps)
    
    def __getitem__(self, idx):
        encoding = tokenize_bert(self.tokenizer, self.hyps[idx], self.prems[idx])
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx])}


def inference(model, dataset):
    res = []
    dl = DataLoader(dataset, batch_size=len(dataset))
    for sample in dl:
        print(sample)
        sub = model(**sample)
        print(sub)
        res.append(sub)
        break
    return res

def write_result(model, ds, df):
    res = inference(model, ds)
    ids = df["id"].to_list()
    res_dict = {
        "id": ids,
        "prediction": res
    }
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv("submission.csv", sep=",", header=False, mode="a")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    df_test = pd.read_csv(os.path.join("data", "train.csv"))
    df_test = df_test[["id", "premise", "hypothesis", "language", "label"]]
    un_langs = df_test["language"].unique()
    multil_df = df_test[df_test["language"] != "English"]
    english_df = df_test[df_test["language"] == "English"]

    multil_ds = BertDataset(multil_df, MULTIL_TOKENIZER)
    english_ds = BertDataset(english_df, ENGLISH_TOKENIZER)

    eng_model = RobertaForSequenceClassification.from_pretrained(ENGLISH_PATH).to(DEVICE)
    mult_model = XLMRobertaForSequenceClassification.from_pretrained(MULTIL_PATH).to(DEVICE)

    write_result(eng_model, english_ds, english_df)
    write_result(mult_model, multil_ds, multil_df)

    # eng_res = inference(eng_model, english_ds)
    # mult_res = inference(mult_model, multil_ds)

    # eng_ids = english_df["id"].to_list()
    # eng_dict = {
    #     "id": eng_ids,
    #     "prediction": eng_res
    # }
    # eng_res_df = pd.DataFrame(eng_dict)
