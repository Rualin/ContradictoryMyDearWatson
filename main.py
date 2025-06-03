import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import base as ba


torch.set_default_device(ba.DEVICE)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
MULTILANG_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
ENGLISH_MODEL_NAME = "roberta-large"
MAX_INPUT_LENGTH = 260
MULTIL_TOKENIZER = XLMRobertaTokenizer.from_pretrained(MULTILANG_MODEL_NAME)
ENGLISH_TOKENIZER = RobertaTokenizer.from_pretrained(ENGLISH_MODEL_NAME)
print("End of loading tokenizer")
MULTIL_ARGS = TrainingArguments(
    output_dir=os.path.join("results", "multil"),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=True,
    overwrite_output_dir=True,
    save_total_limit=1,
    greater_is_better=True
)
ENGLISH_ARGS = TrainingArguments(
    output_dir=os.path.join("results", "english"),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=True,
    overwrite_output_dir=True,
    save_total_limit=1,
    greater_is_better=True
)

def tokenizing(tokenizer, sent1, sent2=None):
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
        encoding = tokenizing(self.tokenizer, self.hyps[idx], self.prems[idx])
        # print(encoding)
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx])}


if __name__ == "__main__":
    torch.cuda.empty_cache()
    df_train = pd.read_csv(os.path.join("data", "train.csv"))
    df_train = df_train[["id", "premise", "hypothesis", "language", "label"]]
    un_langs = df_train["language"].unique()
    multil_df = df_train[df_train["language"] != "English"]
    english_df = df_train[df_train["language"] == "English"]

    mult_train_ds = BertDataset(multil_df, MULTIL_TOKENIZER)
    generator = torch.Generator(device=ba.DEVICE).manual_seed(42)
    mult_train_ds, mult_val_ds = torch.utils.data.random_split(mult_train_ds, [0.85, 0.15], generator)

    eng_train_ds = BertDataset(english_df, ENGLISH_TOKENIZER)
    eng_train_ds, eng_val_ds = torch.utils.data.random_split(eng_train_ds, [0.85, 0.15], generator)

    eng_model = RobertaForSequenceClassification.from_pretrained(ENGLISH_MODEL_NAME, num_labels=3).to(ba.DEVICE)
    eng_trainer = Trainer(
        model=eng_model,
        args=ENGLISH_ARGS,
        train_dataset=eng_train_ds,
        eval_dataset=eng_val_ds,
        compute_metrics=ba.compute_metrics
    )
    eng_trainer.train()

    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []

    for log in eng_trainer.state.log_history:
        if "train_loss" in log:
            train_losses.append(log["train_loss"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
        if "eval_accuracy" in log:
            eval_accs.append(log["eval_accuracy"])
        if "eval_f1" in log:
            eval_f1s.append(log["eval_f1"])
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    eval_accs = np.array(eval_accs)
    eval_f1s = np.array(eval_f1s)
    mets = [train_losses, eval_losses, eval_accs, eval_f1s]
    print(mets)
    names = ["Train_loss", "Eval_loss", "Eval_acc", "Eval_f1"]
    mets = dict(zip(names, mets))
    plot = ba.metrics_plot(**mets)
    plot.savefig(os.path.join("plots", f"English_RoBERT_Final.png"))

    multil_model = XLMRobertaForSequenceClassification.from_pretrained(MULTILANG_MODEL_NAME, num_labels=3).to(ba.DEVICE)
    multil_trainer = Trainer(
        model=multil_model,
        args=MULTIL_ARGS,
        train_dataset=mult_train_ds,
        eval_dataset=mult_val_ds,
        compute_metrics=ba.compute_metrics
    )
    multil_trainer.train()

    train_losses = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []

    for log in multil_trainer.state.log_history:
        if "train_loss" in log:
            train_losses.append(log["train_loss"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
        if "eval_accuracy" in log:
            eval_accs.append(log["eval_accuracy"])
        if "eval_f1" in log:
            eval_f1s.append(log["eval_f1"])
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    eval_accs = np.array(eval_accs)
    eval_f1s = np.array(eval_f1s)
    mets = [train_losses, eval_losses, eval_accs, eval_f1s]
    print(mets)
    names = ["Train_loss", "Eval_loss", "Eval_acc", "Eval_f1"]
    mets = dict(zip(names, mets))
    plot = ba.metrics_plot(**mets)
    plot.savefig(os.path.join("plots", f"MultilanguageRoBERT_Final.png"))
