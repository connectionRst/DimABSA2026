# %% [markdown]
# <figure>
#   <img src="https://raw.githubusercontent.com/shadowkshs/DimABSA2026/refs/heads/main/banner.png" width="100%">
# </figure>

# %% [markdown]
# # SemEval-2026 Task 3 (Track A: DimABSA, Track B: DimStance)
# # Subtask 1: Dimensional Aspect Sentiment Regression (DimASR)
#
# -----
#
# ## Starter Notebook
# Leveraging Pretrained Language Models for Dimensional Sentiment Regression
#
#
# ## Introduction:
#
# You are welcome to participate in our SemEval Shared Task!
#
# In this starter notebook, we will take you through the process of fine-tuning a pre-trained language model on a sample data to build a sentiment regressor. The notebook was adapted from a Hugginface implementation for such tasks.
#
# ### Outline:
#
# - Installation and importation of necessary libraries
# Setting up the project parameters.
# Running training and evaluation
# Before you start:
#
# - It is strongly advised that you use a GPU to speed up training. To do this, go to the "Runtime" menu in Colab, select "Change runtime type" and then in the popup menu, choose "GPU" in the "Hardware accelerator" box.
#
# ### NB:
#
# The codes in this notebook are provided to familiarize yourselves with fine-tuning language models for sentiment regression. You may extend and (or) modify as appropriate to obtain competitive performances.
#
# ### Languages and Domains:
# #### Track A: Subtask 1
# - eng_restaurant
# - eng_laptop
# - jpn_hotel
# - jpn_finance
# - rus_restaurant
# - tat_restaurant
# - ukr_restaurant
# - zho_restaurant
# - zho_laptop
# #### Track B: Subtask 1
# - deu-stance
# - eng-stance
# - hau-stance
# - kin-stance
# - swa-stance
# - twi-stance
#
#
# ### Model:
# This Starter Notebook uses the bert-base-multilingual-cased pretrained model, developed by Google. The model was trained with a masked language modeling (MLM) objective on the top 104 languages with the largest Wikipedia presence. You can find the model here: https://huggingface.co/google-bert/bert-base-multilingual-cased
#
# If your target language is not included in the common set supported by this model, you can search for a more suitable model on Hugging Face: https://huggingface.co/models
#
#

# %%
from typing import List, Dict, Literal, overload
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from scipy.stats import pearsonr
from tqdm import tqdm
import requests

import json
import re
import math
import argparse
import os
import itertools

PREFIX = "../../task-dataset/track_a"
MODEL_PREFIX = os.path.expanduser("~/.cache/modelscope/hub/models/")

domain_lang = {
    "hotel": ["jpn"],
    "restaurant": ["eng", "rus", "tat", "ukr", "zho"],
    "laptop": ["eng", "zho"],
    "finance": ["jpn", "zho"]  # watch out for this
}

# TODO auto generate from domain_lang, but python grammar sucks
lang_domain = {
    "jpn": ["hotel", "finance"],
    "eng": ["laptop", "restaurant"],
    "zho": ["laptop", "restaurant", "finance"],
    "rus": ["restaurant"],
    "tat": ["restaurant"],
    "ukr": ["restaurant"],
}

def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

# these two func are only used in task1
def jsonl_to_df(data):
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    return df

def df_to_jsonl(df, out_path):
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# %% [markdown]
# ### Step 2: Build Dataset and DataLoader
#
# - Define a custom `VADataset` class for PyTorch:
#   - Joins Aspect + Text into a single input string.
#   - Uses BERT tokenizer to create `input_ids` and `attention_mask`.
#   - Returns `[Valence, Arousal]` labels as float tensor.
# - Convert the processed DataFrames into PyTorch `Dataset` objects.
# - Wrap them with `DataLoader` for mini-batch training and evaluation.

# %%
#==== Dataset ====
class VADataset(Dataset):
    '''
    A PyTorch Dataset for Valence–Arousal regression.

    - Combines aspect and text into a single input (e.g., "keyboard: The keyboard is good").
    - Tokenizes the input using a HuggingFace tokenizer.
    - Returns:
        * input_ids: token IDs, shape [max_len]
        * attention_mask: mask, shape [max_len]
        * labels: [Valence, Arousal], shape [2], float tensor

    Args:
        dataframe (pd.DataFrame): must contain "Text", "Aspect", "Valence", "Arousal".
        tokenizer: HuggingFace tokenizer.
        max_len (int): max sequence length.
    '''
    def __init__(self, dataframe, tokenizer, tok_max_len):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        self.tokenizer = tokenizer
        self.tok_max_len = tok_max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = f"{self.aspects[idx]}: {self.sentences[idx]}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tok_max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# %% [markdown]
# ### Step 3: Build and Train TransformerVARegressor
#
# - Define **`TransformerVARegressor`**:
#   - Uses pretrained Transformer (e.g. BERT) as backbone.
#   - Adds dropout and linear layer to predict **Valence** and **Arousal**.
#
# - Implement helper methods:
#   - `train_epoch`: one training pass with optimizer and loss.
#   - `eval_epoch`: validation pass without gradient updates.
#
# - Set training parameters:
#   - `lr = 1e-5`, `epochs = 5`, `loss_fn = MSELoss`.
#
# - Run training loop:
#   - For each epoch, print training and validation loss to monitor progress.
#

# %%
#====step 3 build your model ====
class TransformerVARegressor(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores.

    - Uses a pretrained BERT backbone to encode text.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Adds a dropout layer and a linear head to output 2 values: [Valence, Arousal].
    - Includes helper methods for one training epoch and one evaluation epoch.

    Args:
        model_name (str): HuggingFace model name, default "bert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.

    Methods:
        train_epoch(dataloader, optimizer, loss_fn, device):
            Train the model for one epoch.
            Returns average training loss.

        eval_epoch(dataloader, loss_fn, device):
            Evaluate the model for one epoch (no gradient).
            Returns average validation loss.
    '''
    def __init__(self, basemodel, dropout=0):
        super().__init__()
        self.backbone = basemodel
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence + Arousal

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        x = self.dropout(cls_output)
        return self.reg_head(x)

# TODO refractor to pass tokenizer
def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, miniters = 10):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# %% [markdown]
# ### Step 4: Evaluate model performance on dev set
#
# - Define helper function `get_prd`:
#   - For **dev**: get both predictions and gold labels.
#   - For **pred**: only get predictions (no gold labels).
# - Define `evaluate_predictions_task1`:
#   - Compute Pearson Correlation Coefficient (PCC) for Valence (V) and Arousal (A).
#   - Compute normalized RMSE for combined VA score.
# - Run evaluation on laptop and restaurant dev sets.
# - Print metrics to check how well the models perform.
#

# %%
#==== step 4 use dev data to check your model's performance ====
@overload
def get_prd(
    model: TransformerVARegressor,
    dataloder,
    type: Literal["dev"]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def get_prd(
    model: TransformerVARegressor,
    dataloder,
    type: Literal["pred"]
) -> tuple[np.ndarray, np.ndarray]: ...
def get_prd(model: TransformerVARegressor, dataloder, type = "dev"):
    global accl
    if type == "dev":
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
                all_labels.append(labels)
        preds = np.vstack(all_preds)
        lables = np.vstack(all_labels)

        pred_v = preds[:,0]
        pred_a = preds[:,1]

        gold_v = lables[:,0]
        gold_a = lables[:,1]

        return pred_v, pred_a, gold_v, gold_a

    elif type == "pred":
        all_preds = []
        with torch.no_grad():
            for batch in dataloder:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask).cpu().numpy()
                all_preds.append(outputs)
        preds = np.vstack(all_preds)

        pred_v = preds[:, 0]
        pred_a = preds[:, 1]

        return pred_v, pred_a

def rmse_pairwise(pred_a, pred_v, gold_a, gold_v):
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]
    total_sq_error = sum((pv - gv)**2 + (pa - ga)**2 for gv,pv,ga,pa in zip(gold_v, pred_v, gold_a, pred_a))
    rmse_va = math.sqrt(total_sq_error / len(gold_v))

    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }

def rmse_concat(pred_a, pred_v, gold_a, gold_v):
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]

    gold_va = list(gold_v) + list(gold_a)
    pred_va = list(pred_v) + list(pred_a)
    total_sq_error = [(a - b)**2 for a,b in zip(gold_va, pred_va)]
    rmse_va = math.sqrt(sum(total_sq_error) / len(gold_v))
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }

def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v, is_norm = False):
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range.")
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]

    gold_va = list(gold_v) + list(gold_a)
    pred_va = list(pred_v) + list(pred_a)
    def rmse_norm(gold_va, pred_va, is_normalization = True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        if is_normalization:
            return math.sqrt(sum(result)/len(gold_v))/math.sqrt(128)
        return math.sqrt(sum(result)/len(gold_v))
    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }

# %% [markdown]
# ### Step 5: Save and submit prediction results
#
# - Define helper `df_to_jsonl`:
#   - Sort by ID number.
#   - Group rows by ID.
#   - Save predictions in JSONL format (`ID`, `Aspect_VA`).
# - Run the model on the predict sets (laptop & restaurant).
# - Fill in predicted Valence/Arousal values.

# %%
#==== step 5 save & submit your predict results ====
def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

# %%

# TODO call main function here, not directly implement here
# Training bert on your data
def mian_train():
    lowest_rmseva = 5
    best_epoch = 0
    def _get_prd_dev(*, epoch = 0):
        nonlocal lowest_rmseva, best_epoch
        # print("get_prd(dev) start")
        pred_v, pred_a, gold_v, gold_a = get_prd(model, dev_loader, type="dev")
        eval_score = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
        print(f"model: {model_name} dev_eval: {eval_score}")
        if (eval_score["RMSE_VA"] < lowest_rmseva):
            lowest_rmseva = eval_score["RMSE_VA"]
            best_epoch = epoch
            print("this is best")

    base_model = AutoModel.from_pretrained(model_path)
    model = TransformerVARegressor(base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    train_dataset = VADataset(train_df, tokenizer, tok_max_len)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    dev_dataset = VADataset(dev_df, tokenizer, tok_max_len)
    dev_loader = DataLoader(dev_dataset, batch_size=batchsize, shuffle=True)

    # lr = locals().get("lr", 1e-4)
    # epochs = locals().get("epochs", 50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # okay do benchmark on every epoch
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        print(f"model: {model_name} Epoch: {epoch+1}: train={train_loss:.4f}")
        val_loss = eval_epoch(model, dev_loader, loss_fn)
        print(f"model: {model_name} Epoch: {epoch+1}: val={val_loss:.4f}")
        _get_prd_dev(epoch=epoch)

    base_model.save_pretrained(f"./models/{lang}/{model_name}")
    tokenizer.save_pretrained(f"./models/{lang}/{model_name}")

def mian_infer():
    base_model = AutoModel.from_pretrained(model_path)
    model = TransformerVARegressor(base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    os.makedirs(f"./outputs/{model_name}/subtask_1", exist_ok=True)

    for domain, predict_df in predict_dfs:
        pred_dataset = VADataset(predict_df, tokenizer, tok_max_len)
        pred_loader = DataLoader(pred_dataset, batch_size=batchsize, shuffle=True)
        pred_v, pred_a, = get_prd(model, pred_loader, type="pred")

        predict_df["Valence"] = pred_v
        predict_df["Arousal"] = pred_a

        df_to_jsonl(predict_df, f"./outputs/{model_name}/subtask_1/pred_{lang}_{domain}.jsonl")

# main():
# TODO may need to put lang before domain
parser = argparse.ArgumentParser()
parser.add_argument('train_or_infer', default="infer")
# change what domain you want
# parser.add_argument('domain', nargs='?', default="laptop")
# change the language you want to test
parser.add_argument('lang', nargs='?', default="jpn")
parser.add_argument('--model-name', default="jhu-clsp/mmBERT-base")  # TODO rename to follow api
# unsloth: use 2e-4 if the data is small
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=bool, default=False)
args = parser.parse_args()
assert args.train_or_infer in ("train", "infer")

# task config
task = 1
lang = args.lang
assert lang in lang_domain
# model config
model_name = args.model_name
# choose your encoding model
supported_model_name = (
    "google-bert/bert-base-multilingual-cased",
    "ai-modelscope/roberta-large",
    "answerdotai/ModernBERT-large",
    "jhu-clsp/mmBERT-base"
)
assert model_name in supported_model_name
if args.train_or_infer == "train":
    model_path = MODEL_PREFIX + model_name
elif args.train_or_infer == "infer":  # TODO more elegant way to read saved model
    model_path = os.path.abspath(f"./models/{lang}/{model_name}")
lr = args.lr
epochs = args.epochs
batchsize = 32
tok_max_len = 512
print(f"{lang=}, {model_path=}")
print(f"lr: {lr}, epochs: {epochs}, batchsize: {batchsize}")
device = "cuda"

# ### Step 1: Load the competition data
#
# - Read JSONL files (train/dev/predict) into Colab.
# - Train files contain Valence–Arousal (VA) labels.
# - Predict files have no VA labels.
# - This script:
#   1. Loads the JSONL data.
#   2. Splits 10% of train data as dev set.
#   3. Converts JSONL into DataFrames (ID, Text, Aspect, Valence, Arousal).
#   4. Prints the first few rows for checking.

# FIXME would the model being too small to fit all train data?
# TODO split these logic to their main?
# train_urls = [f"{PREFIX}/subtask_1/{lang}/{lang}_{domain}_train_alltasks.jsonl" for domain in domain_lang.keys() for lang in domain_lang[domain]]
train_urls = [f"{PREFIX}/subtask_1/{lang}/{lang}_{domain}_train_alltasks.jsonl" for domain in lang_domain[lang]]
train_raw = []  # cause python flattening is sh*t, would rather use concat
for url in train_urls: train_raw += load_jsonl(url)
train_df = jsonl_to_df(train_raw)
# split 10% for dev
train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

# for one lang do all domain pred
predict_urls = [ (domain, f"{PREFIX}/subtask_1/{lang}/{lang}_{domain}_dev_task{task}.jsonl") for domain in lang_domain[lang] ]
predict_raws = [ (dom, load_jsonl(url)) for (dom, url) in predict_urls ]
predict_dfs = [ (dom, jsonl_to_df(raw)) for (dom, raw) in predict_raws ]


# ### Display the dataframe
# TODO move this part to main
from IPython.display import display, Markdown

display(Markdown(f"### subtask_1_{lang}_alldomain train_df"))
display(train_df.head())

display(Markdown(f"### subtask_1_{lang}_alldomain dev_df"))
display(dev_df.head())

display(Markdown(f"### subtask_1_{lang}_alldomain predict_df"))
for _, prd_df in predict_dfs:
    display(prd_df.head())

if args.train_or_infer == "train":
    mian_train()
elif args.train_or_infer == "infer":
    mian_infer()
else:
    assert args.train_or_infer in ("train", "infer")
