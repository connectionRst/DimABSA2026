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
import json
from typing import List, Dict, Literal, overload
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

from scipy.stats import pearsonr
from tqdm import tqdm
import math
import re
import requests


def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

# %% [markdown]
# ### First, visit the [DimABSA2006](https://github.com/DimABSA/DimABSA2026) repository, check the task-dataset.

# %% [markdown]
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
# 

# %%
device = "cuda"

#task config
subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "laptop" #change what domain you want to test

train_url = f"../../task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
predict_url = f"../../task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"

# model config
model_name = f"/home/na/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B" # change your transformer model
# model_name = "/home/na/.cache/modelscope/hub/models/google-bert/bert-base-multilingual-cased"
lr = 1e-5 # learning rate
epochs = 5
batchsize = 4

train_raw = load_jsonl(train_url)
predict_raw = load_jsonl(predict_url)

# %% [markdown]
# another transformer models you can try:
# 1. roberta-large
# 2. roberta-base
# 3. bert-base-multilingual-uncased
# 
# more models please visit [huggingface](https://huggingface.co/models)

# %%
#==== step 1 load the data ====
# you can change the env for your task.
# train data should have the VA labels, predit data without VA labels

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

train_df = jsonl_to_df(train_raw)
predict_df = jsonl_to_df(predict_raw)

# split 10% for dev
train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

# %% [markdown]
# ### Display the dataframe

# %%
from IPython.display import display, Markdown

display(Markdown(f"### {subtask}_{lang}_{domain} train_df"))
display(train_df.head())

display(Markdown(f"### {subtask}_{lang}_{domain} dev_df"))
display(dev_df.head())

display(Markdown(f"### {subtask}_{lang}_{domain} predict_df"))
display(predict_df.head())

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
    def __init__(self, dataframe):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = f"{self.aspects[idx]}: {self.sentences[idx]}"
        return {
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            "text": text
        }

batchsize = locals().get("batchsize", 64)
# convert to Dataset and Dataloader
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = VADataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

dev_dataset = VADataset(dev_df)
dev_loader = DataLoader(dev_dataset, batch_size=batchsize, shuffle=True)


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
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence + Arousal

    def forward(self, **tok):
        outputs = self.backbone(**tok)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        x = self.dropout(cls_output)
        return self.reg_head(x)



def toker(text, tokenizer=tokenizer):
    """
    Wrapper for tokenizer to generate batch_dict easily
    maybe need to rename to 'get_batch_dict'
    """
    return tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

def train_epoch(model, dataloader, optimizer, loss_fn):
    global device
    model = model.to(device)
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        labels = batch["labels"].to(device)
        batch_dict = toker(batch["text"]).to(device)

        optimizer.zero_grad()
        outputs = model(**batch_dict)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, loss_fn):
    model = model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            batch_dict = toker(batch["text"]).to(device)

            outputs = model(**batch_dict)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training bert on your data
model = TransformerVARegressor(model_name, dropout = 0)
lr = locals().get("lr", 1e-5)
epochs = locals().get("epochs", 50)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()  

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    val_loss = eval_epoch( model, dev_loader, loss_fn)
    print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

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
    type: Literal["dev"] = "dev"
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
            for batch in tqdm(dataloder):
                # input_ids = batch["input_ids"].to(device)
                # attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()
                batch_dict = toker(batch["text"])
                outputs = model(**batch_dict).cpu().numpy()
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
                outputs = model(**batch).cpu().numpy()
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

print("get_prd(dev) start")
pred_v, pred_a, gold_v, gold_a = get_prd(model, dev_loader, type="dev")
eval_score = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
print(f"{model_name} dev_eval: {eval_score}")

# %% [markdown]
# ### Step 5: Save and submit prediction results
# 
# - Define helper `df_to_jsonl`:
#   - Sort by ID number.
#   - Group rows by ID.
#   - Save predictions in JSONL format (`ID`, `Aspect_VA`).
# - Run the model on the predict sets (laptop & restaurant).
# - Fill in predicted Valence/Arousal values.
# - Export three JSONL files:
# 
# 
# 
# 
#   - `pred_eng_laptop.jsonl`
#   - `pred_eng_restaurant.jsonl`
#   - `pred_zho_laptop.jsonl`
# - These files can be uploaded as the final submission.
# 

# %% [markdown]
# ### File Naming Guidelines
# When submitting your predictions on the Codabench task page:
# 
# Decide the target language(s) and domain(s). Each submission file corresponds to one language-domain combination.
# For each language-domain combination, name the file pred_[lang_code]_[domain].jsonl, where
# - [lang_code] represents a 3-letter language code, and
# - [domain] represents a domain.
# For example, Hausa predictions for the movie domain should be named pred_hau_movie.jsonl.
# If submitting for multiple languages or domains, submit one prediction file per language-domain combination. For example, submitting for multiple languages or domains would look like this:
# ```plaintext
# subtask_1
# ├── pred_eng_restaurant.jsonl
# ├── pred_eng_laptop.jsonl
# └── pred_zho_laptop.jsonl

# %%
#==== step 5 save & submit your predict results ====
def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

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

pred_dataset = VADataset(predict_df, tokenizer)
pred_loader = DataLoader(pred_dataset, batch_size=batchsize, shuffle=True)
pred_v, pred_a, = get_prd(model, pred_loader,type="pred")

predict_df["Valence"] = pred_v
predict_df["Arousal"] = pred_a


df_to_jsonl(predict_df, f"pred_{lang}_{domain}.jsonl")

# %% [markdown]
# ### Download the submit files

# %%
import os
import shutil
import zipfile

# Create the folder subtask if it does not exist
os.makedirs(subtask, exist_ok=True)

# Move the three files into the subtask folder
for fname in [f"pred_{lang}_{domain}.jsonl"]:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join(subtask, fname))

# Create a zip file named "submit.zip" containing the folder subtask
with zipfile.ZipFile(f"{subtask}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files_in_dir in os.walk(subtask):
        for file in files_in_dir:
            path = os.path.join(root, file)
            # Keep folder structure inside the zip
            zf.write(path, os.path.relpath(path, "."))

# Download the created zip file to local machine

# %% [markdown]
# ### Conclusion
# 
# In this notebook, we walked through the full pipeline for **Dimensional Aspect Sentiment Regression (DimASR)**:
# 
# 1. **Load data**: Import the competition JSONL files, split train/dev sets, and convert to DataFrames.  
# 2. **Build dataset & dataloaders**: Define a custom `VADataset` to tokenize text and prepare `[Valence, Arousal]` labels.  
# 3. **Train & evaluate**: Train BERT-based regressors and check model performance on the dev sets using PCC and RMSE metrics.  
# 4. **Predict & submit**: Run the trained models on the prediction sets, generate VA scores, and save results as JSONL for submission.  
# 
# This pipeline ensures that your model is trained, validated, and ready for competition submission. You can further improve results by tuning hyperparameters, trying different pretrained models, or applying data augmentation strategies.
# 


