import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def LoadData(path_to_file):
    data = pd.read_csv(path_to_file)
    return data

def SplitData(data, test_size = 0.1):
    df_train, df_test = train_test_split(data, test_size=0.1, shuffle=False)
    df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=False)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_valid

class SCS_dataset(Dataset):

    def __init__(self, df, tokenizer, max_len):
        choice_pool = ['b', 'c', 'd', 'e']
        X = list(df['a'])
        train_len = len(list(df['a']))
        for i in range(train_len):
            rand_choice = random.choice(choice_pool)
            X.append(df[rand_choice][i])
        
        Y1 = [1]*train_len
        Y2 = [0]*train_len
        Y = [*Y1, *Y2]
        self.tokenizer = tokenizer
        self.contents = X
        self.targets = Y
        self.max_len = max_len
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, item):
        content = str(self.contents[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        content,
        max_length=self.max_len,
        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
        truncation=True
        )
        return {
            'content': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SCS_dataset(df, tokenizer, max_len)
    return DataLoader(ds, batch_size = batch_size, shuffle=True, num_workers=2)

class ProblemClassifier(nn.Module):

    def __init__(self, n_classes):
        super(ProblemClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bert2out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.drop = nn.Dropout(p=0.2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        class_space = self.bert2out(output.pooler_output)
        class_drop = self.drop(class_space)
        class_scores = self.logsoftmax(class_drop)
        return class_scores

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)

        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

        preds = torch.argmax(outputs, dim=-1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(outputs, dim=-1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def test_model(model, df, tokenizer, max_length):
    model = model.eval()

    test_len = len(list(df['a']))
    targets = torch.tensor(df['target'])
    preds = []
    option_domain = ['a','b','c','d','e']
    correct_predictions = 0
    with torch.no_grad():
        for i in range(test_len):
            input_ids = torch.zeros(len(option_domain), max_length, dtype=torch.long)
            attention_mask = torch.zeros(len(option_domain), max_length, dtype=torch.long)
            for j in range(len(option_domain)):
                encoding = tokenizer.encode_plus(
                df[option_domain[j]][i],
                max_length=max_length,
                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',  # Return PyTorch tensors
                truncation=True
                )
                input_ids[j,:] = encoding['input_ids']
                attention_mask[j,:] = encoding['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            pred_cols = torch.argmax(outputs, dim=0)
            preds.append(pred_cols[1].item())
        preds = torch.tensor(preds)
        print(preds)
        correct_predictions += torch.sum(preds == targets)
        print(preds == targets)

    return correct_predictions.double() / test_len