import numpy as np
import pandas as pd

import transformers
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
import torch

from sklearn.model_selection import train_test_split

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv("../sat_data/SAT_set_1blank.csv")
print(df.head())
choices = [ "a)", "b)", "c)", "d)", "e)" ]

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForMaskedLM.from_pretrained(PRE_TRAINED_MODEL_NAME)
model.eval()

sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

print(tokenizer.sep_token, tokenizer.sep_token_id)
print(tokenizer.cls_token, tokenizer.cls_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.unk_token, tokenizer.unk_token_id)
print(tokenizer.mask_token, tokenizer.mask_token_id)

encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=False, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

print(encoding.keys())

print(len(encoding['input_ids'][0]))
print(encoding['input_ids'][0])

print(len(encoding['attention_mask'][0]))
print(encoding['attention_mask'])

print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))

df_train, df_test = train_test_split(df, test_size=0.5, random_state=RANDOM_SEED)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=RANDOM_SEED)
print("Train size: ", df_train.shape)
print("Val size: ", df_val.shape)
print("Test size: ", df_test.shape)


def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

for ii in df_train.index:
  print(df_train["question"][ii])
  for choice in choices:
    print(df_train[choice][ii])
  predict_masked_sent(df_train["question"][ii], top_k=5)
  print()
  break










