import numpy as np
import pandas as pd
from ast import literal_eval

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


# creating train, eval, test datasets
df_train, df_test = train_test_split(df, test_size=0.5, random_state=RANDOM_SEED)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=RANDOM_SEED)
print("Train size: ", df_train.shape)
print("Val size: ", df_val.shape)
print("Test size: ", df_test.shape)


# helper function for predicting masked word
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

    predicted_words = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
        predicted_words.append(predicted_token)

    return predicted_words



# importing GLoVe embedding
filenameGlove = "glove.6B.50d"
gloveEmbedding = { "<PAD>": np.zeros(200), "UNKA": np.random.rand(200) }
gloveEmbeddingIdx = { "<PAD>": 0, "UNKA": 1 }
nGloveEmbeddingDim = 0

fileGlove = open("../glove_data/"+filenameGlove+".txt", "r")

try:
  print("Loading "+filenameGlove+" weights from cache.")
  gloveEmbeddingMatrix = np.load("../glove_data/"+filenameGlove+"_weights.npy")
  gloveEmbeddingMatrix = torch.tensor(gloveEmbeddingMatrix.astype(float))
  for ii, line in enumerate(fileGlove):
    try:
      lineTokens = line.split(" ")
      gloveEmbeddingIdx[lineTokens[0]] = len(gloveEmbeddingIdx)
      if nGloveEmbeddingDim == 0:
        nGloveEmbeddingDim = len(lineTokens) - 1
    except:
      print("Issue with line ", ii, ".")
      print(lineTokens)
except:
  print("No cache found. Loading from txt and generating cache.")
  for ii, line in enumerate(fileGlove):
    try:
      lineTokens = line.split(" ")
      lineTokens[-1] = lineTokens[-1][:-1]  # removes newline character
      gloveEmbedding[lineTokens[0]] = np.array(lineTokens[1:]).astype(float)
      gloveEmbeddingIdx[lineTokens[0]] = len(gloveEmbeddingIdx)
      if nGloveEmbeddingDim == 0:
        nGloveEmbeddingDim = len(lineTokens) - 1
    except:
      print("Issue with line ", ii, ".")
      print(lineTokens)
  gloveEmbedding["<PAD>"] = np.zeros(nGloveEmbeddingDim)
  gloveEmbedding["UNKA"] = np.random.rand(nGloveEmbeddingDim)
  gloveEmbeddingMatrix = np.array(list(gloveEmbedding.values()))
  np.save(filenameGlove+"_weights.npy", gloveEmbeddingMatrix)
  gloveEmbeddingMatrix = torch.tensor(gloveEmbeddingMatrix.astype(float))

nGloveVocab = len(gloveEmbeddingIdx)
print("Size of embedding matrix: ", gloveEmbeddingMatrix.size())

#model = RNNTagger(nGloveEmbeddingDim, hiddenSize, nTag)
#opt = optim.Adam(model.parameters(), lr=0.001)
#lossFunc = nn.NLLLoss()

paddingIdx = gloveEmbeddingIdx["<PAD>"]
wordEmbedding = nn.Embedding(num_embeddings=nGloveVocab, embedding_dim=nGloveEmbeddingDim, padding_idx=paddingIdx)
wordEmbedding.load_state_dict({'weight': gloveEmbeddingMatrix})

print()

# running over train, eval, test datasets
for ii in df_train.index:
  print(df_train["question"][ii])

  # embeddings for choice words
  choice_words = []
  choice_words_idx = []
  for choice in choices:
    word = literal_eval(df_train[choice][ii])[0][0]
    choice_words.append(word)
    choice_words_idx.append(gloveEmbeddingIdx[word.lower()])

  choice_words_idx = torch.tensor(np.array(choice_words_idx)).int()
  print("Choice word indices: ", choice_words_idx)
  choice_word_embeddings = wordEmbedding(choice_words_idx).view(-1, 1, nGloveEmbeddingDim)

  # embeddings for predicted words
  predicted_words = predict_masked_sent(df_train["question"][ii], top_k=5)
  predicted_words_idx = []
  for predicted_word in predicted_words:
    predicted_words_idx.append(gloveEmbeddingIdx[predicted_word.lower()])

  predicted_words_idx = torch.tensor(np.array(predicted_words_idx)).int()
  print("Predicted word indices: ", predicted_words_idx)
  predicted_word_embeddings = wordEmbedding(predicted_words_idx).view(-1, 1, nGloveEmbeddingDim)

  print()
  break

print("Best prediction: ", predicted_words[0])
for ii, embedding in enumerate(choice_word_embeddings):
  print("Closeness of ", choice_words[ii], ": ", torch.dot(predicted_word_embeddings[0].view(-1), embedding.view(-1)) / (torch.norm(predicted_word_embeddings[0])*torch.norm(embedding)) )










