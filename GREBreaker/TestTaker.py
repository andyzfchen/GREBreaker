import numpy as np
import pandas as pd
from ast import literal_eval

#import transformers
from transformers import BertModel, BertTokenizer, BertForMaskedLM  #, AdamW, get_linear_schedule_with_warmup
import torch

from sklearn.model_selection import train_test_split

from torch import nn, optim
#from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TestTaker(object):
  def __init__(self, data_file_path, model_name=None):
    self.RANDOM_SEED = 42
    np.random.seed(self.RANDOM_SEED)
    torch.manual_seed(self.RANDOM_SEED)

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device set: ", self.device, ".")

    if model_name is None:
      self.PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    else:
      self.PRE_TRAINED_MODEL_NAME = model_name

    self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    self.model = BertForMaskedLM.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    self.model.eval()

    print("Loading dataset from filepath: ", data_file_path)
    df = pd.read_csv(data_file_path)
    print("Columns in dataset: ", df.columns)
    self.choices = [ "a)", "b)", "c)", "d)", "e)" ]

    # creating train, eval, test datasets
    print("Preparing train, evaluation, and test datasets.")
    self.df_train, self.df_test = train_test_split(df, test_size=0.5, random_state=self.RANDOM_SEED)
    self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.1, random_state=self.RANDOM_SEED)
    print("Train size: ", self.df_train.shape)
    print("Val size: ", self.df_val.shape)
    print("Test size: ", self.df_test.shape)

    print()


  def predict_masked_sentence(self, text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = self.tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(self.device)

    # Predict all tokens
    with torch.no_grad():
      outputs = self.model(tokens_tensor)
      predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    predicted_words = []
    for i, pred_idx in enumerate(top_k_indices):
      predicted_token = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]
      token_weight = top_k_weights[i]
      print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
      predicted_words.append(predicted_token)

    return predicted_words


  def set_glove_embedding(self, d_dim=50):
    assert d_dim in [ 50, 100, 200, 300 ]

    filenameGlove = "glove.6B."+str(d_dim)+"d"
    gloveEmbedding = { "<PAD>": np.zeros(200), "UNKA": np.random.rand(200) }
    self.gloveEmbeddingIdx = { "<PAD>": 0, "UNKA": 1 }
    self.nGloveEmbeddingDim = 0

    fileGlove = open("../glove_data/"+filenameGlove+".txt", "r")

    try:
      print("Loading "+filenameGlove+" weights from cache.")
      gloveEmbeddingMatrix = np.load("../glove_data/"+filenameGlove+"_weights.npy")
      gloveEmbeddingMatrix = torch.tensor(gloveEmbeddingMatrix.astype(float))
      for ii, line in enumerate(fileGlove):
        try:
          lineTokens = line.split(" ")
          self.gloveEmbeddingIdx[lineTokens[0]] = len(self.gloveEmbeddingIdx)
          if self.nGloveEmbeddingDim == 0:
            self.nGloveEmbeddingDim = len(lineTokens) - 1
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
          self.gloveEmbeddingIdx[lineTokens[0]] = len(self.gloveEmbeddingIdx)
          if self.nGloveEmbeddingDim == 0:
            self.nGloveEmbeddingDim = len(lineTokens) - 1
        except:
          print("Issue with line ", ii, ".")
          print(lineTokens)
      gloveEmbedding["<PAD>"] = np.zeros(self.nGloveEmbeddingDim)
      gloveEmbedding["UNKA"] = np.random.rand(self.nGloveEmbeddingDim)
      gloveEmbeddingMatrix = np.array(list(gloveEmbedding.values()))
      np.save("../glove_data/"+filenameGlove+"_weights.npy", gloveEmbeddingMatrix)
      gloveEmbeddingMatrix = torch.tensor(gloveEmbeddingMatrix.astype(float))

    nGloveVocab = len(self.gloveEmbeddingIdx)
    print("Size of embedding matrix: ", gloveEmbeddingMatrix.size())

    #model = RNNTagger(self.nGloveEmbeddingDim, hiddenSize, nTag)
    #opt = optim.Adam(model.parameters(), lr=0.001)
    #lossFunc = nn.NLLLoss()

    paddingIdx = self.gloveEmbeddingIdx["<PAD>"]
    self.wordEmbedding = nn.Embedding(num_embeddings=nGloveVocab, embedding_dim=self.nGloveEmbeddingDim, padding_idx=paddingIdx)
    self.wordEmbedding.load_state_dict({'weight': gloveEmbeddingMatrix})

    print("GLoVe embedding loaded successfully.")
    print()


  def evaluate_sentence(self, df, ii):
    print("Sentence: ", df["question"][ii])

    # embeddings for choice words
    choice_words = []
    choice_words_idx = []
    print("Choices:")
    for choice in self.choices:
      word = literal_eval(df[choice][ii])[0][-1]
      choice_words.append(word)
      choice_words_idx.append(self.gloveEmbeddingIdx[word.lower()])
      print(choice, word)

    choice_words_idx = torch.tensor(np.array(choice_words_idx)).int()
    choice_word_embeddings = self.wordEmbedding(choice_words_idx).view(-1, 1, self.nGloveEmbeddingDim)
    print("Correct choice: ", df["ans"][ii])
    print()

    # embeddings for predicted words
    print("Predicted words:")
    predicted_words = self.predict_masked_sentence(df["question"][ii], top_k=5)
    predicted_words_idx = []
    for predicted_word in predicted_words:
      predicted_words_idx.append(self.gloveEmbeddingIdx[predicted_word.lower()])

    predicted_words_idx = torch.tensor(np.array(predicted_words_idx)).int()
    predicted_word_embeddings = self.wordEmbedding(predicted_words_idx).view(-1, 1, self.nGloveEmbeddingDim)

    for jj, embedding in enumerate(choice_word_embeddings):
      print("Closeness of ", choice_words[jj], ": ", torch.dot(predicted_word_embeddings[0].view(-1), embedding.view(-1)) / (torch.norm(predicted_word_embeddings[0])*torch.norm(embedding)) )

    print()


  def train(self):
    for ii in self.df_train.index:
      self.evaluate_sentence(self.df_train, ii)
      break
    

  def validate(self):
    for ii in self.df_val.index:
      self.evaluate_sentence(self.df_val, ii)


  def test(self):
    for ii in self.df_test.index:
      self.evaluate_sentence(self.df_test, ii)















