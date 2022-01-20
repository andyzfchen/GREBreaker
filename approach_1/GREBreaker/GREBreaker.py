import argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, sentence):
        # using padding
        #sentence = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        #lstm_out, _ = self.lstm(sentence)
        #X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # no padding
        lstm_out, _ = self.lstm(sentence)
        #lstm_out = self.relu(lstm_out)
        #lstm_out = self.tanh(lstm_out)

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores

def train(training_file):
    assert os.path.isfile(training_file), 'Training file does not exist'

    # #######################
    # Your code starts here
    # #######################

    # hyperparameters
    filenameGlove = "../glove_data/glove.6B.50d.txt"
    hiddenSize = 100      # LSTM units inside a layer
    nEpoch = 5            # training epochs
    nSample = 5000        # samples per epoch

    # preparing LSTM model and GloVe embedding
    print("Preparing LSTM model and GloVe embedding.")
    gloveEmbedding = { "<PAD>": np.zeros(200), "UNKA": np.random.rand(200) }
    gloveEmbeddingIdx = { "<PAD>": 0, "UNKA": 1 }
    nGloveEmbeddingDim = 0
    fileGlove = open(filenameGlove, "r")
    try:
      print("Loading "+filenameGlove+" weights from cache.")
      gloveEmbeddingMatrix = np.load(filenameGlove+"_weights.npy")
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
    
    model = RNNTagger(nGloveEmbeddingDim, hiddenSize, nTag)
    opt = optim.Adam(model.parameters(), lr=0.001)
    lossFunc = nn.NLLLoss()

    paddingIdx = gloveEmbeddingIdx["<PAD>"]
    model.wordEmbedding = nn.Embedding(num_embeddings=nGloveVocab, embedding_dim=nGloveEmbeddingDim, padding_idx=paddingIdx)
    model.wordEmbedding.load_state_dict({'weight': gloveEmbeddingMatrix})

    print()


    # actual training
    print("Performing the actual training.")
    model.train(True)
    for jj in range(nEpoch):
      print("Epoch number ", jj+1, " of ", nEpoch, ".")
      print("Learning rate: ", opt.param_groups[0]['lr'])

      fileIn = open(training_file, "r", encoding='windows-1252')
      sampleLoss = 0
      for ii, line in enumerate(fileIn):
        lineTokens = line.split(" ")
        lineTokens[-1] = lineTokens[-1][:-1]  # removes newline character
        [lineTokens.pop(jj) for jj, token in reversed(list(enumerate(lineTokens))) if token == ""]
        words = torch.tensor(np.array([gloveEmbeddingIdx[word.lower()] if word in gloveEmbeddingIdx.keys() else gloveEmbeddingIdx["UNKA"] for word in lineTokens[::2]])).int()
        pos = torch.tensor(np.array([tags[token] for token in lineTokens[1::2]]))

        # forward pass
        words = model.wordEmbedding(words).view(-1, 1, model.embedding_dim)
        predPos = model(words)

        # backprop and optimize
        loss = lossFunc(predPos, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sampleLoss += loss.item()
          
        if (ii+1) % 500 == 0:
          print("Trained ", ii+1, " lines.")
          print("Average Loss:  ", sampleLoss/nSample)
          sampleLoss = 0

        if ii > nSample:
          break
      fileIn.close()
      print()


    print("Training is complete.")
    print()


