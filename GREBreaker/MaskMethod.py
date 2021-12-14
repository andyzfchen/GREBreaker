#import numpy as np
#import pandas as pd
#from ast import literal_eval
#
#import transformers
#from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
#import torch
#
#from sklearn.model_selection import train_test_split
#
#from torch import nn, optim
#from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#
from TestTaker import TestTaker
import numpy as np
import os

data_files = [ 
  "../sat_data/SAT_set_1blank.csv",
  "../scs_data/SCS_set_1blank.csv",
  "../501sc_data/501sc_set_1blank.csv",
]

data_names = [
  "SAT",
  "SCS",
  "501sc",
]

glove_embedding_dims = [
  50,
  100,
  200,
  300,
]

#DATA_FILE_PATH = "../sat_data/SAT_set_1blank.csv"
#DATA_FILE_PATH = "../scs_data/SCS_set_1blank.csv"
#DATA_FILE_PATH = "../501sc_data/501sc_set_1blank.csv"

if not os.path.exists("../cache"):
  os.mkdir("../cache")
if not os.path.exists("../cache/word_prediction"):
  os.mkdir("../cache/word_prediction")

for ii, data_file in enumerate(data_files):
  for glove_embedding_dim in glove_embedding_dims:
    tt = TestTaker(data_file)
    tt.set_glove_embedding(glove_embedding_dim)
    tt.test()
    tt.init_train()
    train_acc, val_acc = tt.train()

    np.save("../cache/word_prediction/"+data_names[ii]+"_glove"+str(glove_embedding_dim)+"_train_acc.npy", train_acc)
    np.save("../cache/word_prediction/"+data_names[ii]+"_glove"+str(glove_embedding_dim)+"_val_acc.npy", val_acc)

    


'''
evaluation ideas:
- average predicted word vectors and compare to each choice
- compare each predicted word with each choice
- use larger dimensional glove

training ideas:
- use (1-best score word) as loss function to train BERT
- use (1-x) as loss function to train BERT
  - x = max( dot(correct answer, each predicted word) )
  - x = mean( dot(correct answer, each predicted word) )
'''





