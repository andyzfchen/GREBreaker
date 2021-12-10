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

DATA_FILE_PATH = "../sat_data/SAT_set_1blank.csv"
#DATA_FILE_PATH = "../scs_data/SCS_set_1blank.csv"
#DATA_FILE_PATH = "../501sc_data/501sc_set_1blank.csv"

tt = TestTaker(DATA_FILE_PATH)
tt.set_glove_embedding(50)
tt.test()
tt.init_train()
tt.train()


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





