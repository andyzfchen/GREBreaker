import numpy as np
import pandas as pd
import tokenize
import re

df = pd.read_csv("../sat_data/SAT_set_filled.csv")
print(df.head())
print(df.columns)
print(df.shape)

for n_blank in range(1,2):
  print("Making dataset with "+str(n_blank)+" blanks.")

  df1 = df[df["blanks"]==n_blank][["ans","question"]]
  n_row, n_col = df1.shape

  choice_a = []
  choice_b = []
  choice_c = []
  choice_d = []
  choice_e = []
  sentences = []

  choice_names = [ "a)", "b)", "c)", "d)", "e)" ]
  choices = [ choice_a, choice_b, choice_c, choice_d, choice_e ]

  max_len = 0

  for jj in df1.index:
    mask_idx = [ii for ii, ss in enumerate(df1["question"][jj].split()) if "[MASK]" in ss]
    n_token = len(df1["question"][jj].split())

    if n_token > max_len:
      max_len = n_token

    for kk in range(len(choices)):
      temp_tokens = df[choice_names[kk]][jj].split()

      temp_choice = [temp_tokens[ii:(ii+1+(len(temp_tokens)-n_token))] for ii in mask_idx]

      choices[kk].append([ [re.search("[a-z]+", word)[0] for word in choice] for choice in temp_choice])

      if df1["ans"][jj] == choice_names[kk][0]:
        sentences.append(df[choice_names[kk]][jj])

  for kk in range(len(choices)):
    df1[choice_names[kk]] = choices[kk]

  df1["sentence"] = sentences

  df1.reset_index(drop=True, inplace=True)
  print(df1.head())
  print(df1.shape)
  print("Max length: ", max_len)

  df1.to_csv("SAT_set_"+str(n_blank)+"blank.csv")

