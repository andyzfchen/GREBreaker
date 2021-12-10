import numpy as np
import pandas as pd
import tokenize
import re

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df1 = pd.read_csv("../scs_data/scs_training.csv")
print(df1.head())
print(df1.columns)
print(df1.shape)

#print("Making dataset with "+str(n_blank)+" blanks.")

n_row, n_col = df1.shape

questions = []
answers = []
choice_a = []
choice_b = []
choice_c = []
choice_d = []
choice_e = []
sentences = []

column_names = [ "a)", "b)", "c)", "d)", "e)" ]
choice_names = [ "a", "b", "c", "d", "e" ]
choice_names_temp = [ "a", "b", "c", "d", "e" ]
choices = [ choice_a, choice_b, choice_c, choice_d, choice_e ]

max_len = 0

for jj in df1.index:
  np.random.shuffle(choice_names_temp)

  mask_idx = np.where([word1 != word2 for word1, word2 in zip(df1["a"][jj].split(), df1["b"][jj].split())])[0][0]

  question = df1["a"][jj].split()
  question[mask_idx] = re.sub("[A-Za-z]+", "[MASK]", question[mask_idx])
  n_word = len(question)
  question = " ".join(question)
  questions.append(question)
  
  sentences.append(df1["a"][jj])

  answer = choice_names[np.where([word == "a" for word in choice_names_temp])[0][0]]
  answers.append(answer)
  
  for ii in range(len(choice_names)):
    tokens = df1[choice_names_temp[ii]][jj].split()
    assert len(tokens) == n_word
    try:
      choice = re.search("[A-Za-z]+", tokens[mask_idx])[0]
    except:
      for kk in range(len(choice_names)):
        print(df1[choice_names_temp[kk]][jj])
      print(tokens)
      print(tokens[mask_idx])
      exit()
    choices[ii].append([[choice]])

  if (jj+1)%500 == 0:
    print("Processed ", jj+1, " rows of ", n_row, ".")

data = {}
data["ans"] = answers
data["question"] = questions
data["sentence"] = sentences
for ii in range(len(choice_names)):
  data[column_names[ii]] = choices[ii]

df_out = pd.DataFrame(data, columns=data.keys())
df_out.to_csv("SCS_set_1blank.csv")
      
