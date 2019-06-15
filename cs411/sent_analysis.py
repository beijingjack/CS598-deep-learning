
import numpy as np
import os
import nltk
import itertools
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


word2id = np.load('imdb_dictionary_w2id.npy')

comment = "Overall this is a decent movie. The sound effect is great, the actors are really professional, and the actresses are beautiful. If the director can pay more attention to details, it would be better"
comment = nltk.word_tokenize(comment)
comment = [w.lower() for w in comment]

print (comment)

comment_token_ids = [word2id.item().get(token,-1)+1 for token in comment]
print (comment_token_ids)

x_input = []
x_input.append(comment_token_ids)

model = torch.load('BOW.model')
# model.cuda()
# model.eval()


pred = model(x_input)
pred = pred.data.item()

print (pred)

