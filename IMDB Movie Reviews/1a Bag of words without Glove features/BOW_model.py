import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, t): #x is a list with batchsize by sequence length
    
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = torch.LongTensor(x[i]) #single sequence of token ids
            embed = self.embedding(lookup_tensor) #embed is a tensor with sequence by hidden units
            embed = embed.mean(dim=0) #a review becomes a 300 by 1 vector
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],t), h[:,0]