import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys

from RNN_model import RNN_model

imdb_dictionary = np.load('/Users/liuchunlei/Desktop/IMDB Movie reviews/preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 # imdb_dictionary.shape[0], 8000 can reduce the number of weights without igonoring too much unique tokens

x_train = []
with open('/Users/liuchunlei/Desktop/IMDB Movie reviews/preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1 #the first 12500 are positive reviews, the next 12500 are negative reviews, 50000 are unlabelled reviews

x_test = []
with open('/Users/liuchunlei/Desktop/IMDB Movie reviews/preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1


vocab_size += 1
batch_size = 200
no_of_epochs = 6
model = RNN_model(vocab_size,300)
# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
L_Y_train = len(y_train) #25000
L_Y_test = len(y_test)


train_loss = []
train_accu = []
test_accu = []


for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]] #batchsize by different sequence length
        sequence_length = 100
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int) #batchsize by same sequence lenght 100
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]
#         x_input = Variable(torch.FloatTensor(x_input)).cuda()
#         target = Variable(torch.FloatTensor(y_input)).cuda()
        data = torch.LongTensor(x_input)
        target = torch.FloatTensor(y_input)
        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        loss.backward()

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data[0]
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))
    
    if((epoch+1)%3)==0:
        # do testing loop

        # ## test
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]] #batchsize by different sequence length
            sequence_length = 100
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int) #batchsize by same sequence lenght 100
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            y_input = y_train[I_permutation[i:i+batch_size]]
            data = torch.LongTensor(x_input)
            target = torch.FloatTensor(y_input)
            loss, pred = model(data,target,train=True)

            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data[0]
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

torch.save(model,'rnn.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data.npy',data)