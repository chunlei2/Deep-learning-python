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


#Build the model
class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()
        
        self.lstm = nn.LSTMCell(in_size,out_size)
        self.out_size = out_size
        
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self,x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
#             self.c = Variable(torch.zeros(state_size)).cuda()
#             self.h = Variable(torch.zeros(state_size)).cuda()
            self.c = torch.zeros(state_size)
            self.h = torch.zeros(state_size)
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x

class RNN_language_model(nn.Module):
    def __init__(self,vocab_size, no_of_hidden_units):
        super(RNN_language_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout()

        self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout2 = LockedDropout() 

        self.decoder = nn.Linear(no_of_hidden_units, vocab_size)

        self.loss = nn.CrossEntropyLoss()#ignore_index=0)

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        self.lstm2.reset_state()
        self.dropout2.reset_state()

    def forward(self, x, train=True): #batch_size, time_steps
    
        embed = self.embedding(x) # batch_size, time_steps, features
        no_of_timesteps = embed.shape[1]
        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps - 1):

            h = self.lstm1(embed[:,i,:]) #batch_size, features
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.3,train=train)

            h = self.lstm2(h)
            h = self.bn_lstm2(h)
            h = self.dropout2(h,dropout=0.3,train=train)

            h = self.decoder(h) #batch, vocab_size

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,vocab_size)
        target_prediction = outputs.permute(1,0,2) # batch, time, vocab
        outputs = outputs.permute(1,2,0) # (batch_size,vocab_size,time_steps)

        if(train==True):

            target_prediction = target_prediction.contiguous().view(-1,vocab_size) # (batch_size*(time_steps-1))by vocab_size
            target = x[:,1:].contiguous().view(-1) #batch_size*(time_step - 1)
            loss = self.loss(target_prediction,target)

            return loss, outputs
        else:
            return outputs

#Read in the data
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
 #the first 12500 are positive reviews, the next 12500 are negative reviews, 50000 are unlabelled reviews

x_test = []
with open('/Users/liuchunlei/Desktop/IMDB Movie reviews/preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)

model = RNN_language_model(8001, 300)
vocab_size += 1
batch_size = 200
no_of_epochs = 6
# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
L_Y_train = len(x_train) #75000
L_Y_test = len(x_test)


train_loss = []
train_accu = []
test_accu = []


#train the model
print('begin training...')
for epoch in range(0,6):
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 100
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl<sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
#         x_input = Variable(torch.LongTensor(x_input),requires_grad=True).cuda()
        x_input = torch.LongTensor(x_input)
        optimizer.zero_grad()
        loss, pred = model(x_input) # pred:(batch_size,vocab_size,time_steps-1)
        loss.backward()

        norm = nn.utils.clip_grad_norm(model.parameters(),2.0)

        optimizer.step()   # update gradients
        
        values,prediction = torch.max(pred,1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
        epoch_acc += accuracy
        epoch_loss += loss.data[0]
        epoch_counter += batch_size
        
        if (i+batch_size) % 1000 == 0 and epoch==0:
            print(i+batch_size, accuracy/batch_size, loss.data[0], norm, "%.4f" % float(time.time()-time1))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    ## test
    if((epoch+1)%1==0):
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        #torch.from_numpy(
        for i in range(0, 2000, batch_size):
            #apply .cuda() to move to GPU
            sequence_length = 100
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl<sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
#             x_input = Variable(torch.LongTensor(x_input)).cuda()
            x_input = torch.LongTensor(x_input)
            pred = model(x_input,train=False)
            
            values,prediction = torch.max(pred,1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_counter += batch_size
            #train_accu.append(accuracy)
            if (i+batch_size) % 1000 == 0 and epoch==0:
                print(i+batch_size, accuracy/batch_size)
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    if(((epoch+1)%2)==0):
        torch.save(model,'temp.model')
        torch.save(optimizer,'temp.state')
        data = [train_loss,train_accu,test_accu]
        data = np.asarray(data)
        np.save('data.npy',data)
torch.save(model,'language.model')


#generate fake reviews
imdb_dictionary = np.load('/Users/liuchunlei/Desktop/IMDB Movie reviews/preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}
#generate fake reviews
model = RNN_language_model(8001, 300)
model = torch.load('temp.model')
print('model loaded...')
# model.cuda()
model.eval()

tokens = [['a'], ['i']]
token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in tokens]
x = torch.LongTensor(token_ids)

##### preload phrase

embed = model.embedding(x) # batch_size, time_steps, features

state_size = [embed.shape[0],embed.shape[2]] # batch_size, features
no_of_timesteps = embed.shape[1]

model.reset_state()

outputs = []
for i in range(no_of_timesteps):

    h = model.lstm1(embed[:,i,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.5,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.5,train=False)

    h = model.decoder(h)

    outputs.append(h)

outputs = torch.stack(outputs) #time_steps, batch_size, vocab_size
outputs = outputs.permute(1,2,0) #batch_size, vocab_size, time_steps
output = outputs[:,:,-1] #batch_size, vocab_size

temperature = 1.0 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1) #pick one word
    review.append(x.cpu().data.numpy()[:,0])
    
    ## predict the next word
    embed = model.embedding(x) # batch_size, time_steps, features
    no_of_timesteps = embed.shape[1]
    for i in range(no_of_timesteps):    
         
        h = model.lstm1(embed[:, i, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h,dropout=0.3,train=False)

        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h,dropout=0.3,train=False)

        output = model.decoder(h)

review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)
temperature = 0.5 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1) #pick one word
    review.append(x.cpu().data.numpy()[:,0])
    
    ## predict the next word
    embed = model.embedding(x) # batch_size, time_steps, features
    no_of_timesteps = embed.shape[1]
    for i in range(no_of_timesteps):    
         
        h = model.lstm1(embed[:, i, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h,dropout=0.3,train=False)

        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h,dropout=0.3,train=False)

        output = model.decoder(h)

review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)

temperature = 1.5 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1) #pick one word
    review.append(x.cpu().data.numpy()[:,0])
    
    ## predict the next word
    embed = model.embedding(x) # batch_size, time_steps, features
    no_of_timesteps = embed.shape[1]
    for i in range(no_of_timesteps):    
         
        h = model.lstm1(embed[:, i, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h,dropout=0.3,train=False)

        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h,dropout=0.3,train=False)

        output = model.decoder(h)

review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)