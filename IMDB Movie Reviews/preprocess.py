import numpy as np
import os
import nltk
import itertools
import torch.nn as nn
import torch

## create directory to store preprocessed data
if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')

## get all of the training reviews (including unlabeled reviews)
train_directory = '/Users/liuchunlei/Desktop/IMDB Movie reviews/aclImdb/train/'
test_directory = '/Users/liuchunlei/Desktop/IMDB Movie reviews/aclImdb/test/'

train_pos_filenames = os.listdir(train_directory + 'pos/')
train_neg_filenames = os.listdir(train_directory + 'neg/')
train_unsup_filenames = os.listdir(train_directory + 'unsup/')

test_pos_filenames = os.listdir(test_directory + 'pos/')
test_neg_filenames = os.listdir(test_directory + 'neg/')

train_pos_filenames = [train_directory+'pos/'+filename for filename in train_pos_filenames]
train_neg_filenames = [train_directory+'neg/'+filename for filename in train_neg_filenames]
train_unsup_filenames = [train_directory+'unsup/'+filename for filename in train_unsup_filenames]

test_pos_filenames = [test_directory+'pos/'+filename for filename in test_pos_filenames]
test_neg_filenames = [test_directory+'neg/'+filename for filename in test_neg_filenames]

train_filenames = train_pos_filenames + train_neg_filenames + train_unsup_filenames
test_filenames = test_pos_filenames + test_neg_filenames

count = 0
x_train = []
for filename in train_filenames:
    with open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_train.append(line)
    count += 1
print(count)

count = 0
x_test = []
for filename in test_filenames:
    with open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_test.append(line)
    count += 1
print(count)


## number of tokens per review
no_of_tokens = []
for tokens in x_train:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))
#The mean review contains ~267 tokens with a standard deviation of ~200. Although there are over 20 million total tokens, they’re obviously not all unique.

### word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

## sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices][0:8000] #keep the most frequent 8000 words
word_to_id = {token:index for index, token in enumerate(id_to_word)} 
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
print(np.sum(count[0:100])/np.sum(no_of_tokens))
print(np.sum(count[0:8000])/np.sum(no_of_tokens))


## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]


## save dictionary
np.save('preprocessed_data/imdb_dictionary.npy',np.asarray(id_to_word))

## save training data to single text file
with open('preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
## save test data to single text file
with open('preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

glove_filename = '/Users/liuchunlei/Desktop/IMDB Movie reviews/glove.840B.300d.txt'
with open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

with open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
        
with open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")