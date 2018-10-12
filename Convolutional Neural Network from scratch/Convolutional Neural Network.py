import numpy as np
import h5py
import copy
import random
from random import randint

random.seed(9301)
np.random.seed(9301)
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] ).reshape((60000, 28, 28))
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] ).reshape((10000, 28, 28))
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#size of matrix
size_inputs = 28
#number of outputs
num_outputs = 10
#number of channels
num_channels = 3
#size of filters
size_filters = 5
model = {}
model['K'] = np.random.binomial(1, 0.5, size=(num_channels, size_filters, size_filters)) #(5, 5, 5) filter
model['W'] = np.random.randn(num_outputs, num_channels, size_inputs-size_filters+1, size_inputs-size_filters+1) / np.sqrt(size_inputs) #(10, 5, 24, 24)
model['b'] = np.zeros((num_outputs,)) #(10,)
model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def convolution(x, model, size_inputs, size_filters):
    Z = np.zeros((num_channels, size_inputs - size_filters + 1, size_inputs - size_filters + 1)) #(5,24,24)
    if type(model) == dict:
        for s in range(num_channels):
            for i in range(size_inputs - size_filters + 1):
                for j in range(size_inputs - size_filters + 1):
                    sub_x = x[i:i+size_filters,j:j+size_filters]
                    Z[s][i][j] = np.tensordot(sub_x, model['K'][s], axes=2)
    elif type(model) == np.ndarray:
        for s in range(num_channels):
            for i in range(size_inputs - size_filters + 1):
                for j in range(size_inputs - size_filters + 1):
                    sub_x = x[i:i+size_filters,j:j+size_filters]
                    Z[s][i][j] = np.tensordot(sub_x, model[s], axes=2)        
    return Z
def forward(Z,x,y,model):
    H = np.maximum(Z, 0) #(5, 24, 24)
    U = np.tensordot(model['W'], H, axes=3) + model['b'] #(10,)
    p = softmax_function(U) #(10,)
    return p
def backward(x,y,p,Z,model, model_grads):
    H = np.maximum(Z, 0) #(5, 24, 24)
    dU = -1.0*p 
    dU[y] = dU[y] + 1.0 #(10,)
    db = dU #(10,)
    dH = np.tensordot(model['W'], dU, axes = ([0], [0])) #(5, 24, 24)
    dZ_dH = np.multiply(1/(1+np.exp(-Z)), dH)
    dK = convolution(x, dZ_dH, size_inputs, size_inputs - size_filters + 1)
    dW = []
    for i in range(10):
        dW.append(dU[i]*H) #may be wrong
    dW = np.array(dW)
    model_grads['W'] = dW
    model_grads['K'] = dK
    model_grads['b'] = db
    return model_grads
LR = .01
num_epochs = 7
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random]
        Z = convolution(x, model, size_inputs, size_filters)
        p = forward(Z, x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p,Z, model, model_grads)
        model['W'] = model['W'] + LR*model_grads['W']
        model['K'] = model['K'] + LR*model_grads['K']
        model['b'] = model['b'] + LR*model_grads['b']
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n]
    Z = convolution(x, model, size_inputs, size_filters)
    p = forward(Z, x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n]
    Z = convolution(x, model, size_inputs, size_filters)
    p = forward(Z, x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )