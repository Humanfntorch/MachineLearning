# -*- coding: utf-8 -*-
"""
@author: Jacob Rogers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Initializes a node in the network graph. 
    # Each node has at least one incoming edge (base value or calculated)
    # Each node should have at least one outgoing edge
    # Each node has the ability to use activation function + derivative
class node:
    def __init__(self):
        self.in_edge = None
        self.out_edge = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        # return np.tanh(x)
    
    def p_sigmoid(self, x):
        return (self.sigmoid(x)*(1 - self.sigmoid(x)))
        # return 1-np.tanh(x)**2;

class hidden_layer(node):
    def __init__(self):
        return
    
    def forward_propagate(self, x):
        self.in_edge = x
        self.out_edge = self.sigmoid(self.in_edge)
        return(self.out_edge)
    
    def back_propagate(self, gradient, learn_rate):
        return (self.p_sigmoid(self.in_edge)*gradient)
    

# Each layer is composed of a defined number of nodes, each with:
    # A set of weights where w_ij belongs to node i and connects to node j
    # The ability to distinguish layer type (excluding output layer) [hidden/non-hidden]
    # The ability to propagate forward. [hidden/non-hidden]
    # The ability to propagate backwards. [hidden/non-hidden]
class layer(node):
    # Weights represents how weights are initialized:
        # "rand" weights are initialized randomly
        # "zero" weights are initialized with 0
    # dims defines the size of input nodes/output nodes as (input, output)
    # width defines the width of the gaussian distribution to draw from. Default 5
    def __init__(self, dims, isHidden, weights = 'rand'):
        self.isHidden = isHidden
        if(not isHidden):
            if(weights == 'rand'):
                self.weights = np.random.normal(size = (dims[0], dims[1]))
                self.bias = np.random.normal(size = (1, dims[1]))
            elif(weights == 'zero'):
                self.weights = np.zeros([dims[0], dims[1]])
                self.bias = np.zeros([1, dims[1]])
            elif(weights == 'ones'):
                self.weights = np.ones([dims[0], dims[1]])
                self.bias = 0
            
    def forward_propagate(self, x):
        if(self.isHidden):
            self.in_edge = x
            self.out_edge = self.sigmoid(self.in_edge)
            return(self.out_edge)
        else:
            self.in_edge = x
            self.out_edge = np.dot(self.in_edge, self.weights) + self.bias
            return self.out_edge
    
    def back_propagate(self, gradient, learn_rate):
        if(self.isHidden):
            return(self.p_sigmoid(self.in_edge)*gradient)
            # return(self.in_edge*(1 - self.in_edge)*gradient)
            #return(self.out_edge*(1 - self.out_edge)*gradient)
        else:
            # sum of product of weights with current gradient = back propagated gradient
            updated_grad = np.dot(gradient, self.weights.T)
            # update weight based on running gradient cost
            self.weights = self.weights - learn_rate*np.dot(self.in_edge.T, gradient)
            # update bias based on running gradient cost
            self.bias = self.bias - learn_rate*gradient
            return updated_grad
    
class neural_network:
   def __init__(self, init_learn_rate):
       self.learning_rate = init_learn_rate
       self.layers = []
       self.losses = []
       self.y_preds = []
       self.accuracy = None
       
   def insert_layer(self, layer):
       self.layers.append(layer)
       
   def _loss(self, y, y_t):
       return(1/2*((y - y_t)**2).mean())
       
   def del_loss(self, y, y_t):
       return((y_t - y)/y.size)
   
   def train(self, x, y, epochs):
       self.losses = []
       
       # Scheduling for learning rate
       for i in range(epochs):
           d = (i + 1)**10
           loss = 0
           self.learning_rate = (self.learning_rate)/(1 + (self.learning_rate*i)/d)
           for j in range(len(x)):
               # initial input from first layer
               out_edge = x[j]
               for layer in self.layers:
                   out_edge = layer.forward_propagate(out_edge)
               # Propagated to output layer. Calc loss and del_loss.   
               loss += self._loss(y[j], out_edge)
               dloss = self.del_loss(y[j], out_edge)
               # Back propagate the cached values with the partial loss
               for k in range(len(self.layers) - 1, -1, -1):
                    dloss = self.layers[k].back_propagate(dloss, self.learning_rate)
           self.losses.append(loss/len(x))
        
          
           
   def prediction(self, x, y, threshold = 0.5):
       # reset list of weights
       self.y_preds = []
       self.accuracy = 0
       # Perform forward propagation on trained network using x as input
       for i in range(len(x)):
           out_edge = x[i]
           for layer in self.layers:
               out_edge = layer.forward_propagate(out_edge)
           if(out_edge > threshold):
               self.y_preds.append(1)
           else:
               self.y_preds.append(-1)
       self.accuracy = sum(y == self.y_preds)/len(y) 
       return self.y_preds



dfTest = pd.read_csv('test.csv')
dfTrain = pd.read_csv('train.csv')
# Real label name is genuine/forged. I'm changing it to label for easier use
cols = ["variance of Wavelet Transformed image", "skewness of Wavelet Transformed image", "curtosis of Wavelet Transformed image", "entropy of image", "label"]
dfTest.columns = cols
dfTrain.columns = cols
# convert all zero labels to -1
dfTrain['label'] = dfTrain['label'].replace(0, -1)
dfTest['label'] = dfTest['label'].replace(0, -1);

# Store x/y as train test variables
yTest = dfTest['label'].values
yTrain = dfTrain['label'].values
xTest = dfTest.drop(columns = ['label']).values
xTrain = dfTrain.drop(columns = ['label']).values
xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1])
xTest = xTest.reshape(xTest.shape[0], 1, xTest.shape[1])


width = [5, 10, 25, 50, 100]
epochs = 100
threshold = 0.5
weight_type = 'zero'
for i in range(len(width)):
    nn = neural_network(1)
    # Layer 1
    nn.insert_layer(layer((4, width[i]), False, weight_type))
    nn.insert_layer(layer((), True))
    # Layer 2
    nn.insert_layer(layer((width[i], 1), False, weight_type))
    nn.insert_layer(layer((), True))
    # Layer 3

    nn.train(xTrain, yTrain, epochs)
    y = nn.prediction(xTrain, yTrain, threshold)

    print("Node width = {width}, Training Error = {err}".format(width = width[i], err = 1 - nn.accuracy))
    y = nn.prediction(xTest, yTest, threshold)
    print("Node width = {width}, Testing Error = {err}".format(width = width[i], err = 1 - nn.accuracy))

    plot_epochs = np.arange(1, epochs + 1, 1)
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(plot_epochs, nn.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss v Epoch")
