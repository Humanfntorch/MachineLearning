# -*- coding: utf-8 -*-
"""
@author: Jacob Rogers
"""
import numpy as np
import pandas as pd
import time
class Perceptron:
    
    def __init__(self):
        self.weights = None
        self.errors = None
        self.predictions = None
        self.counts = None
        self.avgWeight = None
        
    
    def predict(self, x, w):
        # compute dot product for prediction and compute the sign (< 0 -> -1, > 0 -> 1, = 0 -> 0)
        y = np.sign(np.dot(x, w))
        # convert any 0 vals to -1
        zeroIdx = np.where(y == 0)[0]
        for idx in zeroIdx:
            y[idx] = -1
        return y
    
    # Possible variants as type string:
    # 'std' = Basic perceptron prediction. Uses last weight found in epoch
    # 'avg' = Average perceptron prediction. Uses total counts*weights*x and returns the sgn of the sum
    # 'vote' = Voted perceptron prediction. Uses sgn(sum(total counts*sgn(weights*x))) and returns predicted result
    def accuracy(self, x, y, variant = 'std'):
        # Ensure y has correct dimensions for numpy comparisons
        y.values.resize(y.shape[0], 1)
        if(variant == 'std'):
            y_pred = self.predict(x, self.weights[len(self.weights) - 1])
            return (sum(y.values == y_pred)/len(y))
        elif(variant == 'vote'):
            # determine the count for the last weight
            count = 1
            w_t = self.weights[len(self.weights) - 1]
            for i in range(len(y)):
                xyDot = x.iloc[i].T*y.iloc[i]
                wxyDot = np.dot(w_t.T, xyDot)
                if(wxyDot <= 0):
                    self.counts.append(count)
                    break
                else:
                    count += 1
            y_preds = []
            
            for i in range(len(self.weights)):
                y_pred = self.predict(x, self.weights[i])*(self.counts[i])
                y_preds.append(y_pred)
            
            sumY_preds = np.sum(y_preds, axis = 0)
            sumY_preds[sumY_preds == 0] = -1

            final_pred = np.sign(sumY_preds)
            return (sum(y.values == final_pred)/len(y))
        
        elif(variant == "avg"):
            # determine the count for the last weight
            count = 1
            w_t = self.weights[len(self.weights) - 1]
            for i in range(len(y)):
                xyDot = x.iloc[i].T*y.iloc[i]
                wxyDot = np.dot(w_t.T, xyDot)
                if(wxyDot <= 0):
                    self.counts.append(count)
                    break
                else:
                    count += 1
            y_preds = []
            weightTemp = sum(self.weights)/len(self.weights)
            self.avgWeight = weightTemp
            final_pred = self.predict(x, weightTemp)
            return (sum(y.values == final_pred)/len(y))
            
            
    def _updateWeights(self, predictions, y, x, w, r):
        w_t = w.values.T
        count = 1
        for i in range(len(y)):
            xyDot = x.iloc[i].T*y.iloc[i]
            wxyDot = np.dot(w_t, xyDot)
            if(wxyDot <= 0):
                self.weights.append(w_t.T)
                self.counts.append(count)
                w_t = w_t + xyDot.values*r
                count = 1
            else:
                count += 1
        # indices for incorrect predictions
        idx = np.where(y.values != predictions)[0].tolist()
        # Append total number of errors
        self.errors.append(len(idx))
        # w_t =  w.values.T + np.dot(x.iloc[idx].values.T, y.iloc[idx].values)*r
        return pd.DataFrame(data = w_t.T)
    
    def perceptron(self, data, label_name, r, epoch = 1):
        # Set empty lists to store attributes
        self.weights = []
        self.errors = []
        self.predictions = []
        self.counts = []     
        self.avgWeight = None
        
        weightData = [0]*(data.shape[1] - 1)
        w = pd.DataFrame(data = weightData) 
        for t in range(epoch):
            # shuffles data while containing all vals
            sampleData = data.sample(frac = 1)
            y = sampleData[label_name]
            # Resize the numpy array for y's values
            y.values.resize(y.shape[0], 1)
            x = sampleData.drop(columns = [label_name])
            y_t = self.predict(x, w)
            w_t = self._updateWeights(y_t, y, x, w, r)
            w = w_t
            self.predictions.append(y_t)
        self.weights.append(w.values)
        return w
        

dfTest = pd.read_csv('test.csv')
dfTrain = pd.read_csv('train.csv')
# Real label name is genuine/forged. I'm changing it to label for easier use
cols = ["variance of Wavelet Transformed image", "skewness of Wavelet Transformed image", "curtosis of Wavelet Transformed image", "entropy of image", "label"]
dfTest.columns = cols
dfTrain.columns = cols

# convert all zero labels to -1
dfTrain['label'] = dfTrain['label'].replace(0, -1)
dfTest['label'] = dfTest['label'].replace(0, -1)




# Train/Testing dataframe converted to x/y
xTest = dfTest.drop(columns = ['label'])
yTest = dfTest['label']
# Initialize class object
per = Perceptron()
# epoch
T = 10
# learning rate
r = 0.01

stdAcc = 0
stdWeight = 0
votedAcc = 0
avgAcc = 0
votedCounts = []
votedWeights = 0
avgWeights = 0
testIter = 30

def average_lists(lst, numIters):
    tmpList = lst[0]
    dfTmp = pd.DataFrame(data = tmpList)
    for i in range(1, len(lst)):
        dfTmp2 = pd.DataFrame(data = lst[i])
        dfTmp = dfTmp.add(dfTmp2, fill_value = 0)
    dfTmp = dfTmp / numIters
    return np.floor(dfTmp.values)


print("Please wait while experiment performs repeated tests for averages")
print()
for i in range(testIter):
    stdWeight += per.perceptron(dfTrain, 'label', r, epoch = T)
    stdAcc += per.accuracy(xTest, yTest) 
    votedAcc += per.accuracy(xTest, yTest, variant = 'vote')
    votedCounts.append(np.asarray(per.counts))
    avgAcc += per.accuracy(xTest, yTest, variant = 'avg')
    avgWeights += per.avgWeight
    
print("Standard Perceptron Average Error (percent): ", 100*(1 - (stdAcc/testIter)))
print("Standard Perceptron Average Weight: ", stdWeight/testIter)
print("Voted Perceptron Average Error (percent): ", 100*(1 - (votedAcc/testIter)))
avgVotedCount = average_lists(votedCounts, testIter)
# print("Voted Perceptron Average Count: ", avgVotedCount)
# print("Voted Perceptron Last Used Weights: ", per.weights)
print("Average Perceptron Average Error (percent): ", 100*(1 - (avgAcc/testIter)) )  
print("Average Perceptron Averaged Weights: ", avgWeights/testIter)



