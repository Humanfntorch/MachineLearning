# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:01:06 2022

@author: Jacob Rogers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DecisionTree as dt
pd.options.mode.chained_assignment = None  # default='warn'

def build_stump(data, label_name):
    return dt.decision_tree(data, label_name, 2, dt.ENTROPYFLAG, maxChildren = 2)

def calc_alpha(error):
    # Prevent invalid entries in log
    return (1/2*np.log(((1 - error) + 1e-10)/(error + 1e-10)))
  

def calc_error(data, w_t, y_t, label_name):
    return(sum(w_t.loc[(data[label_name] != y_t)])/sum(w_t))

def calc_weights(w_t, data, y_t, label_name, alpha):
    boolSeries = (y_t != data[label_name])*1
    weight = w_t*np.exp(-1*alpha*(boolSeries))
    
    return (weight / sum(weight))

# Increases chances for misclassified data being sampled.
# Also negates the need for a weighted Info Gain calc
def boost_weights(w_t, data, label_name):
    
    # Determine the label with highest weight (therefore most incorrect label)
    listOfValWeights = w_t.value_counts().index.tolist()
   
    lblTempMax = data[label_name][(w_t == max(listOfValWeights)).index]
    maxOccuringLabel = lblTempMax.value_counts().idxmin()
    
    w_t.loc[w_t == max(listOfValWeights)] = sum(w_t.loc[data[label_name] == maxOccuringLabel])
    #w_t.loc[w_t == max(listOfValWeights)] += max(listOfValWeights)

def pred_scores(h_f, data, label_name):
    return sum(h_f != data[label_name]), sum(h_f == data[label_name]) 

def calc_h_t(stumps, alphas, data, label_name):
    h_t = []
    for i in range(len(stumps)):
        y_t = dt.prediction(stumps[i])
        h_t.append(alphas[i]*y_t)
        
class Ada_Boost:
    
    def __init__(self):
        self.alphas = []
        self.stumps = []
        self.T = 1
        self.trainErrors = []
        self.predictErrors = []
        
    def build_stumps(self, data, label_name, T):
        self.alphas = []
        self.stumps = []
        self.T = T
        self.predictErrors = [0]*T
        
        # Set initial weight -> uniform prob = 1/N -> in pd series
        w_t = pd.Series(np.ones(len(data[label_name]))* 1 / data[label_name].shape[0])
        for i in range(0, T):
            # Set initial threshold, required for stump err < threshold to be considered
            threshold = 1
            while(threshold > 0.5):
                # Sample data with higher chance for misclassified data
                sampledData = data.sample(n = len(data), weights = w_t, replace = True, ignore_index = True)                
                # Build stump and calc prediction errors
                stump = build_stump(sampledData, label_name)
                y_t, pe = dt.prediction(stump, sampledData, label_name)
                
                # Calculate ada boost vals
                trainErr = calc_error(sampledData, w_t, y_t, label_name)
                threshold = trainErr
                alpha = calc_alpha(trainErr)
                w_t = calc_weights(w_t, sampledData, y_t, label_name, alpha)
                boost_weights(w_t, sampledData, label_name)
            
            # Stump was found. Add to self and signal stump was found
            self.stumps.append(stump)
            self.alphas.append(alpha)
            self.trainErrors.append(trainErr)
            print("Stump added at: ", i)
            
            
    def predict_vote(self, data, label_name, T):
        # stores hypothesis * alpha for prediction from each stump
        h_t = []
        self.predictErrors = [0]*len(self.stumps)
        tmpStumps = self.stumps.copy()
        for i in range(0, T):
            if(self.stumps[i] in tmpStumps):
                df, predErr = dt.prediction(self.stumps[i], data, label_name)
                # assign prediction to duplicate stumps. Helps optimize tot program
                duplicates = [i for i, stump in enumerate(self.stumps) if stump in self.stumps]
                for idx in duplicates:
                    h_i =  df.values * self.alphas[idx]
                    h_t.append(h_i)
                    
                    self.predictErrors[idx] += predErr
                    tmpStumps[idx] = None
        
        h_final = np.sign(np.sum(h_t, 0))
        return h_final

dfTest = pd.read_csv('test.csv')
dfTrain = pd.read_csv('train.csv')
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
dfTest.columns = cols
dfTrain.columns = cols
numeric_atts = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
unknown_atts = ['job', 'education', 'contact', 'poutcome']
dt.convert_numerical_binary(dfTest, numeric_atts)
dt.convert_numerical_binary(dfTrain, numeric_atts)
dt.convert_unknown(dfTest, unknown_atts)
dt.convert_unknown(dfTrain, unknown_atts)

# Convert data to -1 -> No /1 -> Yes
dfTrain.y = pd.Series(np.where(dfTrain.y.values == 'yes', 1, -1),
          dfTrain.index)
dfTest.y = pd.Series(np.where(dfTest.y.values == 'yes', 1, -1),
          dfTest.index)


ada = Ada_Boost()
adatrainErr = []
adatrainPredErr = []
adatestErr = []
adatestPredErr = []
T = 500
ada.build_stumps(dfTrain, 'y', T)
print("Beginning Predictions")
for i in range(0, T, 1):
    print("Performing prediction with stumps: ", i + 1)
    h_fTrain = ada.predict_vote(dfTrain, 'y', i)
    adatrainPredErr.append(ada.predictErrors[i])
    trerr, trcorr = pred_scores(h_fTrain, dfTrain, 'y')
    h_fTest = ada.predict_vote(dfTest, 'y', i)
    adatestPredErr.append(ada.predictErrors[i])
    testerr, testcorr = pred_scores(h_fTest, dfTest, 'y')
    adatrainErr.append(trerr)
    adatestErr.append(testerr)


x = np.arange(0, T, 1)


fig, ax = plt.subplots()
ax.plot(x, adatrainErr, label = 'TrainErr')
ax.plot(x, adatestErr, label = 'TestErr')
ax.set_xlabel('T: Iterations')
ax.set_ylabel("Number of Errors")
ax.set_title("Errors vs Number of Iterations AdaBoost")
ax.legend(loc='upper right')
fig.savefig("TrainTestAda.png")


fig, ax = plt.subplots()
ax.plot(x, adatrainErr, alpha=0.5, label='Train Err')
ax.plot(x, adatrainPredErr,  alpha=0.5, label='Train Prediction Err')
ax.plot(x, adatestErr, alpha=0.5, label='Test Err')
ax.plot(x, adatestPredErr, alpha=0.5, label='Test Prediction Err')
ax.set_xlabel('Number of Classifiers')
ax.set_ylabel("Number of Errors")
ax.set_title("All Errors per Stump")
ax.legend(loc='upper right')
fig.savefig("AllErrorsAda.png")

class BaggedTrees:
    def __init__(self):
        self.trees = []
        self.bootStrapVal = 1
        self.predictErrors = []
        self.alphas = []
    
    def buildTrees(self, data, label_name, T):
        self.trees = []
        self.trainErrors = []
        self.predictErrors = []
        for i in range(T):
            bootStrapSamples = data.sample(n = len(data), replace = True)
            tree = dt.decision_tree(bootStrapSamples, label_name, maxDepth = len(bootStrapSamples.columns) - 1, splitAlgorithm = dt.ENTROPYFLAG, maxChildren = 2)
            self.trees.append(tree)
            print("tree done: ", i)
            
            
    def Random_Forest(self, data, label_name, num_attributes, T):
        self.trees = []
        self.trainErrors = []
        self.predictErrors = []
        for i in range(T):
            bootStrapSamples = data.sample(n = len(data), replace = True)
            tree = dt.decision_tree(bootStrapSamples, label_name, maxDepth = len(bootStrapSamples.columns) - 1, splitAlgorithm = dt.ENTROPYFLAG, maxChildren = 2, randForest = num_attributes)
            self.trees.append(tree)
            print("tree done: ", i)
        
    def vote(self, data, label_name, T):
        h_t = []
        trainErrors = [0]*len(self.trees)
        tmpTrees = self.trees.copy()
        for i in range(0, T):
            if(self.trees[i] in tmpTrees):
                hf, predErr = dt.prediction(self.trees[i], data, label_name)
                # assign prediction to duplicate stumps. Helps optimize tot program
                duplicates = [i for i, tree in enumerate(self.trees) if tree in self.trees]
                for idx in duplicates:
                    h_t.append(hf)
                    trainErrors[idx] += predErr
                    tmpTrees[idx] = None
        
        h_final = np.sign(np.sum(h_t, 0))
        return h_final, trainErrors
        

bt = BaggedTrees()

trainErr = []
trainPredErr = []
testErr = []
testPredErr = []

T = 500
bt.buildTrees(dfTrain, 'y', T)

print("Beginning Predictions")
for i in range(0, T, 1):
    print("Performing prediction with trees: ", i + 1)
    h_fTrain, trainPredErr = bt.vote(dfTrain, 'y', i)
    trerr, trcorr = pred_scores(h_fTrain, dfTrain, 'y')
    h_fTest, testPredErr = bt.vote(dfTest, 'y', i)
    testerr, testcorr = pred_scores(h_fTest, dfTest, 'y')
    trainErr.append(trerr)
    testErr.append(testerr)



x = np.arange(0, T, 1)


fig, ax = plt.subplots()
ax.plot(x, trainErr, label = 'TrainErr')
ax.plot(x, testErr, label = 'TestErr')
ax.set_xlabel('T: Iterations')
ax.set_ylabel("Number of Errors")
ax.set_title("Errors vs Number of Iterations BaggedTrees")
ax.legend(loc='upper right')
fig.savefig("TrainTestBT.png")

fig, ax = plt.subplots()
ax.plot(x, trainErr, alpha=0.5, label='Train Err')
ax.plot(x, trainPredErr, alpha=0.5, label='Train Prediction Err')
ax.plot(x, testErr, alpha=0.5, label='Test Err')
ax.plot(x, testPredErr, alpha=0.5, label='Test Prediction Err')
ax.set_xlabel('Number of Classifiers')
ax.set_ylabel("Number of Errors")
ax.set_title("All Errors per Tree")
ax.legend(loc='upper right')
fig.savefig("AllErrorsBT.png")


bt = BaggedTrees()

trainErrRF = []
trainPredErrRF = []
testErrRF = []
testPredErrRF = []

T = 500
num_att = [2, 4, 6]
bt.Random_Forest(dfTrain, 'y', num_att, T)

print("Beginning Random Forest Predictions")
for i in range(0, T, 1):
    print("Performing prediction with trees: ", i + 1)
    h_fTrain, trainPredErrRF = bt.vote(dfTrain, 'y', i)
    trerr, trcorr = pred_scores(h_fTrain, dfTrain, 'y')
    h_fTest, testPredErrRF = bt.vote(dfTest, 'y', i)
    testerr, testcorr = pred_scores(h_fTest, dfTest, 'y')
    trainErrRF.append(trerr)
    testErrRF.append(testerr)



x = np.arange(0, T, 1)


fig, ax = plt.subplots()
ax.plot(x, trainErrRF, label = 'TrainErr')
ax.plot(x, testErrRF, label = 'TestErr')
ax.set_xlabel('T: Iterations')
ax.set_ylabel("Number of Errors")
ax.set_title("Errors vs Number of Iterations RandomForest")
ax.legend(loc='upper right')
fig.savefig("TrainTestRF.png")

fig, ax = plt.subplots()
ax.plot(x, trainErrRF, alpha=0.5, label='Train Err')
ax.plot(x, trainPredErrRF, alpha=0.5, label='Train Prediction Err')
ax.plot(x, testErrRF, alpha=0.5, label='Test Err')
ax.plot(x, testPredErrRF, alpha=0.5, label='Test Prediction Err')
ax.set_xlabel('Number of Classifiers')
ax.set_ylabel("Number of Errors")
ax.set_title("All Errors per Tree")
ax.legend(loc='upper right')
fig.savefig("AllErrorRF.png")




