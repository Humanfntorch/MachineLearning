# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:23:52 2022

@author: Jacob Rogers

This is for the Kaggle competition for CS6350.

The task is to predict whether an individual has an income > $50k
based on surveyed data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sn
import matplotlib.pyplot as plt
import csv


# Replaces all instances of '?' in a column with one of the top 3 majority choices (excluding '?') in that column. 
def complete_missing_value(df):
    for col in df.columns:
        if(len(df[col][df[col] == '?']) > 0):
            mstCmnLbl = df[col].value_counts().index[:4]
            newVal = np.random.choice(mstCmnLbl)
            if(newVal == '?'):
                mstCmnLbl = mstCmnLbl.drop("?")
                newVal = np.random.choice(mstCmnLbl)
            df.loc[df[col] == "?", col] = newVal


label_name = 'income>50K'

dfTrain = pd.read_csv('train_final.csv', header = 0, na_values = "?")
dfTest = pd.read_csv('test_final.csv', header = 0, na_values = "?")


# determine number of NAs in each df
trainNA = len(dfTrain[dfTrain.isna().any(axis = 1)]) # 1848
testNA = len(dfTest[dfTest.isna().any(axis = 1)]) # 1772
fracTrainNA = trainNA / len(dfTrain) # ~ 7.4%
fracTestNA = testNA / len(dfTest) # 0~ 7.4%

# Drop NA in both dfs and call them xtr, ytr => train and xt => test
corrNA = dfTrain.dropna() # Used to test correlations
dfTrain = dfTrain.dropna()


# # Determine percentages of yes/no labels
# lessThan = len(ytr[ytr == 0]) / len(ytr) # ~ 75.1%
# greaterThan = len(ytr[ytr == 1]) / len(ytr) # ~ 24.9%

corrMat = corrNA.corr()
# The category 'fnlwgt' produces a negative correlation

# print corrMat
sn.heatmap(corrMat, annot = True)
plt.show()
# sn.heatmap(corrMat2, annot = True)
# plt.show()


# drop 'fnlwgt' due to no corr
# dfTrain = dfTrain.drop(columns = ['fnlwgt'])
# dfTest = dfTest.drop(columns = ['fnlwgt'])

# Determine numerical atts and categorical atts
numericalAtts = dfTrain.select_dtypes(include = ['number']).columns
categoricalAtts = dfTrain.select_dtypes(include = object).columns
# assert (len(numericalAtts) + len(categoricalAtts)) == len(xtr.columns)


y = dfTrain[label_name].values
# encoding, which bins categorical types in bins [0, n] for ez numerical manipulation
features_final = pd.get_dummies(dfTrain)
for col in dfTrain.drop(columns=[label_name]).columns:
  if col not in numericalAtts:
    temp={}
    for i in range(len(dfTrain[col].unique())):
      temp[dfTrain[col].unique()[i]]=i
    work = dfTrain[col].map(temp)
    dfTrain[col]=work
    
features_final = pd.get_dummies(dfTest)
for col in dfTest.columns:
  if col not in numericalAtts:
    temp={}
    for i in range(len(dfTest[col].unique())):
      temp[dfTest[col].unique()[i]]=i
    work = dfTest[col].map(temp)
    dfTest[col]=work
    

# Scale numerical data for appropriate range: STANDARD
standard_scaler = StandardScaler()
scaler = standard_scaler.fit(dfTrain[numericalAtts])
dfTrain[numericalAtts] = scaler.transform(dfTrain[numericalAtts])
scaler = standard_scaler.fit(dfTest[numericalAtts.drop(labels = label_name)])
dfTest[numericalAtts.drop(labels = label_name)] = scaler.transform(dfTest[numericalAtts.drop(labels = label_name)])

# Scale numerical data for appropriate range: MINMAX
# min_max = MinMaxScaler()
# scaler = min_max.fit(dfTrain[numericalAtts])
# dfTrain[numericalAtts] = scaler.transform(dfTrain[numericalAtts])
# scaler = min_max.fit(dfTest[numericalAtts.drop(labels = label_name)])
# dfTest[numericalAtts.drop(labels = label_name)] = scaler.transform(dfTest[numericalAtts.drop(labels = label_name)])


x = dfTrain.drop(columns = [label_name]).values
dfTest = dfTest.drop(columns = ["ID"])
xTest = dfTest.values

### RANDOM FOREST REGRESSOR MODEL

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(n_estimators = 300)
# rfTree = rf.fit(x, y)
# y_pred = rfTree.predict(x)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0
# acc = sum(y_pred == y)/len(y)
# print(acc)

# y_pred = rfTree.predict(xTest)

# idPredictionList = list()
# for i in range(len(y_pred)):
#     idPredictionList.append([i + 1, y_pred[i]])
# csvHeader = ["ID", "Prediction"]
# with open("initPrediction9RF.csv", 'w', newline = '') as f:
#     writer = csv.writer(f)
#     writer.writerow(csvHeader)
#     writer.writerows(idPredictionList)

### END RANDOM FOREST REGRESSION MODEL



### SGD CLASSIFIER

# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier(loss="hinge", penalty="l2", max_iter = 1000)
# clf.fit(x, y)

# y_pred = clf.decision_function(x)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0

# acc = sum(y_pred == y)/(len(y))
# print("SGD accuracy: ", acc)

# y_pred = clf.decision_function(xTest)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0
# idPredictionList = list()
# for i in range(len(y_pred)):
#     idPredictionList.append([i + 1, y_pred[i]])
# csvHeader = ["ID", "Prediction"]
# with open("initPrediction2SGD.csv", 'w', newline = '') as f:
#     writer = csv.writer(f)
#     writer.writerow(csvHeader)
#     writer.writerows(idPredictionList)

### END SGD CLASSIFIER


### SGD REGRESSOR

# from sklearn.linear_model import SGDRegressor
# clf = SGDRegressor(loss="squared_error", penalty="l2", max_iter = 1000)
# clf.fit(x, y)

# y_pred = clf.predict(x)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0

# acc = sum(y_pred == y)/(len(y))
# print("SGD accuracy: ", acc)
# y_pred = clf.predict(xTest)
# idPredictionList = list()
# for i in range(len(y_pred)):
#     idPredictionList.append([i + 1, y_pred[i]])
# csvHeader = ["ID", "Prediction"]
# with open("initPredictionMinMaxSGD4.csv", 'w', newline = '') as f:
#     writer = csv.writer(f)
#     writer.writerow(csvHeader)
#     writer.writerows(idPredictionList)

### END SGD REGRESSOR


### SVM REGRESSION

# from sklearn.svm import SVR
# clf = SVR(kernel='rbf')
# clf.fit(x, y)
# y_pred = clf.predict(x)
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0
# acc = sum(y == y_pred)/len(y)
# print("rbf acc: ", acc)

# y_pred = clf.predict(xTest)
# idPredictionList = list()
# for i in range(len(y_pred)):
#     idPredictionList.append([i + 1, y_pred[i]])
# csvHeader = ["ID", "Prediction"]
# with open("initPredictionMinMaxSVMRBF3.csv", 'w', newline = '') as f:
#     writer = csv.writer(f)
#     writer.writerow(csvHeader)
#     writer.writerows(idPredictionList)

### END SVM REGRESSION

### NEURAL NETWORK

import NN
# reshape x and xTest to fit neural network
x = x.reshape(x.shape[0], 1, x.shape[1])
xTest = xTest.reshape(xTest.shape[0], 1, xTest.shape[1])
nn = NN.neural_network(1)
nn.insert_layer(NN.layer((14, 30), False))
nn.insert_layer(NN.layer((), True))
nn.insert_layer(NN.layer((30, 30), False))
nn.insert_layer(NN.layer((), True))
nn.insert_layer(NN.layer((30, 30), False))
nn.insert_layer(NN.layer((), True))
nn.insert_layer(NN.layer((30, 1), False))
nn.insert_layer(NN.layer((), True))
epochs = 500
nn.train(x, y, epochs)
y_pred = nn.prediction(x, y)
y_pred = np.asarray(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0])
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
acc = sum(y == y_pred)/len(y)
print(acc)
y_pred = nn.prediction(xTest, y)
y_pred = np.asarray(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0])
idPredictionList = list()
for i in range(len(y_pred)):
    idPredictionList.append([i + 1, y_pred[i]])
csvHeader = ["ID", "Prediction"]
with open("initPredictionNN2.csv", 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(csvHeader)
    writer.writerows(idPredictionList)    
plot_epochs = np.arange(1, epochs + 1, 1)
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(plot_epochs, nn.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v Epoch")

### END NEURAL NETWORK
        

