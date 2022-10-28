# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:52:41 2022

@author: Jacob Rogers
"""
import numpy as np
import random

ENTROPYFLAG = 'entropy'
MEFLAG = 'me'
GINIFLAG = 'GINI'


def entropy(att, lbl):
    H = 0
    attTotal = len(att)
    for i in lbl.unique():
        attTemp = att[att == i]
        lblTemp = lbl[lbl == i]
        tot_freq = len(attTemp)
        if tot_freq == 0:
            continue
        hits = len(lblTemp.index.intersection(attTemp.index))
        x_pos = hits/attTotal
        x_neg = 1 - x_pos
        if x_pos == 0:
            x_pos = 1
        if x_neg == 0:
            x_neg = 1
        
        H += -(x_pos*np.log2(x_pos) + (x_neg)*np.log2(x_neg))
        break
    return H

def gini_index(attLbl, lbl):
    counts = []
    tot_count = len(attLbl)
    gi = 0
    for i in lbl.unique():
        counts.append(len(attLbl[attLbl == i]))
    if (tot_count == 0):
        return gi
    else:
        for cnt in counts:
            gi += (cnt/tot_count)**2
    return (1 - gi)
        
def majority_error(attLbl, lbl):
    majLbl = lbl.value_counts().index.tolist()[:1][0]
    majCnt = max(lbl.value_counts())
    missCount = len(attLbl[attLbl != majLbl])
    return missCount, missCount/majCnt

def gain(df, lbl, FLAG):
    Hset = 0
    if(FLAG == MEFLAG):
        misscnt, frat = majority_error(lbl, lbl)
        Hset = misscnt/len(lbl)
    elif(FLAG == ENTROPYFLAG):
        Hset = entropy(lbl, lbl)
    elif(FLAG == GINIFLAG):
        Hset = gini_index(lbl, lbl)
        
    attWeightedGainDict = {}
    attList = list(df.columns)
    attList.remove(lbl.name)
    for attName in attList:
        H = 0
        for val in df[attName].unique():
            attLbl = df[lbl.name][df[attName] == val]
            w = len(attLbl)/len(lbl)
            
            if(FLAG == ENTROPYFLAG):
                H += w*entropy(attLbl, lbl)
            elif(FLAG == MEFLAG):
                missCnt, me = majority_error(attLbl, lbl)
                w = missCnt / len(lbl)
                H +=  w*me
            elif(FLAG == GINIFLAG):
                H += w*gini_index(attLbl, lbl)
                
        attWeightedGainDict[attName] = Hset - H
    return attWeightedGainDict
def find_depth(dictionary):
    return sum([find_depth(depth) if isinstance(depth, dict) else 0 for depth in dictionary.values()])

import time
def _ID3(dtree, df, lbl, treeDepth, maxDepth, FLAG, numChildren, maxChildren, randForest):
    
    if((treeDepth >= maxDepth and numChildren == 0)):
        return 
    elif(treeDepth != 0 and treeDepth + 1 > maxDepth):
        key = list(dtree.keys())[0]
        dtree[key] = lbl.value_counts().idxmax()
        return
    # if(treeDepth + 1 == maxDepth):
    #     return {lbl.value_counts().idxmax()}
    elif(numChildren + 1 == maxChildren):
        key = list(dtree.keys())[0]
        dtree[key] = lbl.value_counts().idxmax()
        return
    elif(len(df) == 0):
        return

    nodeBool = numChildren < maxChildren
    if(not(randForest is None)):
        randForest = [i for i in randForest if i <= len(df.columns) - 1]
        if(len(randForest) < 1):
            return {lbl.value_counts().idxmax()}
        num_rand_att = random.choice(list(randForest))
        dfTemp = df.drop(columns = [lbl.name])
        rand_atts = dfTemp.sample(n = num_rand_att, replace = False, axis = 1).columns.tolist()
        #Ensure we carry label through 
        rand_atts.append(lbl.name)
        dfTemp = df[rand_atts]
        lblTemp = dfTemp[lbl.name]
        gainDict = gain(dfTemp, lblTemp, FLAG)
        split_att = max(gainDict, key = gainDict.get)
        split_att_vals = df[split_att].unique()
    else:    
        gainDict = gain(df, lbl, FLAG)
        split_att = max(gainDict, key = gainDict.get)
        split_att_vals = df[split_att].unique()
    
    if(not(maxChildren is np.inf)):
        split_att_vals = split_att_vals[0:maxChildren]
        
    if(len(dtree) > 0 and numChildren != 0):
        key = list(dtree.keys())[0]
        dtree[key][split_att] = {}
        dtree = dtree[key] 
    else:
       dtree[split_att] = {}
    treeDepth += 1
    
    
    
    # copy data in temp to manipulate recursively
    dfTemp = df
    for val in split_att_vals:
        lblTemp = df[lbl.name][df[split_att] == val]
        # val has unique lbl
        if(len(lblTemp.unique()) == 1 and nodeBool):
            dtree[split_att][val] = lblTemp.unique()[0]
            numChildren += 1
            dfTemp = dfTemp.drop(labels = lblTemp.index.tolist(), axis = 0)
        # val is empty, add most common lbl
        elif(len(lblTemp.unique()) == 0 and nodeBool):
            dtree[split_att][val] = max(lbl.value_counts().index.tolist())
            numChildren += 1
            dfTemp = dfTemp.drop(labels = lblTemp.index.tolist(), axis = 0)
        else:
            if(nodeBool):
                numChildren += 1
                dfChildTemp = dfTemp[dfTemp[split_att] == val].drop(columns = [split_att])
                childTree = {}
                childTree[val] = {}
                _ID3(childTree, dfChildTemp, dfChildTemp[lbl.name], treeDepth, maxDepth, FLAG, numChildren, maxChildren, randForest)
                dtree[split_att][val] = childTree[val]
    
    dfTemp = dfTemp.drop(columns = [split_att])
    lblTemp = dfTemp[lbl.name]
    _ID3(dtree, dfTemp, lblTemp, treeDepth + 1, maxDepth, FLAG, 0, maxChildren, randForest)

def decision_tree(data, label_name, maxDepth, splitAlgorithm, maxChildren:float = np.inf, randForest = None):
    dt = {}
    _ID3(dt, data, data[label_name], 0, maxDepth, splitAlgorithm, 0, maxChildren, randForest)
    return dt
    
         
def classify(dtree, element):
    stopIter = False
    while(not stopIter):
        if not isinstance(dtree, dict):
            return dtree
        
        dtreeIter = iter(dtree)
        val = next(dtreeIter, None)
        if(val == None):
            stopIter = True
        else:
            attVal = element[val]
            if attVal in dtree[val]:
                dtree = dtree[val][attVal]
            else: 
                return None
    return None

def prediction(dtree, data, label_name):
    hitCnt = 0
    missCnt = 0
    data['prediction'] = None
    for idx, row in data.iterrows():
        classification = classify(dtree, data.iloc[idx])
        if(classification is None):
            data['prediction'].iloc[idx] = data[label_name].mean()
            if(data[label_name].mean() == data[label_name].iloc[idx]):
                hitCnt += 1
            else:
                missCnt += 1
            continue
        for el in classification:
            if(el == data[label_name].iloc[idx]):
                hitCnt += 1
                data['prediction'].iloc[idx] = el
            else:
                missCnt += 1
                data['prediction'].iloc[idx] = el
    return data['prediction'], missCnt/(missCnt + hitCnt)

def convert_numerical_binary(df, label_names):
    for label_name in label_names:
        threshold = df[label_name].median()
        df.loc[df[label_name] <= threshold, label_name] = 0
        df.loc[df[label_name] > threshold, label_name] = 1
        
def convert_unknown(df, label_names):
    for label_name in label_names:
        majLbl = df[label_name].value_counts().index.tolist()[:1][0]
        df.loc[df[label_name] == "unknown", label_name] = majLbl
        

 
            
    
        
    
import pandas as pd
d = {"P": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0], "O": ['S', 'S', 'O', 'R', 'R', 'R', 'O', 'S', 'S', 'R', 'S', 'O', 'O', 'R'], "T": ['H', 'H', 'H', 'M', 'C', 'C', 'C', 'M', 'C', 'M', 'M', 'M', 'H', 'M'], "H": ["H", "H", "H", "H", "N", "N", "N", "H", "N" ,"N", "N", "H", "N", "H"], "W": ["W", "S", "W", "W", "W", "S", "S", "W", "W", "W", "S", "S", "W", "S"]}
df = pd.DataFrame(data = d)
missing_ex = {"P": [5/14, 4/14, 5/14], "O":["S", "O", "R"], "T":["M", "M", "M"], "H":["N", "N", "N"], "W":["W", "W", "W"]}
d0 = {"y":[0, 0, 1, 1, 0, 0, 0], "x1": [0, 0, 0, 1, 0, 1, 0], "x2": [0, 1, 0, 0, 1, 1, 1], "x3": [1, 0, 1, 0, 1, 0, 0], "x4": [0, 0, 1, 1, 0, 0, 1]}
df0 = pd.DataFrame(data = d0)
dfMissing = pd.DataFrame(data = missing_ex)
#df = pd.concat([df, dfMissing], ignore_index = True)
dfCarsTest = pd.read_csv('cartest.csv')
dfCarsTrain = pd.read_csv('cartrain.csv')
with open ("cardata-desc.txt", 'r') as f:
    for line in f:
        terms = line.strip().split(",")
dfCarsTest.columns = terms
dfCarsTrain.columns = terms

#carTree1 = decision_tree(dfCarsTrain, "label", 1, ENTROPYFLAG)
carTree2 = decision_tree(dfCarsTrain, "label", 2, ENTROPYFLAG)
carTree3 = decision_tree(dfCarsTrain, "label", 3, ENTROPYFLAG)
# carTree4 = decision_tree(dfCarsTrain, "label", 4, ENTROPYFLAG)
# carTree5 = decision_tree(dfCarsTrain, "label", 5, ENTROPYFLAG)
# carTree6 = decision_tree(dfCarsTrain, "label", 6, ENTROPYFLAG)

# preds1, petr1 = prediction(carTree1, dfCarsTest, 'label')
# preds2, petr2 = prediction(carTree2, dfCarsTest, 'label')
# preds3, petr3 = prediction(carTree3, dfCarsTest, 'label')
# preds4, petr4 = prediction(carTree4, dfCarsTest, 'label')
# preds5, petr5 = prediction(carTree5, dfCarsTest, 'label')
# preds6, petr6 = prediction(carTree6, dfCarsTest, 'label')
# peAvg = (petr1 + petr2 + petr3 + petr4 + petr5 + petr6)/6

# print(peAvg)

#dfBankTest = pd.read_csv('test.csv')
# dfBankTrain = pd.read_csv('train.csv')
# cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
# dfBankTest.columns = cols
# dfBankTrain.columns = cols
# numeric_atts = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
# unknown_atts = ['job', 'education', 'contact', 'poutcome']
# convert_numerical_binary(dfBankTest, numeric_atts)
# convert_numerical_binary(dfBankTrain, numeric_atts)
# convert_unknown(dfBankTest, unknown_atts)
# convert_unknown(dfBankTrain, unknown_atts)
# bT1 = decision_tree(dfBankTrain, "y", 1, GINIFLAG)
# bT2 = decision_tree(dfBankTrain, "y", 2, GINIFLAG)
# bT3 = decision_tree(dfBankTrain, "y", 3, GINIFLAG)
# bT4 = decision_tree(dfBankTrain, "y", 4, GINIFLAG)
# bT5 = decision_tree(dfBankTrain, "y", 5, GINIFLAG)
# bT6 = decision_tree(dfBankTrain, "y", 6, GINIFLAG)
# bT7 = decision_tree(dfBankTrain, "y", 7, GINIFLAG)
# bT8 = decision_tree(dfBankTrain, "y", 8, GINIFLAG)
# bT9 = decision_tree(dfBankTrain, "y", 9, GINIFLAG)
# bT10 = decision_tree(dfBankTrain, "y", 10, GINIFLAG)
# bT11 = decision_tree(dfBankTrain, "y", 11, GINIFLAG)
# bT12 = decision_tree(dfBankTrain, "y", 12, GINIFLAG)
# bT13 = decision_tree(dfBankTrain, "y", 13, GINIFLAG)
# bT14 = decision_tree(dfBankTrain, "y", 14, GINIFLAG)
# bT15 = decision_tree(dfBankTrain, "y", 15, GINIFLAG)
# bT16 = decision_tree(dfBankTrain, "y", 16, GINIFLAG)

# be1 = prediction_error(bT1, dfBankTrain, 'y')
# be2 = prediction_error(bT2, dfBankTrain, 'y')
# be3 = prediction_error(bT3, dfBankTrain, 'y')
# be4 = prediction_error(bT4, dfBankTrain, 'y')
# be5 = prediction_error(bT5, dfBankTrain, 'y')
# be6 = prediction_error(bT6, dfBankTrain, 'y')
# be7 = prediction_error(bT7, dfBankTrain, 'y')
# be8 = prediction_error(bT8, dfBankTrain, 'y')
# be9 = prediction_error(bT9, dfBankTrain, 'y')
# be10 = prediction_error(bT10, dfBankTrain, 'y')
# be11 = prediction_error(bT11, dfBankTrain, 'y')
# be12 = prediction_error(bT12, dfBankTrain, 'y')
# be13 = prediction_error(bT13, dfBankTrain, 'y')
# be14 = prediction_error(bT14, dfBankTrain, 'y')
# be15 = prediction_error(bT15, dfBankTrain, 'y')
# be16 = prediction_error(bT16, dfBankTrain, 'y')

# beAvg = (be1 + be2 + be3 + be4 + be5 + be6 + be7 + be8 + be9 + be10 + be11 + be12 + be13 + be14 + be15 + be16)/16
# print(beAvg)
#print("It worked!")


        
    
    
