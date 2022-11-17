# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:52:41 2022

@author: Jacob Rogers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def y_pred(x, w, b):
    # Calculates the inner product of cols, sums over the vals and stores them in rows
    return (b + w.values*x.values.T)
def cost(b_i, y_i, y):
    # y - w^Tx_i
    y_cost = (y.values - y_i - b_i)
    # J(w) = 1/2*Sigma || y - w^Tx_i ||_1
    return(1/(2*len(y))*np.sum((y_cost)**2))

# x -> Testing Attributes (Expects Pandas.DataFrame)
# y -> Testing Labels (Expects Pandas.DataFrame)
# y_pred -> Predicted Labels 
# b -> Bias Parameter
# w -> Weight Vector Parameter
# r -> Learning Rate
def tune_params(x, y, y_pred, b_old, w_old, r):
    # del_ denotes partial w.r.t given param
    del_b = -1*(np.sum(y.values - y_pred))/len(y)
    #y_diff = y - y_pred
    # del_w = -1*np.sum(np.dot((y.values - y_pred), x))/len(y)
    del_w = -1*sum((y.values - y_pred).T*x.values)/len(y)
    b_new = (b_old - r*del_b)
    w_new = (w_old.values.T) - r*(del_w)
    w_new = pd.DataFrame(data = w_new.T)
    return(b_new, w_new)

class LSM_Regression:

    def __init__(self):
        self.weights = []
        self.costs = []
        self.rates = []
        self.weightNorms = []
        self.biases = []
        self.y_predicts = []
        self.w_f = None
        self.b_f = None

    def thresholdMet(self, w_old, w_new, threshold):
        w_norm = np.linalg.norm(w_new - w_old)
        return(w_norm < threshold)

    def gradient_descent(self, data, label_name, r, threshold, T):
        # Clear each field member before a new call
        self.costs = []
        self.weights = []
        self.rates = []
        self.weightNorms = []
        self.y_predicts = []
        self.biases = []
        self.w_f = None
        self.b_f = None
        
        data = (data - data.mean())/data.std()
        y = data[label_name]
        x = data.drop(columns=[label_name])
        weightData = [0]*x.shape[1]
        w = pd.DataFrame(data = weightData)
        b = 0
        for i in range(0, T):
            y_t = y_pred(x, w, b)
            j_t = cost(b, y_t, y)
            self.costs.append(j_t)
            self.y_predicts.append(y_t)
            b_i, w_i = tune_params(x, y, y_t, b, w, r)
            b = b_i
            w = w_i
            self.weights.append(w.values)
            self.biases.append(b)
            self.w_f = w.values
            self.b_f = b
            if(i > 1):
                if(self.thresholdMet(self.weights[i - 1], w_i, threshold)):
                    self.rates.append(r)
                    return w
                if(j_t > self.costs[i - 1] and r > 0.0002 or i % 5 == 0 and r > 0.002):
                    if(r > 0.2):
                        r -= 0.05
                    else:
                        r /= 1.0001
                    self.rates.append(r)
        return w
        
    def stoch_gradient_descent(self, data, label_name, r, threshold, num_iter , batch_size = 1):
        # Clear each field member before a new call
        self.costs = []
        self.weights = []
        self.rates = []
        self.weightNorms = []
        self.y_predicts = []
        self.biases = []
        self.w_f = None
        self.b_f = None
        
        # Normalize Data
        data = (data - data.mean())/data.std()
        # Set the shape of the weight vector and subtract 1, because of the label.
        weightData = [0]*(data.shape[1] - 1)
        b = 0
        w = pd.DataFrame(data = weightData)
        
        for i in range(0, num_iter):
            sampleData = data
            counter = 0
            while(len(sampleData) > 0):
                # Pull batch_size members at random.
                sampleElement = sampleData.sample(n = batch_size)
                sampleData = sampleData.drop(sampleElement.index)
                y = sampleElement[label_name]
                x = sampleElement.drop(columns=[label_name])
                y_t = y_pred(x, w, b)
                j_t = cost(b, y_t, y)
                self.costs.append(j_t)
                self.y_predicts.append(y_t)
                b_i, w_i = tune_params(x, y, y_t, b, w, r)
                b = b_i
                w = w_i
                self.weights.append(w.values)
                self.biases.append(b)
                self.w_f = w.values
                self.b_f = b
                if(counter > 1):
                    if(self.thresholdMet(self.weights[counter - 1], w_i, threshold)):
                        self.rates.append(r)
                        print("exited system")
                        return w.values
                    else:
                        r /= 1.004
                        self.rates.append(r)
                counter += 1
        return w.values
    
    def predict(self, x, w, b):
        return (np.dot(x, w + b))
            
                    


dfTest = pd.read_csv('test1.csv')
dfTrain = pd.read_csv('train1.csv')
cols = ['Cement', 'Slag', 'Fly ash', 'Water','SP', 'Coarse Aggr', 'Fine Aggr', 'label']
dfTest.columns = cols
dfTrain.columns = cols




### SIMPLE LINEAR TEST MODEL W/ GAUSSIAN NOISE INTRODUCED
## GRADIENT DESCENT

# reg = LSM_Regression()
# noise = np.random.normal(0,1,100)
# x = np.arange(0, 100, 1)
# m = 0.01
# b = 4.8756
# y = m*x + b + noise
# df = pd.DataFrame(data = x)
# df.columns = ['x']
# df['y'] = y

# reg.gradient_descent(df, 'y', .5, 1e-10, 500)
# df = (df - df.mean())/df.std()
# x = df['x']
# y = df['y']

## END OF GRADIENT DESCENT

## STOCHASTIC GRADIENT DESCENT

# reg = LSM_Regression()
# noise = np.random.normal(0,1,100)
# x = np.arange(0, 100, 1)
# m = 0.2
# b = 4.8756
# y = m*x + b + noise
# df = pd.DataFrame(data = x)
# df.columns = ['x']
# df['y'] = y
# reg.stoch_gradient_descent(df, 'y', .1, 1e-6, 50)
# df = (df - df.mean())/df.std()
# x = df['x']
# y = df['y']

## END OF STOCHASTIC GRADIENT DESCENT

## PLOT OF COST VS EPOCH

# costs = np.asarray(reg.costs)
# steps = np.arange(0, len(reg.costs), 1)
# plt.plot(steps,costs, label="Cost")
# plt.xlabel("Number of Steps")
# plt.ylabel("Cost Function")
# plt.title("Cost per step")

## END OF PLOT OF COST VS EPOCH

## PLOT OF REGRESSION LINE VS SCATTER PLOT

# plt.scatter(x, y, color = 'blue', alpha = 0.3)
# x.values.resize(len(x), 1)
# pred_y = np.dot(x, reg.weights[len(reg.weights) - 1] + reg.biases[len(reg.biases) - 1])
# plt.plot(x, pred_y, color = 'red')

## END OF PLOT FOR REGRESSION LINE VS SCATTER PLOT
### END OF SIMPLE LINEAR TEST MODEL W/ GAUSSIAN NOISE INTRODUCED




### TESTING FOR STOCHASTIC GRADIENT DESCENT CORRECTNESS
# from sklearn.linear_model import SGDRegressor
# dfTrain = (dfTrain - dfTrain.mean())/dfTrain.std()
# y = dfTrain['label'].values
# x = dfTrain.drop(columns = ['label']).values
# clf = SGDRegressor(loss="squared_error", penalty="l2", tol = 1e-6)
# clf.fit(x, y, coef_init = [0]*len(x[0]), intercept_init = 0)

# from scipy import stats ## USED FOR SLOPE/INTERCEPT CORRECTNESS TEST
### END OF STOCHASTIC GRADIENT DESCENT CORRECTNESS



### ACTUAL DATA WITH COST PLOT: GRADIENT DESCENT

reg = LSM_Regression()
w = reg.gradient_descent(dfTrain, 'label', 1, 1e-10, 1000)
costs = np.asarray(reg.costs)
steps = np.arange(0, len(reg.costs), 1)
plt.figure(1)
plt.plot(steps, costs, label="Cost")
plt.xlabel("Number of Steps")
plt.ylabel("Cost Function")
plt.title("Cost per step")

print("Grad Descent: ", w)

## PLOT OF Y_PRED VS ACTUAL TEST Y

# dfTest = (dfTest - dfTest.mean())/dfTest.std()
# y = dfTest['label'].values
# x = dfTest.drop(columns = ['label']).values
# y_predict = reg.predict(x, reg.w_f, reg.b_f)
# plt.figure(2)
# plt.plot(y, color = 'blue', label = "actual")
# plt.plot(y_predict, color = 'orange', label = "predicted")
# plt.title("Actual label Relative to Predicted Label")
# plt.legend()

## END OF Y_PRED VS ACTUAL TEST Y PLOT


### END OF ACTUAL DATa: GRADIENT DESCENT


### ACTUAL DATA WITH COST PLOT: STOCHASTIC GRADIENT DESCENT
# batch_size = 2, 3 works very consistently. size = 1 is somewhat consistently.
# r = 0.3, t = 2000
reg = LSM_Regression()
w = reg.stoch_gradient_descent(dfTrain, 'label', 0.3, 1e-6, 100, batch_size = 1)
costs = np.asarray(reg.costs)
steps = np.arange(0, len(reg.costs), 1)
plt.figure(1)
plt.plot(steps,costs, label="Cost")
plt.xlabel("Number of Steps")
plt.ylabel("Cost Function")
plt.title("Cost per step")
print("Stoch Grad Descent: ", w)


## PLOT OF Y_PRED VS ACTUAL TEST Y

# dfTest = (dfTest - dfTest.mean())/dfTest.std()
# y = dfTest['label'].values
# x = dfTest.drop(columns = ['label']).values
# y_predict = reg.predict(x, reg.w_f, reg.b_f)
# plt.figure(2)
# plt.plot(y, color = 'blue', label = "actual")
# plt.plot(y_predict, color = 'orange', label = "predicted")
# plt.title("Actual label Relative to Predicted Label")
# plt.legend()

## END OF Y_PRED VS ACTUAL TEST Y PLOT



### END OF ACTUAL DATA: STOCHASTIC GRADIENT DESCENT


# print("gd weights: ", reggd.weights[len(reggd.weights) - 1])
# print("sgd weights: ", reg.weights[len(reg.weights) - 1])



### SCATTER PLOTS AND SQUIGGLY PLOTS WITH TEST SPLIT
# dfTest = (dfTest - dfTest.mean())/dfTest.std()
# y = dfTest['label'].values
# x = dfTest.drop(columns = ['label']).values

# colors = cm.rainbow(np.linspace(0, 1, len(x[0])))

# for i in range(0, len(x[0])):
#     plt.figure(i)
#     plt.scatter(x[:,i], y, color=colors[i], label = cols[i])
#     plt.xlabel('Attributes')
#     plt.ylabel("Label output")
#     plt.title("Scatter Plot of training Data")
#     plt.legend()
#     pred_y = np.dot(x, reg.weights[len(reg.weights) - 1] + reg.biases[len(reg.biases) - 1])
#     plt.plot(x[:, i], pred_y, label = "Predicted Line")
### END OF FUN TESTING
            
            
