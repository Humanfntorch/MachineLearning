import numpy as np
import pandas as pd
import time

def y_pred(x, w, b):
    # Calculates the inner product of cols, sums over the vals and stores them in rows
    return (b + np.dot(x, w))
def cost(y_i, y):
    # y - w^Tx_i
    y_cost = (y.values - y_i)
    # J(w) = 1/2*Sigma || y - w^Tx_i ||_1
    return(1/2*np.sum(np.dot(y_cost.T, y_cost)))

# x -> Testing Attributes (Expects Pandas.DataFrame)
# y -> Testing Labels (Expects Pandas.DataFrame)
# y_pred -> Predicted Labels 
# b -> Bias Parameter
# w -> Weight Vector Parameter
# r -> Learning Rate
def tune_params(x, y, y_pred, b_old, w_old, r):
    # del_ denotes partial w.r.t given param
    # POSSIBLY NEEDS TO BE Y_PRED - Y
    del_b = (np.sum(y.values - y_pred))/len(y)
    #y_diff = y - y_pred
    del_w = -1*np.sum(np.dot((y.values - y_pred), x))/len(y)
    b_new = (b_old - r*del_b)
    w_new = (w_old)- r*(del_w)
    return(b_new, w_new)

class LSM_Regression:

    def __init__(self):
        self.weights = []
        self.costs = []
        self.rates = []
        self.weightNorms = []

    def thresholdMet(self, w_old, w_new, threshold):
        w_norm = np.norm(w_new - w_old)
        return(w_norm < threshold)

    def Gradient_Descent(self, data, label_name, r, threshold, T):
        self.costs = []
        self.weights = []
        self.rates = []
        self.weightNorms = []
        
        data = (data - data.mean())/data.std()
        y = data[label_name]
        x = data.drop(columns=[label_name])
        weightData = [0]*x.shape[1]
        b = 0
        w = pd.DataFrame(data = weightData)
        r = 1
        
        for i in range(0, T):
            y_t = y_pred(x, w, b)
            j_t = cost(y_t, y)
            self.costs.append(j_t)
            b_i, w_i = tune_params(x, y, y_t, b, w, r)
            b = b_i
            w = w_i
            print("Weights: ", w)
            print("B: ", b)
            time.sleep(2)
            self.weights.append(w_i)
            if(i > 1):
                if(j_t < self.costs[i - 1]):
                    r -= 0.1
                    self.rates.append(r)
                    print("finally: ", i)


dfTest = pd.read_csv('test1.csv')
dfTrain = pd.read_csv('train1.csv')
cols = ['Cement',
'Slag',
'Fly ash',
'Water',
'SP',
'Coarse Aggr',
'Fine Aggr', 'label']
dfTest.columns = cols
dfTrain.columns = cols

reg = LSM_Regression()
reg.Gradient_Descent(dfTrain, "label", 1, 1e-6, 200)




            
