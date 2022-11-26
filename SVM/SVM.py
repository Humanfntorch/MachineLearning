# -*- coding: utf-8 -*-
"""
@author: Jacob Rogers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SVM:
    
    def __init__(self):
        self.weights = []
        self.losses = []
        self.biases = []
        self.w_0 = None
        self.b_f = None
        # Tunable Params
        self.C = None
        self.gamma = None
        
    def tune_params(self, x, y, w, maxVal):
        if(maxVal <= 0):
            self.w_0 = (1 - self.gamma)*self.w_0
        else:
            w = w - (self.gamma*self.w_0) + self.gamma*self.C*len(x)*np.dot(x.T, y)
        return w
    
    def hinge_loss(self, x_i, y_i, w):
        return(max(0, 1 - sum(y_i.values*(np.dot(x_i.values, w)))))
    
    def predict(self, x, w):
        # compute dot product for prediction and compute the sign (< 0 -> -1, > 0 -> 1, = 0 -> 0)
        y = np.sign(np.dot(x, w))
        # convert any 0 vals to -1
        zeroIdx = np.where(y == 0)[0]
        for idx in zeroIdx:
            y[idx] = -1
        return y
    
    def Stoch_SVM(self, data, label_name, gamma_0, C, num_iter , gammaParam = 1, batch_size = 1):
        
        if(gamma_0 <= 0):
            print("Initial gamma parameter value must be a non-negative value greater than zero.")
            return
        
        # Clear each field member before a new call
        self.losses = []
        self.weights = []
        self.rates = []
        self.weightNorms = []
        self.biases = []
        self.w_0 = None
        self.b_f = None
        self.C = C
        # Set the shape of the weight vector and subtract 1, because of the label.
        self.w_0 = np.zeros([data.shape[1] - 1, ])
        w = self.w_0
        for i in range(0, num_iter):
            sampleData = data
            losses = 0
            if(gammaParam == 1):
                a = (0.01)**i
                if a == 0:
                    a = 1e-9
                self.gamma = gamma_0/(1 + (gamma_0/a)*i)
            else:
                self.gamma = gamma_0/(1 + i)
            while(len(sampleData) > 0):
                # Pull batch_size members at random.
                sampleElement = sampleData.sample(n = batch_size)
                sampleData = sampleData.drop(sampleElement.index)
                y = sampleElement[label_name]
                x = sampleElement.drop(columns=[label_name])
                loss = self.hinge_loss(x, y, w)
                losses += loss
                w = self.tune_params(x, y, w, loss)
            self.weights.append(w)   
            self.losses.append(losses)
        return w
    
class Soft_SVM:
    
    def __init__(self):
        self.weights = None
        self.support_vectors = None
        self.support_labels = None
        self.alpha = None
        self.bias = None
        # Tunable Params
        self.C = None
        self.gamma = None
        self.kernel = None
        self.kernel_type = None
        self.gramMtx = None
        self.y = None
        self.x = None
        self.alpha_p = None
       
        
    def gram_matrix(self, k, y):
        return(np.outer(y, y) * k)
    
    def gaussian_kernel(self, x_i, x_j, gamma):
        return np.exp(-(1 / self.gamma) * np.linalg.norm(x_i[:, np.newaxis] - x_j[np.newaxis, :], axis=2) ** 2)
    
    def linear_kernel(self, x_i, x_j):
        return ((self.C + x_i.dot(x_j.T)))
    
    def objective_func(self, gramMtx, alpha):
        return (alpha.sum() - 0.5 * alpha.dot(alpha.dot(gramMtx)) )
    
    def jacobian(self, gramMtx, alpha):
        return(np.ones(len(alpha)) - alpha.dot(gramMtx))
     


    #  Valid kernel_types: 
    # 'linear' Uses the linear kernel given x/y, gamma is not used. Default value.
    # 'gauss' Uses the gaussian kernel given x/y and gamma, gamma must be non-negative and > 0. 
    def svm(self, data, label_name, C, gamma, kernel_type = 'lin'):
        if(kernel_type == 'gauss'):
            assert(gamma > 0)
            
            # Reset all values
            self.weights = None
            self.support_vectors = None
            self.support_labels = None
            self.alpha = None
            self.bias = None
            # Tunable Params
            self.C = None
            self.gamma = None
            self.kernel = None
            self.kernel_type = None
            self.gramMtx = None
            self.y = None
            self.x = None
            self.alpha_p = None
            
        
        # Convert to NP arrays using values
        y = data[label_name]
        x = data.drop(columns = [label_name])
        x = x.values
        y = y.values
        self.C = C
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.y = y
        self.x = x
        
        # Determine Gram matrix using given kernel
        if(kernel_type == 'gauss'):
            k = self.gaussian_kernel(x, x, gamma)
            gramMtx = self.gram_matrix(k, y)
        else:
            k = self.linear_kernel(x, x)
            gramMtx = self.gram_matrix(k, y)
        self.kernel = k
        self.gramMtx = gramMtx
        
        # define constraints for scipy optimize solver:
        # Constraint 1: sum(alpha_i*y_i) = 0 -> dot(alpha, y) = 0
        # Constraint 2: 0 <= alpha_i <= C -> C - alpha_i >= 0
        # placeholders for constraints
        constraints = ({'type' : 'eq', 'fun' : lambda alpha : np.dot(alpha, y), 'jac' : lambda alpha : y})
        
        # Bound alpha using constraint 2
        bounds = [(0, C)]*x.shape[0]
        # Negate the minimization in order to optimize maximally
        alpha_prime = minimize(fun = lambda alpha : -self.objective_func(gramMtx, alpha), x0 = np.ones(len(y)), 
                               method = 'SLSQP', jac = lambda alpha : -self.jacobian(gramMtx, alpha), 
                               constraints = constraints, bounds = bounds)
        
        # Extract optimal alphas and support vectors/labels + weights. 
        self.alpha = alpha_prime.x
        
        # Threshold cutoff where alpha has a meaningful distance from margin.
        # eliminates significantly small values of alpha
        thresh = 1e-8
        
        self.support_vectors = x[self.alpha > thresh]
        # defined as all y's where alpha_i > 0
        self.support_labels = y[self.alpha > thresh]
        # Defined as sum of product between alphas and kernel function
        self.alpha_p = self.alpha[self.alpha > thresh]
        
        self.weights = np.dot(self.support_vectors.T*self.support_labels, self.alpha_p)
        # Defined as difference between support label and product of support vectors*weights
        if(kernel_type == 'lin'):
            self.bias = self.support_labels[0] - (self.alpha * y).dot(self.linear_kernel(x, self.support_vectors[0]))
        else:
            self.bias = self.support_labels[0] - (self.alpha * y).dot(self.gaussian_kernel(x, self.support_vectors[0], gamma))

    def prediction(self, x):
        if(self.kernel_type == 'lin'):
            return (np.sign(np.sum(self.linear_kernel(self.support_vectors, x).T*self.alpha_p*self.support_labels, axis = 1) + self.bias))
        else:
            return (np.sign((self.alpha_p*self.support_labels).dot(self.gaussian_kernel(x.values, self.support_vectors, self.gamma).T)))

    
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
yTest = dfTest['label']
yTrain = dfTrain['label']
xTest = dfTest.drop(columns = ['label'])
xTrain = dfTrain.drop(columns = ['label'])


C = [100/873, 500/873, 700/873]
T = 100


### Problem 1: STOCHASTIC GRADIENT DESCENT 

print("Stochastic Gradient Descent SVM Primal Form Results: ")
gamma_0 = 0.02
svm = SVM()

# Primal Testing using C[0:2]
w = svm.Stoch_SVM(dfTrain, 'label', gamma_0, C[0], T, gammaParam = 1)
y_p = svm.predict(xTrain, w)
print("C = 100/873 Train Error (percentage): ", (sum(y_p != yTrain) / len(yTrain))*100)
y_p = svm.predict(xTest, w)
print("C = 100/873 Test Error (percentage): ", (sum(y_p != yTest) / len(yTest))*100)
print("Weight: ", w)
print()

# losses = np.asarray(svm.losses)
# steps = np.arange(0, len(svm.losses), 1)
# plt.figure(1)
# plt.plot(steps, losses, label="Losses")
# plt.xlabel("Number of Steps")
# plt.ylabel("Loss Function")
# plt.title("Loss per step")

w = svm.Stoch_SVM(dfTrain, 'label', gamma_0, C[1], T, gammaParam = 1)
y_p = svm.predict(xTrain, w)
print("C = 500/873 Train Error (percentage): ", (sum(y_p != yTrain) / len(yTrain))*100)
y_p = svm.predict(xTest, w)
print("C = 500/873 Test Error (percentage): ", (sum(y_p != yTest) / len(yTest))*100)
print("Weight: ", w)
print()

# losses = np.asarray(svm.losses)
# steps = np.arange(0, len(svm.losses), 1)
# plt.figure(2)
# plt.plot(steps, losses, label="Losses")
# plt.xlabel("Number of Steps")
# plt.ylabel("Loss Function")
# plt.title("Loss per step")

w = svm.Stoch_SVM(dfTrain, 'label', gamma_0, C[2], T, gammaParam = 1)
y_p = svm.predict(xTrain, w)
print("C = 700/873 Train Error (percentage): ", (sum(y_p != yTrain)/len(yTrain))*100)
y_p = svm.predict(xTest, w)
print("C = 700/873 Test Error (percentage): ", (sum(y_p != yTest)/len(yTest))*100)
print("Weight: ", w)
print()

# losses = np.asarray(svm.losses)
# steps = np.arange(0, len(svm.losses), 1)
# plt.figure(3)
# plt.plot(steps, losses, label="Losses")
# plt.xlabel("Number of Steps")
# plt.ylabel("Loss Function")
# plt.title("Loss per step")

### END PROBLEM 1


### Problem 2: Soft SVM using linear + Gaussian Kernels

# Linear Kernel:
print("Linear Kernel (Dual form) Results:")
C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], 100, kernel_type = 'lin')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Weights: ", svm.weights)
print("C = 100/873 bias: ", svm.bias)

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], 100, kernel_type = 'lin')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Weights: ", svm.weights)
print("C = 500/873 bias: ", svm.bias)

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], 100, kernel_type = 'lin')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Weights: ", svm.weights)
print("C = 700/873 bias: ", svm.bias)


# Gaussian Kernel: 
gamma = [0.1, 0.5, 1, 5, 100]
print("Gaussian Kernel Results:")
print("gamma = 0.1")
# C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], gamma[0], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], gamma[0], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], gamma[0], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Number of Support Vectors: ", len(svm.support_vectors))

print()
print("gamma = 0.5")
# C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], gamma[1], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], gamma[1], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], gamma[1], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Number of Support Vectors: ", len(svm.support_vectors))

print()
print("gamma = 1")
# C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], gamma[2], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], gamma[2], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], gamma[2], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Number of Support Vectors: ", len(svm.support_vectors))

print()
print("gamma = 5")
# C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], gamma[3], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], gamma[3], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], gamma[3], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Number of Support Vectors: ", len(svm.support_vectors))

print()
print("gamma = 100")
# C[0]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[0], gamma[4], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 100/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 100/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 100/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[1]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[1], gamma[4], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 500/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 500/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 500/873 Number of Support Vectors: ", len(svm.support_vectors))

# C[2]
svm = Soft_SVM()
svm.svm(dfTrain, 'label', C[2], gamma[4], kernel_type = 'gauss')
ypredTrain = svm.prediction(xTrain)
ypredTest = svm.prediction(xTest)
print("C = 700/873 Train Error (percentage): ", (sum(yTrain != ypredTrain)/len(yTrain))*100 )
print("C = 700/873 Test Error (percentage): ", (sum(yTest != ypredTest)/len(yTest))* 100 )
print("C = 700/873 Number of Support Vectors: ", len(svm.support_vectors))





    
    
