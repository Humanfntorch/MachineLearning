# Implementation of both hard and soft SVM (Gaussian/RBF kernel and linear kernel, solved using dual form solution) binary classification scheme. Determines the optimal set of weight vectors
given a set of input data and used to form a binary prediction of fitted data (-1 = False; 1 = True).

# Class: SVM:
  # Fields:
   * self.weights = [] -> The set of weights determined at epoch i in T
   * self.losses = [] -> The set of calculated hinge loss for a given weight at epoch i in T
   * self.biases = [] -> [Deprecated for hard SVM solution]
   * self.w_0 = None -> The initial given weight vector. Defaulted initialization is 0 vector
   * self.b_f = None -> [Deprecated for hard SVM solution]
   * self.C = None -> The hyper paramater to form upper bound for hinge loss weight adjustments. 
   * self.gamma = -> The learning rate parameter updated at epoch i in T
 
 
  
  All other methods are assumed to be private. Use at own risk.
