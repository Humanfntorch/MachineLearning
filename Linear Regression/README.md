# Determines the optimal weight vectors for a given set of data with classification label using Gradient Descent and Stochastic Gradient Descent.
# Class: LSM_Regression:
  # Fields:
  * self.weights = [] -> The set of all weights used during gradient descent calculation
  * self.costs = [] -> The set of all costs determined during gradient descent calculation
  * self.rates = [] -> The learning rates used throughout gradient descent
  * self.weightNorms = [] -> The weighted norms used in determining if threshold condition is met
  * self.biases = [] -> The set of all biases encountered throughout the gradient descent
  * self.y_predicts = [] -> The set of all label predictions encountered throughout gradient descent
  * self.w_f = None -> The final weight returned from the gradient descent algorithm
  * self.b_f = None -> The final bias returned from the gradient descent algorithm
 # Initialization of class object: reg = LSM_Regression()
 
 # Determines whether to stop the gradient descent algorithm based on the L1 Norm of weighted values. If threshold is met, convergence is considered to be met and algorithm is stopped, otherwise, gradient descent continues.
# thresholdMet(self, w_old, w_new, threshold):
 * w_old -> Previous weight value encountered.
 * w_new -> Current (new) weight value encountered.
 * threshold -> The specified threshold where the L1 difference between w_old and w_new is met.
  
  # Batch Gradient Descent algorithm uses entire data set to calculate weights for T number of epochs and returns the final weights (if threshold convergence condition is met or T epochs have been iterated).
# gradient_descent(self, data, label_name, r, threshold, T):
  * data -> a Pandas Dataframe of all data (including column names and label data);
  * label_name -> Name of label column as type string 
  * r -> The initial learning rate to be used in the gradient descent
  * threshold -> The threshold to determine whether conditional requirement of convergence is met
  * T -> The number of epochs to use (if convergence isn't met earlier
  ~ Returns: The final calculated optimized weight vector. 
  
# Stochastic gradient descent algorithm uses batch size number of data each calculation to update the weight vector (until entire dataset is seen) for each epoch in T (convergence condition isn't implemented in stochastic gradient descent. All i in num_iter will be iterated through to calculate the final weight vector).
 # stoch_gradient_descent(self, data, label_name, r, threshold, num_iter , batch_size = 1):
  * data -> a Pandas Dataframe of all data (including column names and label data);
  * label_name -> Name of label column as type string 
  * r -> The initial learning rate to be used in the gradient descent
  * threshold -> The threshold to determine whether conditional requirement of convergence is met
  * num_iter -> The number of epochs used to process the gradient descent
  * batch_size -> The number of data points used in each weight calculation (default is at least one data point).
  ~ Returns: The final calculated optimized weight vector. 
 
 # Uses the fitted form of either Gradient Descent or Stochastic Gradient Descent to predict the labeling of a given set of data (eq: y_hat = (x dot w + b)
  # predict(self, x, w, b):
  * x -> The data set used to form prediction
  * w -> The calculated weight vector from either implementation of gradient descent.
  * b -> The calculated bias term from either implementation of gradient descent. 
  ~ Returns: Regression prediction value from calculated weight + bias using x of the form: y = mx + b  

 
  



 
 
