# Implementation of the Perceptron algorithm that determines a predicted binary classification based on calculated weights using given input data, learning rate, epoch (number of iterations to calculate) and prediction scheme: standard, voted and averaged.

# Class: Perceptron:
  # Fields:
   * self.weights = None -> The set of calculated weights under T epochs 
   * self.errors = None -> The calculated error given a set of weights at epoch 0, 1, ..., T 
   * self.predictions = None -> The set of predictions determined with weight at epoch 0, 1, ..., T
   * self.counts = None -> The lifespan (longevity) of a certain weight. Approximately the number of correct predictions made by weight self.weights[i]
   * self.avgWeight = None -> The calculated average weight from the set of self.weights
   # Initialization of object: per = Perceptron()
 
 # Implementation of the Perceptron algorithm, fits the given input data and determines the optimal set of weights, along with the predictions determined throughout      the optimization of the weights and the errors encountered with each weight. 
 # perceptron(self, data, label_name, r, epoch = 1):
  * data -> a Pandas Dataframe of all data (including column names and label data);
  * label_name -> Name of label column as type string 
  * r -> The learning rate used to optimize the weights based on prediction errors.
  * epoch -> The number of iterations used to determine the optimal set of weights for predictions.
  ~ Returns: The final optimized weight as determined by the perceptron algorithm.
  
  
# Calculates the binary prediction calculated from input data, x, and the weight vector given.
 # predict(self, x, w):
  * x -> Input data for which binary prediction is needed (size: NxM)
  * w -> Weight vector used to calculate binary prediction of the form y = sgn(wx) (size = M)
  ~ Returns: The determined binary prediction (-1 = False; 1 = True) (size = N)
 
 # Determines the accuracy for the fitted perceptron object given the input data x, the true prediction y, and the variation of the perceptron prediction scheme. Primary method for determining how well the perceptron algorithm was able to fit and predict against given data. 
 Valid variations for perceptron prediction schema: 
 'std' -> Uses the final determined optimized weight value determined in the initial fit of perceptron object
 'avg' -> Uses the average perceptron prediction from all encountered sets of weights seen during fit (including total longevity of weights). Preferred variant
 'vote' -> Uses the voted average perceptron prediction from all encountered sets of weights seen during fit (including total longegivty of weights). 
  # accuracy(self, x, y, variant = 'std'):
   * x -> Input data used to calculate prediction binary labeling scheme (expected as numpy array)
   * y -> The true binary labeling scheme used to calculate accuracy of fitted perceptron model.
   * variant -> The variant used to calculate perceptron prediction, default is standard (see above for further variations).
  
  All other methods are assumed to be private. Use at own risk.
