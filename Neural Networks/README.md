# Implementation of a Neural Network that utilizes the sigmoid activation function (sigma = (1 + exp(-x))^-1) and stochastic gradient descent with n number of layers to solve binary classification problems.

# Class: Node:
  # Initializes a node in the network graph. 
   # Each node has at least one incoming edge (base value or calculated)
   # Each node should have at least one outgoing edge
   # Each node has the ability to use activation function + derivative
  # Fields:
   * self.in_edge  -> weighted edge values going into this.
   * self.out_edge -> weighted edge values going out of this.
  # Methods:
  
  
# Uses stochastic gradient descent with num_iter number of epochs to iteratively optimize weights using the primal form solution of the SVM algorithm. 
  # Stoch_SVM(self, data, label_name, gamma_0, C, num_iter , gammaParam = 1, batch_size = 1):
    * data -> a Pandas Dataframe of all data (including column names and label data); 
    * label_name -> Name of label column as type string
    * gamma_0 -> The initial value used for the learning rate (required to be > 0).
    * C -> The scaling value for the hinge_loss error calculation (used to update weights)
    * num_iter -> The number of iterations that the dataset is used to calculate the optimal set of weights
    * gammaParam -> Flag for which learning rate schedule to use. Schedules are as follows:
                    gammaParam = 1 (default): gamma_0/(1 + (gamma_0/a)*i
                    gammaParam = 2:  gamma_0/(1 + i)
      where i is the current iteration in num_iter and a follows the recipe: a = (0.01)^i, if a ~= 0 -> a = 1E-9
      All other paramater values result in undefined behavior.
    *  batch_size -> The number of data points to randomly use in hinge loss/weight calculations at one instance. Note: The entire dataset
      is used before entering into a next epoch in num_iter. Batch_size simply controls the amount of calculations being made at a single instance in time.
      Theoretically should help when dealing with large datasets.
    * Returns: The optmized weight calculated on the final epoch determined by num_iter. 
     
# Determines the binary classification prediction using the input dataset x and the weight vector w.    
  # predict(self, x, w):
    * x -> Input dataset (Expected as a Pandas dataframe)
    * w -> Weight vector used to form prediction of the form y_hat = mx + b [bias term assumed to be rolled into weight vector]
    * Returns: Binary classification prediction of size N (assuming x is size NxM and w is size M
    
# Implementation of the soft SVM solution (using dual form solution, optimized using Lagrange multipliers and the 'SLSQP' solution optimizer package from Scikit). Calculates the optimal weights using either a linear or Gaussian (RBF) kernel and can be used to form subsequent future predictions of a given input dataset using optmized weights.  
# Class: Soft_SVM    
  # Fields:
   * self.weights = None -> The set of weights determined through Lagrange optimization (Linear Kernel only. Gaussian Kernel is incorrect).
   * self.support_vectors = None -> The set of support vectors defined by a threshold > 1e-8
   * self.support_labels = None -> The set of supporting labels determined by the support vectors from the Lagrange optimization
   * self.alpha = None -> Optimized Lagrange vector
   * self.bias = None -> The recovered bias (Linear Kernel only. Gaussian kernel may/may not be correct)
   * self.C = None -> The scaling parameter used to bound the reach of the lagrangian optimization problem
   * self.gamma = None -> The learning rate used in optimization (Gaussian kernel only)
   * self.kernel = None -> The calculated kernel vector determined by user's kernel choice
   * self.kernel_type = None -> The type of kernel used in optimization (Linear or Gaussian)
   * self.gramMtx = None -> Gram matrix calculated using the kernel vector
   * self.y = None -> The input labels given by user
   * self.x = None -> The input values given by user
   * self.alpha_p = None -> Optimized Lagrange values above the threshold 1e-8

# Dual form solution to the SVM algorithm. Determines the optimal lagrange values given the upper bound hyperparameter C (and gamma if Guassian kernel is chosen) and calcualtes a set of binary classification predictions given the lagrange vectors.
  # svm(self, data, label_name, C, gamma, kernel_type = 'lin'):
   * data -> a Pandas Dataframe of all data (including column names and label data); 
   * label_name -> Name of label column as type string
   * C -> Hyperparameter determining the upper bound of the Lagrange multipliers.
   * gamma -> Width hyperparameter for Gaussian kernel (Linear kernel doesn't use this value)
   * kernel_type -> Type of kernel to be used in optimization solution.
     Valid kernel types to use in kernel_type: 
     - 'lin' -> linear kernel
     - 'gauss' -> Gaussian kernel (requires a valid gamma > 0)

#  Calculates the binary classification prediction based on the Lagrange optimization problem using the user's chosen kernel;
   # pediction(self, x):
    * x -> Input data used to form classification prediction
    * Returns: Set of binary classification determined by dual form solution using user's kernel choice.
    
