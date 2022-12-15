# Implementation of a Neural Network that utilizes the sigmoid activation function (sigma = (1 + exp(-x))^-1) and stochastic gradient descent with n number of layers to solve binary classification problems.

# Class: node:
  # Initializes a node in the network graph. 
   * Each node has at least one incoming edge (base value or calculated)
   * Each node should have at least one outgoing edge
   * Each node has the ability to use activation function + derivative
  # Fields:
   * self.in_edge  -> weighted edge values going into this.
   * self.out_edge -> weighted edge values going out of this.
  # Methods:
   * sigmoid(self, x) -> Calculates the sigmod of input x
   * p_sigmoid(self, x) -> Calculated the partial derivative of sigmoid given input x
This class is not expected to be initialized or used explicitly in the neural network, it is a base class used for inheritance in the layers class. Use at own risk.
  
  
# Class: Layer(node):
   # Each layer is composed of a defined number of nodes, each with:
    * A set of weights where w_ij belongs to node i and connects to node j
    * The ability to distinguish layer type (excluding output layer) [hidden/non-hidden]
    * The ability to propagate forward. [hidden/non-hidden]
    * The ability to propagate backwards. [hidden/non-hidden]
   # Fields:
    * self.isHidden -> A boolean flag specifying whether the layer is an activation layer (contains nodes with activated weight values from previous layer)
    * self.weights -> The set of weights for the layer (dimension must be given in the form of a tuple: (input, output), for an input of 4 and output of 5, dims = (4, 5)
    * self.bias -> The set of biases for the layer (dimensions are: (1, output))
      * Initialization of a layer can be specified using a random normal distribution or set weights to zero. Default is weights = 'rand', for zero weights: weights = 'zero' must be specified in initialization. 
      * Initializing a normal layer with an input of 3 and output of 5, using randomized weights + biases is as follows: layer = layer((3, 5), False).
      * Initializing a hidden (activation) layer is as follow: layer = layer((), True). NOTE: Hidden layers don't use dimensional arguments, they simply compute the activate form of forward/back propagation.
   # Methods: 
      * forward_propagate(self, x): -> Performs forward propagation for both hidden layers and base layers. Returns the calculated output (and caches in/out edges for each node in the layer).
      * back_propagate(self, x): -> Performs back propagation for both hidden layers and base layers. Returns the calculated output (and caches in/out edges for each node in the layer).
 
 
# Class: neural_network:
  # Infrastructure for neural networking algorithm. Contains user inserted layers, then uses stochastic gradient to perform adjusted weight modifications until the specified number of epochs has been iterated. A trained network can be used to perform binary predictions with a given threshold (default value is 0.5, where any output > 0.5 = 1, otherwise is -1). Uses Mean Square Error Loss (and derivative) to perform weight optimization.
  # Fields:
    * self.learning_rate = init_learn_rate -> The initial learning rate used in weight adjustments. Must be specifed in constructor's arguments.
    * self.layers = [] -> The set of user inputted layers in this.
    * self.losses = [] -> The average loss value at epoch i
    * self.y_preds = [] -> The predicted outputs after network has been trained and prediction has been called.
    * self.accuracy = None -> The accuracy of predicted output relative to given ground truth value (determined in prediction method)
   # Methods:
    * insert_layer(self, layer): -> Inserts an instance of Layer into the network (called by neural_network object):
      - Ex: nn = neural_network(0.01); layer = layer((5, 3), False); nn.insert_layer(layer)
    * train(self, x, y, epochs): -> Perform forward and back propogation on input data set x (expected as numpy array with dimension (x.shape[0], 1, x.shape[1])) and initializes the netowrk's weights based on the MSE loss from ground truth output y (expected as numpy array). Iterated over the entire dataset using epochs iterations. Method assumes user has added specified number of layers + hidden layers to sufficiently compute output.
    * prediction(self, x, y, threshold = 0.5): -> Computed the predicted output from input data x and measures accuracy of network against ground truth labels y. (Assumes x and y are both numpy arrays and x has dimensions (x.shape[0], 1, x.shape[1]). Returns the predicted outputs and caches the accuracy score (accesible by calling nn.accuracy)
    
    
    
  
   
