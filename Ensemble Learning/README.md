
Implemented variations of ensemble learning using a decision tree classifier. Includes: AdaBoost, Bagged Trees and Random Forest. 
Requires the import of DecisionTree.py.
There are two primary classes: AdaBoost and BaggedTrees. The description of each class and their methods are given below:

# Class: AdaBoost
* Object -> Controls method implementation of the adaboost algorithm.
# Fields Include:
* alphas = [] -> Error calculation of stump i using log of misprediction count for stump i.
* stumps = [] -> set of cached stumps in this
* T = x (default 1: integer value assumed) -> The number of stump epochs to build
* predictErrors = [0]*T -> The number of prediction errors for all stumps[0:T-1]
* Initialization as: ada = AdaBoost()

# Builds T decision tree classifiers (of depth 1) using the given data, implemented using the adaboost algorithm to boost unseen variations of data for future stump classifiers to be built from.
# Method: build_stumps(self, data, label_name, T)
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* T -> The number of Decision Stumps to be built. 
* Called by AdaBoost object.

# Converts all values with 'unknown' to the majority labels value.
# predict_vote(self, data, label_name, T):
* data -> Pandas DataFrame of data containing values to use for ada predictions
* label_name -> String representation of the column containing the label
* T -> The number of stumps to use in adaboost prediction calculation.
* Returns: h_final -> The final voted prediction determined using T stumps.

 Class: BaggedTrees
* Object -> Controls method implementation of the adaboost algorithm.
# Fields Include:
* trees = [] -> The set of all constructed trees
* bootStrapVal = x (default 1: integer value assumed) ->  The number of bootstrap samples to randomly pull from the given data.
* * alphas = [] -> Error calculation of tree i using log of misprediction count for stump i (deprecated)
* T = x (default 1) -> The number of stump epochs to build
* predictErrors = [0]*T -> The number of prediction errors for all stumps[0:T-1]
* Initialization as: bt = BaggedTrees()

# Generates a set of T trees with maximum tree depth = treeDepth (if left out, generates a fully expanded tree) uing bootstrap number of data points to build tree classifer.
# Method buildBaggedTrees(self, data, label_name, T, treeDepth = None):
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* T -> The number of Decision Stumps to be built. 
* Called by AdaBoost object.
* treeDepth -> The cutoff for the depth of the constructed tree.

# Generates a set of T trees with maximum tree depth - treeDepth (if let out, generates a fully expanded tree), using a given set of num_attributes to randomly choose from in the given data set.
# Random_Forest(self, data, label_name, num_attributes, T, treeDepth = None):
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* num_attributes -> A set [], (), {} of integer values that are used to determine how many attributes should be analyzed using the information gain function in decision tree construction.
* T -> The number of Decision Stumps to be built. 
* Called by AdaBoost object.
* treeDepth -> The cutoff for the depth of the constructed tree.

# Calculates the voted prediction from a set of T bagged trees/random forest tree representations
# vote(self, data, label_name, T):
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* T -> The number of trees to use for voted prediction calculation.

All other methods are assumed to be private. Use at own risk.


