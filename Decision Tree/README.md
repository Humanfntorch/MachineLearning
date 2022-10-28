# Recursively creates a Decision Tree classifer with the specified arguments. Returns calculated decision tree as nested dictionary.

# decision_tree(data, label_name, maxDepth, splitAlgorithm, maxChildren:float = np.inf, randForest = None): 
* data -> a Pandas Dataframe of all data (including column names and label data); 
* label_name -> Name of label column as type string
* maxDepth -> The maximum depth wanted for the tree
* splitAlgorithm -> The algorithm used to split attributes. Valid flags include: ENTROPYFLAG, MEFLAG (majority error), and GINIFLAG.
* maxChildren -> The maximum number of leaf nodes per parent node. Default is is np.inf
* randForest -> A set of numerical values specifying the number of attributes that should be randomly chosen during each split. Paramater can be any datatype that is 
* convertable to a list. Default is None.
* ~ Returns: calculated decision tree as a nested dictionary

# Converts all values with 'unknown' to the majority labels value.
# convert_unknown(df, label_names)
* df -> Pandas DataFrame of data containing value needed for conversion
* label_names -> list of columns containing 'unknown'
* ~ Returns: void

# Converts all numerical attributes to binary attributes based on the median value. All values <= median = 0, all values > median = 1
# convert_numerical_binary(df, label_names):
* df -> Pandas DataFrame of data containing value needed for conversion
* label_names -> list of columns containing 'unknown'
* ~ Returns: void

# Calculates the prediction error of a decision tree, dt, using data with label, label_name. Returns a Pandas DataSeries with predicted values and prediction error.
# def prediction(dtree, data, label_name)
* dtree -> Decision Tree Classifier, as nested dictionary
* data -> Pandas DataFrame containing data to classify (including columns names)
* label_name -> Name of label, type string, of data to predict against.
* ~ Returns: Pandas DataSeries of all predictions, prediction error calculated as: (incorrect)/(incorrect + correct) predictions.


All other methods are assumed to be private and not callable.
NOTE: Some details with implementation need to be addressed: Forming a class. Fixing depth issues. Organizing code and eliminating code smell.
