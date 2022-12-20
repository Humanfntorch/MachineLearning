This project is for the CS6350 course Kaggle Competition project.
The task is to predict whether an individual will have an income > $50k a year based on the 1994 census data collected.
The data used for training is specified under the 'train_final.csv' file, while the data used for forming the final testing prediction is 'test_final.csv'.
The core file used for data pre-processing, training and testing is: KaggleDt.py and an extension file NN.py is included for one of the testing models.
Four machine learning algorithms were used to form a predictive model:
Random Forest Regression (from sklearn)
Stochastic Gradient Descent (from sklearn)
SVR - based on SVM (from sklearn)
Neural Network (based on my implementation, for more information see: https://github.com/Humanfntorch/MachineLearning/blob/main/Neural%20Networks/README.md)

* The data is read in from the csv files using Pandas (into dataframes).
* Seaborn was used to modify a correlation matrix plot created of all data attributes against the output (income prediction).
* Data is scaled using either MinMax or StandardScaler from sklearn (one block of code is commented out, the other is being issued. When wanting to change to the other     form, simply comment out the issued block and uncomment the other block).
* The categorical labels are converted to a histogram binned numerical value using the total number of categorical attributes for a given label and the total occurence     of those labels (more occurence occurs a higher numerical value, less occurences corresponds to a lower numerical value).
* The attribute 'fnlwgt' may be dropped from the dataframe by uncommenting the code block (identified by the comment above), but was left in the dataframe during           testing, due to increased training accuracy when it was remained, however, the correlation matrix shows a negative impact which may result in a lower testing accuracy. 
* Each algorithmic model is first trained on the training dataset and the accuracy on the training data set is predicted by converting the values > 0.5 to a 1 output,      all other values are converted to 0. The accuracy of the model is printed to the console. The testing predictions are calculated using the trained model and              subsequently written to a corresponding .csv file (the name should be changed in the string representation if this is being used), along with the predictions ID          number. NOTE: the income predictions are left as regression calculated values as the Kaggle competition used an area under the curve testing mode to determine the        accuracy of the prediction (as compared to binary predictions). 

Due to the heavy computations involved in each model, the execution time can be quite heavy, therefore each model is commented out and only a single model is left for testing. Currently, the neural network implementation using 6 layers is issued. To test the other models, comment out the neural network and uncomment the model wished to be tested (each model is identifiable by the comment above the code: ### MODEL CLASSIFIER ... // code body here ... ### END MODEL CLASSIFIER)

There is a single method: 
def complete_missing_value(df):
which accepts a Pandas data frame and replaces all missing values with the one of the top 3 majority attributes for the given attribute categories. The current code simply removes the missing data as it is only a small percentage of the total data (with a similar missing percentage seen in the testing data), which seemed like a logical reason to simply remove versus cleanse.

For more detailed information on the project, the pre-processing stage, machine learning models used (why and how parameters were chosen), testing accuracies and discussion, please read the report included in the folder titled: "Final Report.pdf".
