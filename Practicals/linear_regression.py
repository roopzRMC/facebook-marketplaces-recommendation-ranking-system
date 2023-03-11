'''
Fetch the California dataset of California real estate prices, and assign the labels and features to variables called X and y. 
Print the shape of the features and labels.
Divide the dataset into train, validation and test subsets, using the model_selection.train_test_split method. 
    Hint - you will need to apply the method twice to generate the three subsets.

'''

# %%
from sklearn import datasets, model_selection
import numpy as np
# %%
X, y = datasets.fetch_california_housing(return_X_y=True)
# %%
X.shape
# %%
y.shape
# %%
print(type(X))
# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                    random_state=42)
# %%
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25,
                                                                  random_state=42)
# %%
print(X_train.shape)
print(y_train.shape)
# %%
'''
Create a class called LinearRegression. 
The class should have two methods - the class constructor, 
which needs to randomly assign initial weights for each feature, 
and set a random seed for reproducibility. 

A method called __call__ that runs when we call an instance of the class on some data, 
and returns a prediction based on the features in X.

'''

class LinearRegression:
    def __init__(self, n_features):
        np.random.seed(42)
        self.wt = np.random.randn(n_features, 1)
        self.B = np.random.randn(1)

    def __call__(self, X):
        ypred = np.dot(X, self.wt) + self.B
        return ypred
    
    def update_parameters(self, wt, B):
        self.wt = wt
        self.B = B


    


    

# %%

'''
Create an instance of LinearRegression and use it to get the predictions based on the initial weights. 
Print the first 10 examples. Now print the first 10 actual values of y. What do you notice?
'''
# %%
mymodel = LinearRegression(n_features=8)
# %%
ypred = mymodel(X_train)
# %%
print(ypred[:10])
# %%

'''
We now need to tell the model how to improve. Add a new method to the LinearRegression class, called update_parameters. 
This method should update the model's weight and bias attributes to new values which are passed to the method as parameters.

'''

# %%
def get_mse(ypred, ytrue):
    errors = ypred - ytrue
    squared_errors = errors ** 2
    return np.mean(squared_errors)

def minimize_loss(X_train, y_train):
    X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    optimal_w = np.matmul(
        np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
        np.matmul(X_with_bias.T, y_train)
    )
    return optimal_w[1:], optimal_w[0]


cost = get_mse(ypred, y_train)
print(cost)



weights, bias = minimize_loss(X_train, y_train)
print(weights, bias)
# %%
'''

Define a function inside linear_regression.py that takes in the features and labels of the training set and calculates the optimum values of the weights and biases for each feature. 
Call this function on X_train and y_train and assign the outputs to variables called weights

'''

weights, bias = minimize_loss(X_train, y_train)
print(weights)
print(bias)
# %%

mymodel.update_parameters(wt = weights, B = bias)
# %%
#ynewpred = mymodel(X_train)
# %%
ypred2 = mymodel(X_train)
# %%
cost2 = get_mse(ypred2, y_train)
# %%
cost2
# %%
