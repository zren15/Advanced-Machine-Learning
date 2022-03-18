import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Given X, a array y and num_iter return y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (is the shinkage)
    Outputs:y_mean, array of trees from DecisionTreeRegression
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    fm = y_mean
    ### BEGIN SOLUTION
    y_pred = np.ones(len(X))*fm
    for i in range(num_iter):
        res = y - y_pred
        clf = DecisionTreeRegressor(max_depth=max_depth)
        clf.fit(X,res)
        y_pred += nu * clf.predict(X)
        trees.append(clf)
    ### END SOLUTION
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean, nu=0.1):
    """Given X, trees, y_mean predict y_hat
    """
    ### BEGIN SOLUTION
    y_pred = np.ones(len(X))*y_mean
    for tree in trees:
        y_pred += nu * tree.predict(X)
    ### END SOLUTION
    return y_pred

