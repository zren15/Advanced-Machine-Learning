import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    with open(filename,'r') as f:
        contents = f.read().strip()
    contents = contents.split('\n')
    data =[]
    for i in range(len(contents)):
        d = [float(n) for n in contents[i].split(',')]
        data.append(d)
    data = np.array(data)
#     print(data)
    X = data[:,:-1]
    Y = data[:,-1]
    Y = np.array([-1 if y==0 else y for y in Y])
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    n_weight = np.ones(N) / N

    ### BEGIN SOLUTION
    for i in range(num_iter):
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X,y,n_weight)
        y_pred = clf.predict(X) 
        trees.append(clf)
        error = np.sum(n_weight[y_pred!=y])/np.sum(n_weight)
        weight = np.log((1-error)/error)
        trees_weights.append(weight)
        n_weight[y_pred!=y] = n_weight[y_pred!=y]*np.exp(weight) 
   
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    for i in range(len(trees)):
        y += trees_weights[i]*trees[i].predict(X)
    y = [np.sign(y[i]) for i in range(len(y))] 
    ### END SOLUTION
    return y
