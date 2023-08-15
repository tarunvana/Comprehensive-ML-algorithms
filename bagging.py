import numpy as np
from perceptron import perceptron

def bagging(x_train , y_train , x_test , iters = 25) :

    n , dim = x_train.shape
    y_pred = np.zeros((x_test.shape[0] , iters))

    for i in range(iters):
        # Randomly initialise data from dataset 
        idx = np.random.choice(n , n , replace = True)
        x_temp = x_train[idx]
        y_temp = y_train[idx]
        # Obtaining predictions from a weak learner 
        y_pred[:,i] = perceptron(x_temp , y_temp , x_test , iters = 1)

    # Predicting the one which gives majority
    return np.round(np.mean(y_pred, axis=1)).astype(int)