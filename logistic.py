import numpy as np

def logistic_regression(x_train , y_train , x_test , iters = 1000) :
    n , dim = x_train.shape
    alpha = 1e-3 # learning rate in gradient descent 
    w = np.zeros(dim) # weights 
    b = 0 # bias

    for _ in range(iters) :
        theta = np.dot(x_train , w) + b  
        y_tp = 1 / (1 + np.exp(-theta)) 
        error = y_tp - y_train  

        w = w - alpha / n * np.dot(x_train.T , error)
        b = b - alpha / n * np.sum(error)

    calc = np.dot(x_test , w) + b 
    y_prob = 1 / (1 + np.exp(-calc))
    y_pred = [1 if p>= 0.5 else 0 for p in y_prob]

    return y_pred 