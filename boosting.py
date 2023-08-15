import numpy as np
from perceptron import perceptron

def boosting(x , y , x_test):
    d = np.ones(x.shape[0]) / x.shape[0]
    perceptron_weights = []
    alphas = []

    for i in range(25):
        # Adding probability as a feature 
        X_weighted = x * d[:, np.newaxis]

        # obtain train set predictions and model parameters
        y_pred , p_w = perceptron(X_weighted , y , X_weighted , iters = 1 , bool= False)

        # Finding alpha 
        error = np.sum(d * (y_pred != y))
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)

        # UPdating probabilities
        d *= np.exp(-alpha * y * y_pred)
        d /= np.sum(d)

        # Storing perceptron weights
        perceptron_weights.append(p_w)

    predictions = np.zeros(x_test.shape[0])
    # Calculating final predictions as sigma(alpha . h)
    for i, estimator in enumerate(perceptron_weights):
        y_pred = np.array([1 if np.dot(x_test[i] , estimator) >= 0 else 0 for i in range(x_test.shape[0])])
        predictions += alphas[i] * y_pred

    return np.sign(predictions).astype(int)