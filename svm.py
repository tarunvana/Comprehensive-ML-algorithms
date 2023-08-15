import numpy as np

def svm(x_train, y_train, x_test , n_iters = 100):
    dim = x_train.shape[1]
    w = np.zeros(dim)
    b = 0
    alpha = 1e-3 # learning rate in gradient descent 
    beta = 1e-2  # step size 
    cls_map = np.where(y_train <= 0, -1, 1) # Changing values of 0 to -1 .

    for _ in range(n_iters):
        for idx , x in enumerate(x_train):
            line = np.dot(x, w) + b

            if cls_map[idx] * line >= 1: 
                # The points are correctly classified . So update w , b using gradient descent
                dw = beta * w
                db = 0
            else:
                # The points are wrongly classified . Update using gradient descent and gradient of hinge loss
                dw = beta * w - cls_map[idx] * x
                db = -cls_map[idx]

            w -= alpha * dw
            b -= alpha * db

    estimate = np.dot(x_test ,w) + b 
    prediction = np.sign(estimate)

    return np.where(prediction == -1 , 0 , 1) , w , b