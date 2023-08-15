import numpy as np
from scipy.spatial import KDTree
'''
Instead of using traditional code where we find distance between every point in test set to
every point in train set and sort them based on distance to obtain nearest neighbours.

In this code i implemented KDTree data structure which takes lesser time to find 
the k nearest neighbours than traditional method. 
'''
def k_nn(x_train , y_train , x_test , k = 1) :
    kd_tree = KDTree(x_train)
    _ , indices = kd_tree.query(x_test, k = k)
    if k > 1 :
        labels = np.mean(y_train[indices] , axis = 1)
    else :
        labels = y_train[indices]
    return np.round(labels).astype(int)