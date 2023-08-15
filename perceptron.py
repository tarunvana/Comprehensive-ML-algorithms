import numpy as np

def error(x,y,w) :
    '''
    error function to calculate number of wrongly classified samples 
    '''
    y_pred = [1 if np.dot(x[i] , w) >= 0 else 0 for i in range(x.shape[0])]  
    count = 0 
    for i in range(len(y)) :
        count += y[i] != y_pred[i]
    return count

def perceptron(x , y , x_test , iters = 20 , bool = True) :
    '''
    Run simple perceptron algorithm and store value of w and number of errored samples after 
    each whole iteration.
    Select the w which gives least train error after 20 iterations as the weight
    '''
    w = np.zeros(x.shape[1])
    err_w_dic = {}

    for j in range(iters):
        for i in range(x.shape[0]) :
            y_ = 1 if np.dot(x[i] , w) >= 0 else 0 
            if y[i] != y_ :
                w = w + x[i] if y[i] == 1 else w - x[i]
        err_w_dic[j] = [error(x,y,w) , w]

    least_error = min(err_w_dic.items() , key = lambda x : x[1][0])
    w_best = least_error[1][1]

    y_pred = [1 if np.dot(x_test[i] , w_best) >= 0 else 0 for i in range(x_test.shape[0])]

    if bool :
        return y_pred 
    else :
        return y_pred , w_best