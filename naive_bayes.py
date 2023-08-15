import numpy as np

def naive_bayes(x , y , x_test) :
    '''
    Naive - Bayes Classifier implemented step by step
    '''
    n , dim = x.shape

    p_hat = np.mean(y) # Fraction of spam mails

    p_word_labl0 = np.zeros(dim) # Array to store p(word in mail) if label == 0 
    p_word_labl1 = np.zeros(dim) # Array to store p(word in mail) if label == 1
    spam , ham = 0 , 0

    for i in range(x.shape[0]) :
        if y[i] == 0 : 
            p_word_labl0 += np.array([bool(j) for j in x[i]])   
            ham += 1 
        else :
            p_word_labl1 += np.array([bool(j) for j in x[i]])
            spam += 1  

    # Obtaining Probabilities
    p_word_labl0 = p_word_labl0 / ham      
    p_word_labl1 = p_word_labl1 / spam

    # Calculating p(ytest = 1| xtest) and p(ytest = 0 | xtest) and assigning labels 
    y_t = []
    for i in range(len(x_test)) :
        sample = x_test[i]
        p_y_1 = p_hat 
        p_y_0 = 1 - p_hat

        for j in range(len(sample)) :
            if sample[j] >= 1 :
                p_y_1 = p_y_1 * p_word_labl1[j]
                p_y_0 = p_y_0 * p_word_labl0[j]
            else :
                p_y_1 = p_y_1 * (1 - p_word_labl1[j])
                p_y_0 = p_y_0 * (1 - p_word_labl0[j])

        y_t.append(1) if p_y_1 >= p_y_0 else y_t.append(0)
    
    return y_t