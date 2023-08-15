import numpy as np

class Node:
    '''
    Node class to store the info regarding the node
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # The feature on who basis divison occurs
        self.threshold = threshold # The threshold value of the feature 
        self.left = left  # The left divison data <= threshold 
        self.right = right # The right division data > threshold
        self.value = value 

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    """
    Decision Tree Class to implement Tree like structure
    """
    def __init__(self, max_depth = 5 , min_samples_split = 10):
        self.max_depth = max_depth # maximum depth of tree
        self.min_samples_split = min_samples_split # minimum samples in each node 
        self.root = None # To traverse 

    def is_finished(self, depth):
        '''
        Checking whether to stop the node division or not.
        '''
        if depth >= self.max_depth or self.n_class_labels == 1 or self.n_samples < self.min_samples_split :
            return True
        return False

    def entropy(self, y):
        '''
        Entropy og the given labels
        '''
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def splitter(self, X, thresh):
        '''
        The function which divides data based on threshold value 
        '''
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def info_gain(self, X, y, thresh):
        '''
        The function to calculate the information gain
        '''
        left_idx, right_idx = self.splitter(X, thresh)
        g = len(left_idx) / len(y)

        if g in [0,1] : 
            return 0

        return self.entropy(y) - g * self.entropy(y[left_idx]) - (1 - g) * self.entropy(y[right_idx])

    def best_split(self, X, y, features):
        '''
        Checking the information gain and deciding the best feature and its threshold
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self.info_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def build_tree(self, X, y, depth=0):
        '''
        Start the tree with a single node and building it.
        '''
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self.is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self.best_split(X, y, rnd_feats)

        left_idx, right_idx = self.splitter(X[:, best_feat], best_thresh)

        left_child = self.build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)
    
    def traverse(self, x, node):
        '''
        Traversing through the tree to obtain the leaf finally to get prediction value
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)

    def fit(self, X, y):
        '''
        Function to train model and build tree
        '''
        self.root = self.build_tree(X, y)

    def predict(self, X):
        '''
        Predict the labels of data by traversing through the tree
        '''
        predictions = [self.traverse(x, self.root) for x in X]
        return np.array(predictions)