"""logistic model class for binary classification."""

import numpy as np


class LogisticModel(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W = np.zeros([ndims + 1, ])
        elif W_init == 'ones':
            self.W = np.ones([ndims + 1, ])
        elif W_init == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(ndims + 1, ))
        elif W_init == 'gaussian':
            self.W = np.random.normal(loc=0, scale=0.1, size=(ndims + 1, ))
        else:
            print('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        z = X.dot(self.W)
        p = 1 / (1 + np.exp(-z))

        return p

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        self.W = np.reshape(self.W, (self.W.shape[0], 1))
        z = X.dot(self.W)
        z = np.reshape(z, (z.shape[0], 1))

        Y_true = np.reshape(Y_true, (Y_true.shape[0], 1))
        den = 1 + np.exp(-1 * np.multiply(Y_true, z))
        num = np.exp(-1 * np.multiply(Y_true, z))
        grad = np.multiply(np.divide(num, den), -Y_true).T.dot(X)

        return grad.T

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        p = self.forward(X)
        p[p >= 0.5] = 1
        p[p < 0.5] = -1

        return p.flatten()

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        for iter in range(max_iters):

            grad = self.backward(Y_true, X)
            self.W = self.W - learn_rate * grad

            if iter == max_iters - 1:
                l = self.classify(X)
                s = np.sum(Y_true == l)
                acc = s / len(Y_true)
                print('Iter:', iter + 1, 'Accuracy: ', acc)
