"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term, 
            Weight = [Bias, W1, W2, W3, ...] 
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
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            pass
        elif W_init == 'ones':
            pass
        elif W_init == 'uniform':
            pass
        elif W_init == 'gaussian':
            pass
        else:
            print ('Unknown W_init ', W_init) 
        

    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        pass

    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        pass
    
    
    