"""Main function for binary classifier"""


import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.0007
max_iters = 300

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    A, T = read_dataset('../data/trainset', 'indexing.txt')

    # Initialize model.
    ndims = A.shape[1]
    model = LogisticModel(ndims - 1, W_init='ones')

    # Train model via gradient descent.
    # model.fit(T[0:411], A[0:411,:], learn_rate, max_iters)
    model.fit(T, A, learn_rate, max_iters)

    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')

    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')

    # Cross validation
    model.fit(T[411:543], A[411:543, :], learn_rate, max_iters)

    # Test model
    model.fit(T[543:], A[543:, :], learn_rate, max_iters)
    # Try all other methods: forward, backward, classify, compute accuracy
