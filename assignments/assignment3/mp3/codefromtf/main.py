"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.01
max_iters = 300


def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    # Initialize model.
    N, ndims = A.shape
    model = LogisticModel_TF(ndims - 1, W_init='zeros')

    # Build TensorFlow training graph
    model.build_graph(learn_rate)

    # Train model via gradient descent.
    # Compute classification accuracy based on the return of the "fit" method
    # prob = model.fit(T[:774], A[:774,:], max_iters, learn_rate)
    prob = model.fit(T, A, max_iters, learn_rate)
    # Print Sigmoid output
    print(prob)

    prob = model.fit(T[774:], A[774:, :], max_iters, learn_rate)


if __name__ == '__main__':
    tf.app.run()
