"""Main function for train, eval, and test."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_regression import LinearRegression
from train_eval_model import train_model, eval_model, train_model_analytic
from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data
from utils.plot_tools import plot_x_vs_y


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_float('w_decay_factor', 0.0001, 'Weight decay factor.')
flags.DEFINE_integer('num_steps', 100000, 'Number of update steps to run.')
flags.DEFINE_string('opt_method', 'iter', 'Supports ["iter", "analytic"]')
flags.DEFINE_string(
    'feature_columns',
    'BldgType,OverallQual',
    'Comma separated feature names.')

# 100000  Id,BldgType,OverallQual,GrLivArea,GarageArea,SalePrice
# flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
# flags.DEFINE_float('w_decay_factor', 0.0001, 'Weight decay factor.')
# flags.DEFINE_integer('num_steps', 100000, 'Number of update steps to run.')


def main(_):
    """High level pipeline.
    This script performs the trainsing, evaling and testing state of the model.
    """
    learning_rate = FLAGS.learning_rate
    w_decay_factor = FLAGS.w_decay_factor
    num_steps = FLAGS.num_steps
    opt_method = FLAGS.opt_method
    feature_columns = FLAGS.feature_columns.split(',')

    # Load dataset.
    dataset = read_dataset("data/train.csv")

    # Data processing.
    train_set = preprocess_data(dataset, feature_columns=feature_columns,
                                squared_features=True)

    # Initialize model.
    ndim = train_set[0].shape[1]
    model = LinearRegression(ndim, 'zeros')

    # Train model.
    if opt_method == 'iter':
        # Perform gradient descent.
        train_model(train_set, model, learning_rate, num_steps=num_steps)
        print('Performed gradient descent.')
    else:
        # Compute closed form solution.
        train_model_analytic(train_set, model)
        print('Closed form solution.')

    train_loss = eval_model(train_set, model)
    print("Train loss: %s" % train_loss)

    # Plot the x vs. y if one dimension.
    if train_set[0].shape[1] == 1:
        plot_x_vs_y(train_set, model)

    # Eval model.
    raw_eval = read_dataset("data/val.csv")
    eval_set = preprocess_data(raw_eval, feature_columns=feature_columns,
                               squared_features=True)
    eval_loss = eval_model(eval_set, model)
    print("Eval loss: %s" % eval_loss)

    # Test model.
    raw_test = read_dataset("data/test.csv")
    test_set = preprocess_data(raw_test, feature_columns=feature_columns,
                               squared_features=True)
    test_loss = eval_model(test_set, model)
    print("Test loss: %s" % test_loss)


if __name__ == '__main__':
    tf.app.run()