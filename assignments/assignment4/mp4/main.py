"""Main function for train, eval, and test. This file will not be graded."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.support_vector_machine import SupportVectorMachine
from train_eval_model import train_model, eval_model, train_model_qp
from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('w_decay_factor', 0.001, 'Weight decay factor.')
flags.DEFINE_integer('num_steps', 5000, 'Number of update steps to run.')
flags.DEFINE_string(
    'feature_type',
    'default',
    'Feature type, supports [raw, default, custom]')
flags.DEFINE_string('opt_method', 'qp', 'Supports ["iter", "qp"]')


def main(_):
    """High level pipeline.

    This script performs the trainsing, evaling and testing state of the model.
    """
    learning_rate = FLAGS.learning_rate
    w_decay_factor = FLAGS.w_decay_factor
    num_steps = FLAGS.num_steps
    opt_method = FLAGS.opt_method
    feature_type = FLAGS.feature_type

    # Load dataset and data processing.
    train_set = read_dataset("data/train.txt", "data/image_data/")
    train_set = preprocess_data(train_set, feature_type)

    # Initialize model.
    ndim = train_set['image'][0].shape[0]
    model = SupportVectorMachine(
        ndim, 'ones', w_decay_factor=FLAGS.w_decay_factor)

    # Train model.
    if opt_method == 'iter':
        # Perform gradient descent.
        train_model(train_set, model, learning_rate, num_steps=num_steps)
        print('Performed gradient descent.')
    else:
        # Compute closed form solution.
        train_model_qp(train_set, model)
        print('Finished QP Solver.')

    train_loss, train_acc = eval_model(train_set, model)
    print("Train loss: %s" % train_loss)
    print("Train acc: %s" % train_acc)

    # Eval model.
    eval_set = read_dataset("data/val.txt", "data/image_data/")
    eval_set = preprocess_data(eval_set, feature_type)
    eval_loss, eval_acc = eval_model(eval_set, model)
    print("Eval loss: %s" % eval_loss)
    print("Eval acc: %s" % eval_acc)

    # Test model.
    test_set = read_dataset("data/test.txt", "data/image_data/")
    test_set = preprocess_data(test_set, feature_type)
    test_loss, test_acc = eval_model(test_set, model)
    print("Test loss: %s" % test_loss)
    print("Test acc: %s" % test_acc)


if __name__ == '__main__':
    tf.app.run()
