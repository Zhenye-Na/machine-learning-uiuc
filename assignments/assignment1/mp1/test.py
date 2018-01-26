"""Simple unit tests for students."""

import unittest
import tensorflow as tf
from run_computation import run_computation
from toy_functions import *


class RunComputationTests(unittest.TestCase):
    def test_run_simple(self):
        val = tf.constant(True)
        result = run_computation(val)
        self.assertEqual(result, True)


class RunToyFnTests(unittest.TestCase):
    def test_fn_1(self):
        arg1 = tf.constant([1, 4])
        arg2 = tf.constant([2, -1])
        correct = tf.constant([7, 28])
        attempt = toy_fn_1(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)

    def test_fn_2(self):
        arg1 = tf.constant([[1, 2], [3, 4]])
        arg2 = tf.constant([4, 2])
        correct = tf.constant([-1, 3])
        attempt = toy_fn_2(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)

    def test_fn_3(self):
        arg1 = tf.constant([1, 2])
        arg2 = tf.constant([10, 20])
        correct = tf.constant([1, 10, 2, 20])
        attempt = toy_fn_3(arg1, arg2)
        result = run_computation(tf.reduce_all(tf.equal(correct, attempt)))
        self.assertEqual(result, True)


if __name__ == '__main__':
    unittest.main()
