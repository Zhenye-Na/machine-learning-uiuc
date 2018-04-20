"""Simple unit tests for students."""

import unittest
import numpy as np
from models import gan
import tensorflow as tf

class ModelTests(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.model = gan.Gan()

    def test_ouput_shape(self):
        test_z = np.random.uniform(-1,1,[10,2])
        np.testing.assert_array_equal(self.model.session.run(tf.shape(self.model.x_hat),feed_dict={self.model.z_placeholder: test_z}), (10, 784))


    def test_generator_loss_shape(self):
        tf.assert_scalar(self.model.g_loss)

if __name__ == '__main__':
    unittest.main()
