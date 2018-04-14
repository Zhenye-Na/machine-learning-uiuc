"""Simple unit tests."""

import unittest
import numpy as np
import tensorflow as tf
from vae import VariationalAutoencoder


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = VariationalAutoencoder(ndims=20, nlatent=2)

    def test_ouput_shape(self):
        output_dim = tf.shape(self.model.outputs_tensor)[1]
        fd = {self.model.x_placeholder: np.zeros([1, 20])}
        dim = self.model.session.run(output_dim,
                                     feed_dict=fd)
        self.assertEqual(dim, 20)

    def test_loss_shape(self):
        tf.assert_scalar(self.model.loss_tensor)

if __name__ == '__main__':
    unittest.main()
