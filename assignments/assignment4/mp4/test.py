"""Simple unit tests for students. (This file will not be graded.)"""

import unittest
import numpy as np
from utils import io_tools
from utils import data_tools
from models import support_vector_machine
from train_eval_model import qp_helper


class IoToolsTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset(
            "data/train.txt", "data/image_data/")

    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_data_shape(self):
        image = self.dataset['image']
        label = self.dataset['label']
        self.assertEqual(image.shape[0], label.shape[0])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = support_vector_machine.SupportVectorMachine(5, 'zeros')

    def test_forward_shape(self):
        x = np.zeros((10, 5))
        y_hat = self.model.forward(x)
        self.assertEqual(y_hat.shape, (10, 1))

    def test_forward_zero(self):
        x = np.zeros((10, 5))
        y = np.zeros((10, 1))
        y_hat = self.model.forward(x)
        np.testing.assert_array_equal(y, y_hat)


class QpTests(unittest.TestCase):
    def setUp(self):
        self.dataset = io_tools.read_dataset(
            "data/train.txt", "data/image_data/")
        self.dataset = data_tools.preprocess_data(self.dataset, 'raw')
        self.model = support_vector_machine.SupportVectorMachine(
            8 * 8 * 3, 'zeros')

    def test_qp(self):
        P, q, G, h = qp_helper(self.dataset, self.model)
        self.assertEqual(P.shape[0], q.shape[0])
        self.assertEqual(G.shape[0], h.shape[0])


if __name__ == '__main__':
    unittest.main()
