"""Simple unit tests on model for grading."""

import unittest
import numpy as np
from models.gaussian_mixture_model import GaussianMixtureModel
from utils import io_tools

class ModelTests(unittest.TestCase):
    def setUp(self):
        n_dims = 2
        unlabeled_data = np.random.normal(-1,0.1,[100,n_dims])
        unlabeled_data = np.concatenate((unlabeled_data,np.random.normal(1,0.1,[50,n_dims])))

        self.n_dims = 2
        self.n_components = 2
        self.max_iter = 1
        self.model = GaussianMixtureModel(self.n_dims, n_components=self.n_components,
                                     max_iter=self.max_iter)

    def test_io(self):
        train_label, train_data = io_tools.read_dataset('data/simple_test.csv')      
        np.testing.assert_array_equal(train_data.shape,np.asarray([200,2]))


    def test_model(self):
        self.assertEquals(self.model._mu.shape[0], self.n_components)

if __name__ == '__main__':
    unittest.main()
