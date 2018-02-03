"""Simple unit tests on codefromscratch."""

import unittest
import numpy as np
from codefromscratch import io_tools
from codefromscratch import logistic_model

class ScratchTests(unittest.TestCase):
    def setUp(self):
        # load dataset
        self.A = None
        self.T = None
        self.N = None
        try:
            self.A, self.T = io_tools.read_dataset(path_to_dataset_folder='data/trainset',index_filename='indexing.txt')
            self.N = len(self.T)
        except:
            pass
        # Initialize model.
        self.model = None
        self.model_W = None
        try:
            self.model = logistic_model.LogisticModel(ndims = 16, W_init = 'zeros')
            self.model_W = self.model.W
        except:
            pass
        
    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.A)
        self.assertIsNotNone(self.T)
    
    def test_init_model_not_none(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model_W)


if __name__ == '__main__':
    unittest.main()
