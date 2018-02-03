"""Simple unit tests on codefromtf"""

import unittest
import numpy as np
from codefromtf import io_tools
from codefromtf import logistic_model

class TFTests(unittest.TestCase):
    def setUp(self):
        # load dataset
        self.A = None
        self.T = None
        self.N = None
        try:
            self.A, self.T = io_tools.read_dataset_tf(path_to_dataset_folder='data/trainset',index_filename='indexing.txt')
            self.N = len(self.T)
        except:
            pass
        # Initialize model.
        self.model = None
        self.model_W0 = None
        try:
            self.model = logistic_model.LogisticModel_TF(ndims = 16, W_init = 'zeros')
            self.model_W0 = self.model.W0
        except:
            pass
        
    def test_read_dataset_not_none(self):
        self.assertIsNotNone(self.A)
        self.assertIsNotNone(self.T)
    
    def test_init_model_not_none(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model_W0)

if __name__ == '__main__':
    unittest.main()
