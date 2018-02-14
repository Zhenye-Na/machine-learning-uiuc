'''Simple unit tests for students.'''

import unittest
import numpy as np

from model.self_multiclass import MulticlassSVM

class TestSklearn(unittest.TestCase):

    def setUp(self):
        self.X_train = TestSklearn.mnist[:len(TestSklearn.mnist)//2, 1:]
        self.y_train = (TestSklearn.mnist[:len(TestSklearn.mnist)//2, 0]
                        .astype(np.int))

        self.X_test = TestSklearn.mnist[len(TestSklearn.mnist)//2:, 1:]
        self.y_test = (TestSklearn.mnist[len(TestSklearn.mnist)//2:, 0]
                       .astype(np.int))

    @classmethod
    def setUpClass(cls):
        super(TestSklearn, cls).setUpClass()
        print('Loading data...')
        cls.mnist = np.loadtxt('data/mnist_test.csv', delimiter=',')

    def test_score_shape(self):
        msvm = MulticlassSVM('ovr')
        msvm.fit(self.X_train, self.y_train)
        scores = msvm.scores_ovr_student(self.X_test)
        self.assertTrue(scores.shape[0] == 5000 and scores.shape[1] == 10)

if __name__ == '__main__':
    unittest.main()
