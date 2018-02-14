import unittest
import numpy as np
from sklearn import metrics

from model.sklearn_multiclass import sklearn_multiclass_prediction
from model.self_multiclass import MulticlassSVM


class TestSklearn(unittest.TestCase):

    def setUp(self):
        self.X_train = TestSklearn.mnist[:len(TestSklearn.mnist)//5, 1:]
        self.y_train = (TestSklearn.mnist[:len(TestSklearn.mnist)//5, 0]
                        .astype(np.int))

        self.X_test = TestSklearn.mnist[len(TestSklearn.mnist)//5:, 1:]
        self.y_test = (TestSklearn.mnist[len(TestSklearn.mnist)//5:, 0]
                       .astype(np.int))

        self.X = np.array([[0.74062279, 0.17422722],
                           [0.66662075, 0.21494885],
                           [0.94704997, 0.68120293],
                           [0.86448145, 0.96164814],
                           [0.71720711, 0.26782964]])
        self.y = np.array([2, 0, 0, 1, 0])
        self.W = np.array([[0.02640325, 0.21658084],
                           [0.41876674, 0.73586187],
                           [0.03738621, 0.93108151]])

    @classmethod
    def setUpClass(cls):
        super(TestSklearn, cls).setUpClass()
        cls.mnist = np.loadtxt('data/mnist_test.csv', delimiter=',')

    def test_sklearn_ovr_accuracy(self):
        y_pred_train, y_pred_test = sklearn_multiclass_prediction(
            'ovr', self.X_train, self.y_train, self.X_test)

        train_acc = metrics.accuracy_score(self.y_train, y_pred_train)
        test_acc = metrics.accuracy_score(self.y_test, y_pred_test)

        self.assertTrue(abs(train_acc - 1.0) < 1e-6)
        self.assertTrue(abs(test_acc - 0.814875) < 1e-6)

    def test_sklearn_ovo_accuracy(self):
        y_pred_train, y_pred_test = sklearn_multiclass_prediction(
            'ovo', self.X_train, self.y_train, self.X_test)

        train_acc = metrics.accuracy_score(self.y_train, y_pred_train)
        test_acc = metrics.accuracy_score(self.y_test, y_pred_test)

        self.assertTrue(abs(train_acc - 1.0) < 1e-6)
        self.assertTrue(abs(test_acc - 0.892625) < 1e-6)

    def test_sklearn_crammer_accuracy(self):
        y_pred_train, y_pred_test = sklearn_multiclass_prediction(
            'crammer', self.X_train, self.y_train, self.X_test)

        train_acc = metrics.accuracy_score(self.y_train, y_pred_train)
        test_acc = metrics.accuracy_score(self.y_test, y_pred_test)

        self.assertTrue(abs(train_acc - 1.0) < 1e-6)
        self.assertTrue(abs(test_acc - 0.85825) < 1e-6)

    def test_self_ovr_accuracy(self):
        self_ovr = MulticlassSVM('ovr')
        self_ovr.fit(self.X_train, self.y_train)

        train_acc = metrics.accuracy_score(
            self.y_train, self_ovr.predict(self.X_train))
        test_acc = metrics.accuracy_score(
            self.y_test, self_ovr.predict(self.X_test))

        self.assertTrue(abs(train_acc - 1.0) < 1e-6)
        self.assertTrue(abs(test_acc - 0.814875) < 1e-6)

    def test_self_ovo_accuracy(self):
        self_ovr = MulticlassSVM('ovo')
        self_ovr.fit(self.X_train, self.y_train)

        train_acc = metrics.accuracy_score(
            self.y_train, self_ovr.predict(self.X_train))
        test_acc = metrics.accuracy_score(
            self.y_test, self_ovr.predict(self.X_test))

        self.assertTrue(abs(train_acc - 1.0) < 5e-3)
        self.assertTrue(abs(test_acc - 0.892625) < 5e-3)

    def test_grad(self):
        my_cs = MulticlassSVM('crammer-singer')
        norm_diff = np.linalg.norm(my_cs.grad_student(self.W, self.X, self.y) -
        np.array([[-2.30447458, -0.94740057],
                  [ 2.62578591,  1.11242235],
                  [ 0.16124486,  1.71850243]]))
        self.assertTrue(norm_diff < 1e-6)

    def test_loss(self):
        my_cs = MulticlassSVM('crammer-singer')
        loss_diff = abs(my_cs.loss_student(self.W, self.X, self.y)
                        - 7.441854144192854)
        self.assertTrue(loss_diff < 1e-6)

if __name__ == '__main__':
    unittest.main()
