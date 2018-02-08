"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implement a linear regression mode model."""

    def backward(self, f, y):
        """Perform the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
            dimension (ndims+1,1).
        """
        l = f - y

        self.x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1)
        xt = np.transpose(self.x)

        square_grad = np.matmul(xt, l)
        reg_grad = self.w_decay_factor * self.w
        total_grad = square_grad + reg_grad

        return total_grad

    def total_loss(self, f, y):
        """Compute the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """
        # Same as 'Xw - y'
        diff = f - y
        # print(diff.shape)
        # squared_loss
        squared_loss = 1 / 2 * np.matmul(np.transpose(diff), diff)

        # L2 regularization
        reg = self.w_decay_factor / 2 * np.matmul(np.transpose(self.w), self.w)

        # total loss
        total_loss = squared_loss + reg

        # return total_loss[0][0]
        return total_loss

    def predict(self, f):
        """Nothing to do here."""
        return f
