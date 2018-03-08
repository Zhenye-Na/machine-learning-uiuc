"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implement a linear regression mode model."""

    def backward(self, f, y):
        """Perform the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        # Implementation here.
        reg_grad = self.w_decay_factor * self.w
        indicator = np.multiply(y, f)

        loss_grad = np.zeros((self.ndims + 1, 1))
        for i, value in enumerate(indicator):
            if value < 1:
                loss_grad[:, 0] += self.x[i].T * -y[i]

        total_grad = reg_grad + loss_grad

        return total_grad.flatten()

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.

        Total loss is hinge_loss + w_decay_factor / 2 * ||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        # Implementation here.
        hinge_loss = np.sum(np.maximum(0, 1 - np.multiply(y, f)))
        l2_loss = self.w_decay_factor / 2 * self.w.T.dot(self.w)

        total_loss = hinge_loss + l2_loss
        total_loss = np.asscalar(total_loss)

        return total_loss

    def predict(self, f):
        """Convert score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        # Implementation here.
        y_predict = np.zeros(f.shape)
        y_predict[f >= 0] = 1
        y_predict[f < 0] = -1

        return y_predict
