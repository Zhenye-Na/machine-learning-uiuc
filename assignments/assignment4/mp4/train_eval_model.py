"""Train model and eval model helpers."""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implement the training loop of stochastic gradient descent.

    Perform stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Performs gradient descent. (This function will not be graded.)
    # Seperate the data dict to two parts
    x = data['image']
    y = data['label']

    # Assign dimesion
    M, N = x.shape

    # If the number of example is not divisible by batch_size, the last batch
    # will simply be the remaining examples.
    if M % batch_size != 0:
        last_batch_size = M % batch_size
        num_batches = M // batch_size + 1

        for step in range(num_steps):

            idx_total = np.arange(M)
            np.random.shuffle(idx_total)
            # np.random.permutation(len(M)) is another choice to shuffle the
            # data. Here I reserved the previous method.

            for batch in range(num_batches):
                if (batch == num_batches - 1):
                    if shuffle:
                        idx = idx_total[batch * batch_size:]
                    else:
                        idx = np.arange(batch * batch_size, x.shape[0])
                else:
                    if shuffle:
                        idx = idx_total[batch *
                                        batch_size:(batch + 1) * batch_size]
                    else:
                        idx = np.arange(
                            batch * batch_size, (batch + 1) * batch_size)

                # Assignment batch for x and y
                x_batch = x[idx]
                y_batch = y[idx]

                update_step(x_batch, y_batch, model, learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Perform on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    y_batch = np.reshape(y_batch, (x_batch.shape[0], 1))
    y_hat = model.forward(x_batch)
    grad = np.reshape(model.backward(y_hat, y_batch), (model.ndims + 1, 1))
    model.w = model.w - (learning_rate * grad)


def train_model_qp(data, model):
    """Compute and set the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])

    # Implementation here (do not modify the code above)
    x = data['image']
    # dim = x.shape[1]

    # If we have only one example, it will be vector
    try:
        dim = x.shape[1]
    except BaseException:
        dim = x.shape[0]

    # Set model.w
    model.w = z[0:dim + 1]


def qp_helper(data, model):
    """Prepare arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    # Implementation here.
    x = data['image']
    y = data['label']
    # num = x.shape[0]
    # dim = x.shape[1]
    # ndims = num + dim

    # x = data['image'][0]
    # y = data['label'][0]
    # If we have only one example, it will be vector
    try:
        num = x.shape[0]
        dim = x.shape[1]
    except BaseException:
        dim = x.shape[0]
        num = 1
        x = np.reshape(x, (1, x.shape[0]))

    ndims = num + dim

    # P matrix -> Qudratic parameters
    P = np.zeros((ndims + 1, ndims + 1))
    P[0:dim, 0:dim] = np.eye(dim) * model.w_decay_factor

    # q vecor -> linear parameters
    q1 = np.zeros((dim + 1, 1))
    q2 = np.ones((num, 1))
    q = np.vstack((q1, q2))

    # First block of G matrix
    x1 = np.hstack((x, np.ones((num, 1))))
    x2 = np.multiply(-y, x1)
    G1 = np.hstack((x2, -np.eye(num)))

    # Second block of G matrix
    g1 = np.zeros((num, dim + 1))
    g2 = -np.eye(num)
    G2 = np.hstack((g1, g2))

    # G matrix -> Constraints parameters
    G = np.vstack((G1, G2))

    h1 = -np.ones((num, 1))
    h2 = np.zeros((num, 1))
    h = np.vstack((h1, h2))

    return P, q, G, h


def eval_model(data, model):
    """Perform evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    x = data['image']
    y = data['label']

    f = model.forward(x)
    loss = model.total_loss(f, y)
    acc = np.asscalar(np.sum(model.predict(f) == y) / y.shape[0])

    return loss, acc
