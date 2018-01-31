"""Train model and eval model helpers."""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implement the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from data_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Seperate the processed_dataset to two parts
    x = processed_dataset[0]
    y = processed_dataset[1]

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

            for batch in range(num_batches):
                if (batch == num_batches - 1):
                    if shuffle:
                        idx = idx_total[batch * batch_size:]
                    else:
                        idx = np.arange(batch * batch_size, x.shape[0])
                else:
                    if shuffle:
                        idx = idx_total[batch * batch_size:(batch + 1) * batch_size]
                    else:
                        idx = np.arange(batch * batch_size, (batch + 1) * batch_size)

                # Assignment batch for x and y
                x_batch = x[idx]
                y_batch = y[idx]

                model.w = update_step(x_batch, y_batch, model, learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Perform on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    y_batch = np.reshape(y_batch, (x_batch.shape[0], 1))
    y_hat = model.forward(x_batch)
    temp_grad = model.backward(y_hat, y_batch)
    model.w = model.w - (learning_rate * temp_grad)

    return model.w


def train_model_analytic(processed_dataset, model):
    """Compute and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    x = processed_dataset[0]
    y = processed_dataset[1]

    # weight is a (N,1) column vector
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    model.w = np.linalg.inv(x.T.dot(x) + model.w_decay_factor * np.ones((x.T.dot(x).shape))).dot(x.T).dot(y)
    return model.w


def eval_model(processed_dataset, model):
    """Perform evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from data_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Seperate the processed_dataset to two parts
    x = processed_dataset[0]
    y = processed_dataset[1]

    f = model.forward(x)

    # You can disregard accuracy.
    loss = model.total_loss(f, y)
    return loss
