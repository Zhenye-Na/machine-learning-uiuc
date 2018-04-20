"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1,1,[batch_size,2])
        # Train generator and discriminator


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan()

    # Start training
    train(model, mnist_dataset)


if __name__ == "__main__":
    tf.app.run()
