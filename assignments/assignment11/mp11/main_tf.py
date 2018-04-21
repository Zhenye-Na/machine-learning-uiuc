"""Generative Adversarial Networks."""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0007, batch_size=16,
          num_steps=500):
    """Implement the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model (GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate (float): Learning rate. 0.0005
        batch_size (int): batch size used for training.
        num_steps (int): Number of steps to run the update ops. 5000
    """
    # Iterations for discriminator
    # According to original GAN paper, they used k=1
    d_iters = 1

    # Iterations for generator
    g_iters = 1

    print('batch size: %d, epoch num: %d, learning rate: %f' %
          (batch_size, num_steps, learning_rate))
    print('Start training...')

    # Training
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, 2])

        # Update discriminator by ascending its stochastic gradient
        for k in range(d_iters):
            model.session.run(
                model.d_optimizer,
                feed_dict={model.x_placeholder: batch_x,
                           model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

        # Update generator by descending its stochastic gradient
        for k in range(g_iters):
            model.session.run(
                model.g_optimizer,
                feed_dict={model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

        if step % 100 == 0:
            print("Training step: %d out of total steps: %d" %
                  (step, num_steps))


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

    # Plot
    out = np.empty((28 * 8, 28 * 8))
    for x_idx in range(8):
        for y_idx in range(8):
            z_mu = np.random.uniform(-1, 1, [16, 2])
            img = model.generate_samples(z_mu)
            out[x_idx * 28:(x_idx + 1) * 28,
                y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    plt.imsave('gan.png', out, cmap="gray")


if __name__ == "__main__":
    tf.app.run()
