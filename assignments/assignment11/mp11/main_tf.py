"""Generative Adversarial Networks."""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0001, batch_size=32,
          num_steps=100):
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
    # According to original GAN paper, k=1
    d_iters = 5

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
            # model.session.run(
            #     model.d_optimizer,
            #     feed_dict={model.x_placeholder: batch_x,
            #                model.z_placeholder: batch_z,
            #                model.learning_rate_placeholder: learning_rate}
            # )

            _, d_loss = model.session.run(
                [model.d_optimizer, model.d_loss],
                feed_dict={model.x_placeholder: batch_x,
                           model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

            print("d_loss: %f" % (d_loss))

        # Update generator by descending its stochastic gradient
        batch_z = np.random.uniform(-1, 1, [batch_size, 2])
        for k in range(g_iters):
            # model.session.run(
            #     model.g_optimizer,
            #     feed_dict={model.z_placeholder: batch_z,
            #                model.learning_rate_placeholder: learning_rate}
            # )

            _, g_loss = model.session.run(
                [model.g_optimizer, model.g_loss],
                feed_dict={model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

            print("g_loss: %f" % (g_loss))

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
    x_z = np.random.uniform(-1, 1, 20)
    y_z = np.random.uniform(-1, 1, 20)

    out = np.empty((28 * 20, 28 * 20))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.array([[y, x]])
            img = model.generate_samples(z_mu)
            # print(img.shape)
            out[x_idx * 28:(x_idx + 1) * 28,
                y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    plt.imsave('gan.png', out, cmap="gray")
    print(img[0].reshape(28, 28))

    batch_x, _ = mnist_dataset.train.next_batch(16)
    first_array = batch_x[0].reshape(28, 28)
    print(first_array)
    plt.imsave('fig.png', first_array, cmap="gray")

if __name__ == "__main__":
    tf.app.run()
