"""Generative Adversarial Networks."""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=5000):
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

    # Loss
    loss_g = []
    loss_d = []

    # Training
    for step in range(num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1., 1.,
                                    [batch_size, model._nlatent]).astype(np.float32)

        # merge = tf.summary.merge_all()

        # Update discriminator by ascending its stochastic gradient
        for k in range(d_iters):

            _, d_loss = model.session.run(
                [model.d_optimizer, model.d_loss],
                feed_dict={model.x_placeholder: batch_x,
                           model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

            loss_d.append(d_loss)

        # Update generator by descending its stochastic gradient
        for j in range(g_iters):

            _, g_loss = model.session.run(
                [model.g_optimizer, model.g_loss],
                feed_dict={model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

            loss_g.append(g_loss)

        if step % 100 == 0:
            print('Iter: {}'.format(step))
            print('D_loss: {:.4}'.format(d_loss))
            print('G_loss: {:.4}'.format(g_loss))

    #     if step % 50 == 0:
    #         out = np.empty((28 * 20, 28 * 20))
    #         for x_idx in range(20):
    #             for y_idx in range(20):
    #                 z_mu = np.random.uniform(-1., 1.,
    #                                          [16, model._nlatent]).astype(np.float32)
    #                 img = model.generate_samples(z_mu)
    #                 out[x_idx * 28:(x_idx + 1) * 28,
    #                     y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    #         plt.imsave('./tmp/gan_' + str(step) + '.png', out, cmap="gray")

    # np.savetxt("loss_g", np.array(loss_g), delimiter=',')
    # np.savetxt("loss_d", np.array(loss_d), delimiter=',')


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan(nlatent=10)

    # Start training
    train(model, mnist_dataset)

    # Plot
    out = np.empty((28 * 20, 28 * 20))
    for x_idx in range(20):
        for y_idx in range(20):
            z_mu = np.random.uniform(-1., 1.,
                                     [16, model._nlatent]).astype(np.float32)
            img = model.generate_samples(z_mu)
            out[x_idx * 28:(x_idx + 1) * 28,
                y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    plt.imsave('gan.png', out, cmap="gray")


if __name__ == "__main__":
    tf.app.run()
