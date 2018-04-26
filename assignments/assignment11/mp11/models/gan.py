"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class Gan(object):
    """Adversary based generator network."""

    def __init__(self, ndims=784, nlatent=2):
        """Initialize a GAN.

        Args:
            ndims (int): Number of dimensions in the feature.
            nlatent (int): Number of dimensions in the latent space.
        """
        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Learning rate
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Add optimizers for appropriate variables
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        self.d_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='d_optimizer').minimize(self.d_loss, var_list=d_vars)

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        self.g_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='g_optimizer').minimize(self.g_loss, var_list=g_vars)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): reuse variables with same name in scope instead
                of creating new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake
                image(None, 1).

        DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:

            # ---------------------------------------------------------------#
            # # Input Layer
            # inputs = layers.fully_connected(
            #     inputs=x, num_outputs=512, activation_fn=tf.nn.relu)

            # # Drop Out
            # drop_out = tf.layers.dropout(
            #     inputs=inputs, rate=0.05)

            # # Hidden Layer 1
            # hidden1 = layers.fully_connected(
            #     inputs=drop_out, num_outputs=256, activation_fn=tf.nn.relu)

            # # Drop Out
            # drop_out = tf.layers.dropout(
            #     inputs=hidden1, rate=0.05)

            # # Hidden Layer 2
            # hidden2 = layers.fully_connected(
            #     inputs=drop_out, num_outputs=128, activation_fn=tf.nn.relu)

            # # Output Layer
            # y = layers.fully_connected(
            #     inputs=hidden2, num_outputs=1, activation_fn=None)

            # if reuse:
            #     scope.reuse_variables()
            # tf.truncated_normal_initializer()

            # ---------------------------------------------------------------#

            keep_prob = 0.9
            num_h1 = 392
            num_h2 = 196

            # Fully Connected Layer 1, dropout
            w1 = tf.get_variable(name="d_w1",
                                 shape=[self._ndims, num_h1],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b1 = tf.get_variable(name="d_b1",
                                 shape=[num_h1],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1), keep_prob)

            # Fully Connected Layer 2, dropout
            w2 = tf.get_variable(name="d_w2",
                                 shape=[num_h1, num_h2],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b2 = tf.get_variable(name="d_b2",
                                 shape=[num_h2],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

            # Fully Connected Layer 3
            w3 = tf.get_variable(name="d_w3",
                                 shape=[num_h2, 1],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b3 = tf.get_variable(name="d_b3",
                                 shape=[1],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            y = tf.matmul(h2, w3) + b3  # logits

            # ---------------------------------------------------------------#

            # n_units = 392
            # alpha = 0.01

            # # Hidden layer
            # h1 = tf.layers.dense(x, n_units, activation=None)
            # # Leaky ReLU
            # h1 = tf.maximum(h1, alpha * h1)

            # # logits
            # y = tf.layers.dense(h1, 1, activation=None)

        return y

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true
                images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake
                images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # Label smoothing
        # smooth = 0.1

        # create labels for discriminator
        d_labels_real = tf.ones_like(y)
        d_labels_fake = tf.zeros_like(y_hat)

        # compute loss for real/fake images
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y, labels=d_labels_real)
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_hat, labels=d_labels_fake)

        l = tf.reduce_mean(d_loss_fake + d_loss_real)

        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, nlatent).
            reuse (Boolean): reuse variables with same name in scope instead
                of creating new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:

            # ---------------------------------------------------------------#
            # # Input Layer
            # inputs = layers.fully_connected(
            #     inputs=z, num_outputs=128, activation_fn=tf.nn.relu)

            # # Hidden Layer 1
            # hidden1 = layers.fully_connected(
            #     inputs=inputs, num_outputs=256, activation_fn=tf.nn.relu)

            # # Hidden Layer 2
            # hidden2 = layers.fully_connected(
            #     inputs=hidden1, num_outputs=512, activation_fn=tf.nn.relu)

            # # Output Layer
            # x_hat = layers.fully_connected(
            #     inputs=hidden2, num_outputs=self._ndims, activation_fn=tf.nn.softplus)

            # if reuse:
            #     scope.reuse_variables()

            # tf.truncated_normal_initializer()

            # ---------------------------------------------------------------#

            h1_size = 196
            h2_size = 392

            # Fully Connected Layer 1
            w1 = tf.get_variable(name="g_w1",
                                 shape=[self._nlatent, h1_size],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b1 = tf.get_variable(name="g_b1",
                                 shape=[h1_size],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            h1 = tf.nn.relu(tf.matmul(z, w1) + b1)

            # Fully Connected Layer 2
            w2 = tf.get_variable(name="g_w2",
                                 shape=[h1_size, h2_size],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b2 = tf.get_variable(name="g_b2",
                                 shape=[h2_size],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

            # Fully Connected Layer 3
            w3 = tf.get_variable(name="g_w3",
                                 shape=[h2_size, self._ndims],
                                 dtype=tf.float32,
                                 initializer=layers.xavier_initializer(uniform=False))

            b3 = tf.get_variable(name="g_b3",
                                 shape=[self._ndims],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

            x_hat = tf.matmul(h2, w3) + b3

            # ---------------------------------------------------------------#

            # alpha = 0.01
            # n_units = 392

            # # Hidden layer
            # h1 = tf.layers.dense(z, n_units, activation=None)

            # # Leaky ReLU
            # h1 = tf.maximum(h1, alpha * h1)

            # # Logits
            # x_hat = tf.layers.dense(h1, self._ndims, activation=None)

            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake
                images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the generator.

        """
        # create labels for generator
        labels = tf.ones_like(y_hat)

        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_hat, labels=labels))

        return l

    def generate_samples(self, z_np):
        """Generate random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        out = self.x_hat.eval(session=self.session, feed_dict={
                              self.z_placeholder: z_np})
        return out
