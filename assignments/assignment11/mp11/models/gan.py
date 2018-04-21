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
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
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
        self.d_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='d_optimizer').minimize(self.d_loss)

        self.g_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate_placeholder,
            name='g_optimizer').minimize(self.g_loss)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead
                of creating new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake
                image(None, 1).

        DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:

            # if reuse:
            #     scope.reuse_variables()

            # Input Layer
            inputs = layers.fully_connected(
                inputs=x, num_outputs=512, activation_fn=tf.nn.relu)

            # Drop Out
            drop_out1 = tf.layers.dropout(
                inputs=inputs, rate=0.05)

            # Hidden Layer 1
            hidden1 = layers.fully_connected(
                inputs=drop_out1, num_outputs=256, activation_fn=tf.nn.relu)

            # Drop Out
            drop_out2 = tf.layers.dropout(
                inputs=hidden1, rate=0.05)

            # Hidden Layer 2
            hidden2 = layers.fully_connected(
                inputs=drop_out2, num_outputs=64, activation_fn=tf.nn.relu)

            # Output Layer
            y = layers.fully_connected(
                inputs=hidden2, num_outputs=1, activation_fn=None)

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
        sigmoid_y = tf.sigmoid(y)
        sigmoid_y_hat = tf.sigmoid(y_hat)
        # l = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
        #                                             logits=y_hat,
        #                                             name="d_loss"))

        l = -tf.reduce_mean(tf.log(sigmoid_y) + tf.log(1 - sigmoid_y_hat))
        # l = - (tf.log(y) + tf.log(1 - y_hat))

        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead
                of creating new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:

            # if reuse:
            #     scope.reuse_variables()

            # Input Layer
            inputs = layers.fully_connected(
                inputs=z, num_outputs=64, activation_fn=tf.nn.relu)

            # Hidden Layer 1
            hidden1 = layers.fully_connected(
                inputs=inputs, num_outputs=256, activation_fn=tf.nn.relu)

            # Hidden Layer 2
            hidden2 = layers.fully_connected(
                inputs=hidden1, num_outputs=512, activation_fn=tf.nn.relu)

            # Output Layer
            x_hat = layers.fully_connected(
                inputs=hidden2, num_outputs=self._ndims, activation_fn=tf.nn.softplus)

            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake
                images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = -tf.reduce_mean(y_hat, name="g_loss")
        # l = -tf.log(y_hat)
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
