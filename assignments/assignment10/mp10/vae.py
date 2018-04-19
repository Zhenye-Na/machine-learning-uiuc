"""Variation Autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers
from tensorflow.contrib.slim import fully_connected


class VariationalAutoencoder(object):
    """Varational Autoencoder."""

    def __init__(self, ndims=784, nlatent=2):
        """Initialize a VAE.

        (**Do not change this function**)
        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """
        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def _sample_z(self, z_mean, z_log_var):
        """Sample z using reparametrization trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)
        Returns:
            z (tf.Tensor): Random sampled z of dimension (None, _nlatent)
        """
        # z = tf.random_normal(tf.shape(z_mean), mean=z_mean, stddev=tf.sqrt(
        #     tf.exp(z_log_var)), dtype=tf.float32)
        epsilon = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        z = z_mean + tf.sqrt(tf.exp(z_log_var)) * epsilon
        return z

    def _encoder(self, x):
        """Encoder block of the network.

        Builds a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes, and outputs two branches each with _nlatent nodes
        representing z_mean and z_log_var. Network illustrated below:

                             |-> _nlatent (z_mean)
        Input --> 100 --> 50 -
                             |-> _nlatent (z_log_var)

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, _ndims).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension
                (None, _nlatent).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, _nlatent).
        """
        # Number of outputs for encoder
        num_outputs = self._nlatent * 2
        # Input Layer
        input_ = fully_connected(
            inputs=x, num_outputs=100, activation_fn=tf.nn.softplus)

        # Hidden Layer
        hidden = fully_connected(
            inputs=input_, num_outputs=50, activation_fn=tf.nn.softplus)

        # Output Layer
        output = fully_connected(
            inputs=hidden, num_outputs=num_outputs, activation_fn=None)

        # Extract mean and log-variance from Output
        z_mean = output[:, 0:self._nlatent]
        z_log_var = output[:, self._nlatent:]
        return z_mean, z_log_var

    def _decoder(self, z):
        """Decode back into image, from a sampled z.

        Builds a three layer network of fully connected layers,
        with 50, 100, _ndims nodes.

        z (_nlatent) --> 50 --> 100 --> _ndims.

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, _nlatent).
        Returns:
            f(tf.Tensor): Decoded features, tensor of dimension (None, _ndims).
        """
        # Number of outputs for decoder
        num_outputs = self._ndims

        # Dense Layer
        dense = fully_connected(inputs=z, num_outputs=50,
                                activation_fn=tf.nn.softplus)

        # Hidden Layer
        hidden = fully_connected(
            inputs=dense, num_outputs=100, activation_fn=tf.nn.softplus)

        # Output Layer
        output = fully_connected(
            inputs=hidden, num_outputs=num_outputs, activation_fn=tf.nn.sigmoid)
        return output

    def _latent_loss(self, z_mean, z_log_var):
        """Construct the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, _nlatent)
            z_log_var(tf.Tensor): Tensor of dimension (None, _nlatent)

        Returns:
            latent_loss(tf.Tensor): A scalar Tensor of dimension ()
                containing the latent loss.
        """
        latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
            tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var, 1),
            name="latent_loss")
        return latent_loss

    def _reconstruction_loss(self, f, x_gt):
        """Construct the reconstruction loss, assuming Gaussian distribution.

        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                _ndims).
            x_gt(tf.Tensor): Ground truth for each example, dimension (None,
                _ndims).
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """
        # recon_loss = tf.losses.mean_squared_error(labels=x_gt, predictions=f)
        recon_loss = tf.nn.l2_loss(f - x_gt, name="recon_loss")
        return recon_loss

    def loss(self, f, x_gt, z_mean, z_log_var):
        """Compute the total loss.

        Computes the sum of latent and reconstruction loss.

        Args:
            f (tf.Tensor): Decoded image for each example, dimension (None,
                _ndims).
            x_gt (tf.Tensor): Ground truth for each example, dimension (None,
                _ndims)
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)

        Returns:
            total_loss: Tensor for dimension (). Sum of
                latent_loss and reconstruction loss.
        """
        total_loss = self._latent_loss(
            z_mean, z_log_var) + self._reconstruction_loss(f, x_gt)
        return total_loss

    def update_op(self, loss, learning_rate):
        """Create the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate, name='Adam').minimize(loss)
        return train_op

    def generate_samples(self, z_np):
        """Generate random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        out = None
        out = self.outputs_tensor.eval(
            session=self.session, feed_dict={self.z: z_np})
        return out
