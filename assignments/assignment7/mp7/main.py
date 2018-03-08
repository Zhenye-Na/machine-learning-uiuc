"""Main function for train, eval, and test."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from linear_mrf import LinearMRF
from data_tools import load_dataset, plot_image


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 10, 'Number of update steps to run.')
flags.DEFINE_float('convergence_margin', 0.01,
                   'Margin of convergence for inference')
flags.DEFINE_string('input_file_path', 'data/circle.png', 'Original Image.')


def main(_):
    """High level pipeline.
    This script performs the trainsing, evaling and testing state of the model.
    """
    learning_rate = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    convergence_margin = FLAGS.convergence_margin
    input_file_path = FLAGS.input_file_path

    # Load dataset.
    original_img, noisy_samples = load_dataset(input_file_path, 10)
    height = original_img.shape[0]
    width = original_img.shape[1]

    original_img = original_img.flatten()
    noisy_samples = [sample.flatten() for sample in noisy_samples]

    # Initialize model.
    model = LinearMRF(width, height)

    model.train(original_img, noisy_samples, learning_rate, num_epochs,
                convergence_margin)

    # Evaluate model on training dataset
    denoised_images = model.test(noisy_samples, convergence_margin)

    # Plot inference result on image
    plot_image(noisy_samples[0].reshape(height, width), 'Noisy Image',
               'data/noisy_sample.png')
    plot_image(denoised_images[0], 'Denoised Version', 'data/denoised_img.png')


if __name__ == '__main__':
    tf.app.run()
