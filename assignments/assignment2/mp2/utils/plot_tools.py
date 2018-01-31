"""Helper functions for plotting."""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def plot_x_vs_y(dataset, model):
    """Plot x vs y. Do not have to change this function.

    Args:
        dataset(list): processed dataset, from data_tools.
        model(LinearRegression): Linear regression model.
    """
    plt.scatter(dataset[0], dataset[1])
    # x_tick = np.arange(np.min(dataset[0]), np.max(dataset[0]), 0.001)
    x_tick = np.linspace(np.min(dataset[0]), np.max(dataset[0]))
    y_tick = model.w[0] * x_tick + model.w[1]
    plt.plot(x_tick, y_tick, 'r')
    plt.savefig('xy_plot.png')
