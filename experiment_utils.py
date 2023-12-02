"""Utility functions for experiments."""

import matplotlib.pyplot as plt


def visualize_history(history: dict, measure1, measure2, logscale=False):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history[measure1])
    axs[0].plot(history["val_" + measure1])
    axs[0].set_xlabel("epoch")
    axs[0].set_xlabel(measure1)
    if logscale:
        axs[0].set_yscale("log")
    axs[1].plot(history[measure2], label="training")
    axs[1].plot(history["val_" + measure2], label="validation")
    axs[1].set_xlabel("epoch")
    axs[1].set_xlabel(measure2)
    fig.legend(labelcolor="linecolor")
    fig.tight_layout(pad=1.5)
    return fig
