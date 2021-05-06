import os
import matplotlib.pyplot as plt
import numpy as np

from .heatmap import heatmap


def plt_distribution(data, bins, title, xlabel, ylabel, save_dir):
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    plt.gca().set(title=title, 
                  xlabel=xlabel, 
                  ylabel=ylabel)
    plt.savefig(save_dir)


def plt_corr_matrix(corr_mat, feat_lab, save_dir):
    fig, ax = plt.subplots(figsize=(15, 15))
    im, cbar = heatmap(corr_mat, feat_lab, feat_lab, ax=ax, cmap="YlGn")
    fig.tight_layout()
    plt.savefig(save_dir)


def plt_eval_metrics(x_data, y_data, prefix, save_dir):
    fig, ax = plt.subplots(1, 5, sharex=True)
    x = np.array(x_data)
    for i, item in enumerate(y_data.items()):
        k, v = item
        sub_ax = ax[i]
        sub_ax.plot(x, np.array(v), label=k)
        sub_ax.legend()
        np.save(os.path.join(save_dir, prefix + k), np.array(v))
    plt.savefig(os.path.join(save_dir, prefix))

