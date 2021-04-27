import os
import matplotlib.pyplot as plt

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