#!/usr/bin/python3

"""
collection of plotting functions for networks and diss. matrices
"""

# Author : Emanuele Porcu <porcu.emanuele@gmail.com>

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import numpy as np


def plot_network(G, title, save=False):
    """
    plot a single network of words
    with weights
    """
    fig = plt.figure()
    plt.title(title)
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    pos = nx.spring_layout(G, k=1, iterations=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=weights, alpha=0.08, edge_color="b"
    )
    plt.axis("off")
    if save:
        fig.savefig(f"{title}.png", transparent=False)
    plt.show()


def plot_nets(net_dict, nrows, ncols, save=False, del_extra_axes=None):
    """plots multiple networks as subplots"""

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 30))
    for g, ax in zip(net_dict.keys(), axes.ravel()):
        ax.set_title(g, fontdict={"fontsize": 30})
        edges = net_dict[g].edges()
        weights = [net_dict[g][u][v]["weight"] for u, v in edges]
        pos = nx.spring_layout(net_dict[g], k=1, iterations=20)
        nx.draw_networkx_labels(
            net_dict[g], pos, font_size=20, font_family="sans-serif", ax=ax
        )
        nx.draw_networkx_edges(
            net_dict[g],
            pos,
            edgelist=net_dict[g].edges(),
            width=weights,
            alpha=1,
            edge_color="b",
            ax=ax,
        )
        ax.set_axis_off()

    # delete extra axis
    plt.delaxes(axes.ravel()[del_extra_axes])
    plt.tight_layout()
    if save:
        fig.savefig("genre.png")
    plt.show()


def plot_dissmat(data, titles, labels, save=False, fig_name=None, nlabels=10):
    """
    plots dissimilarity matrices
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    for s, ax, tit in zip(data, axes.ravel(), titles):
        ax.set_title(f"{tit}", fontdict={"fontsize": 10})
        im = ax.imshow(s)
        prop_labels = list(range(0, len(s), len(s) // nlabels))
        ax.set_xticks(prop_labels)
        ax.set_xticklabels(np.array(labels)[prop_labels], rotation=70, fontsize=15)
        ax.set_yticks(prop_labels)
        ax.set_yticklabels(np.array(labels)[prop_labels], fontsize=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    plt.tight_layout(h_pad=1)
    if save:
        fig.savefig(f"{fig_name}.png", transparent=False)
    plt.show()


def plot_similarity(distances, x, title, legend=None, save=False):
    """
    plots times series of similarities
    """
    fig = plt.figure(figsize=(15, 5))
    plt.title(title)
    for i in distances:
        plt.plot(i, linewidth=2)
        plt.xticks(ticks=range(len(i)), labels=x, rotation=70, fontsize=15)
        plt.ylabel("distance [a.u.]", fontsize=15)
    if legend is not None:
        plt.legend(legend)
    plt.tight_layout()
    if save:
        fig.savefig(f"{title}.png", transparent=False)
    plt.show()
