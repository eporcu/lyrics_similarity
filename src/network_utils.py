#!/usr/bin/python3

"""
Some utilities for creating networks 
and doing some comparisons between networks.
It leverages on networkx and netrd
"""
# Author : Emanuele Porcu <porcu.emanuele@gmail.com>

from itertools import combinations
import netrd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def is_in_lyrics(lyrics, words):
    """
    check which of the words are
    contained in a song
    """
    return [w for w in words if w in lyrics]


def create_nets(df, words, col="lyrics"):
    """
    weights are simply the number of co-occurrence of two words
    """
    net_dict = {"source": [], "target": [], "weight": []}
    for i in df[col]:
        word_l = is_in_lyrics(i, words)
        if len(word_l) > 0:
            net_dict["source"].extend([word_l[0]] * len(word_l[1:]))
            net_dict["target"].extend(word_l[1:])

    net_dict["weight"] = [1] * len(net_dict["source"])
    net_df = pd.DataFrame(net_dict)
    return net_df.groupby(["source", "target"], as_index=False).count()


def get_networks(df, words):
    """
    creates a dictionary of networks,
    adds and normalize the weights

    Parameters
    ----------
    df : pandas dataframe for a network inputs
    words : list of words

    Returns
    -------
    net_dict : dict of all networks

    """
    net_dict = {}
    for g in df.release_date.unique():
        net_df = df.query(f"release_date == {g}")
        if not net_df.empty:
            net = create_nets(net_df, words, col="lyrics")
            # net.weight = min_max(net.weight, a=1, b=2)
            net_dict[g] = nx.from_pandas_edgelist(
                net,
                source="source",
                target="target",
                edge_attr=True,
                create_using=nx.Graph(),
            )
    return net_dict


def balance_df(df, genre, balance=True):
    """
    makes the number of songs
    equal for each year, by randomly
    selecting a set of songs.
    Number of selected songs is equal
    to the median over the number
    of songs per year.

    Parameters
    ----------
    df : pandas dataframe for a network inputs
    genre : str of selected genre
    balance : bool if True balances the
              dataset over the years, if
              False there is no balancing

    Returns
    -------
    pandas dataframe with balanced elements
    """
    genre_df = df.query(f"genre == '{genre}'")
    if balance:
        grouped = genre_df.groupby(["release_date"], as_index=False).size()
        cut_off = int(np.median(grouped["size"]))
        all_songs = []
        for i in genre_df.release_date.unique():
            year_df = genre_df.query(f"release_date == {i}")
            if len(year_df) >= cut_off:
                all_songs.append(year_df.sample(n=cut_off, random_state=123))
        return pd.concat(all_songs, ignore_index=True)
    else:
        return genre_df


def min_max(data, a=2, b=4):
    """
    min max scaling for weights
    """
    if data.max() == data.min():
        return data
    else:
        data = a + (((data - data.min()) * (b - a)) / (data.max() - data.min()))
    return data


class NetComparison:
    """
    Wrapper around networkx and netrd methods
    for comparisons. It creates a simple
    pipeline.
    """

    def __init__(self, net_dict, method):
        self.net_dict = net_dict
        self.method = method

    def _get_spectra(self):
        """
        runs different selected methods
        for network comparisons: "adjacency", "laplacian",
        "portrait_div", "netlsd".
        For portrait divergence and ntlsd
        runs all the combinations of comparisons
        to mimic the scipy pdist method.

        Parameters
        ----------
        method : str with one of the legal methods
                 "adjacency", "laplacian",
                 "portrait_div", "netlsd"

        Returns
        -------
        spectra : list of spectra for each network
        """
        print(f"running {self.method} method")
        if self.method == "adjacency":
            spectra = [
                nx.adjacency_spectrum(G, weight="weight")
                for _, G in self.net_dict.items()
            ]
        elif self.method == "laplacian":
            spectra = [
                nx.normalized_laplacian_spectrum(G, weight="weight")
                for _, G in self.net_dict.items()
            ]
        elif self.method == "portrait_div":
            comb = list(combinations(list(self.net_dict.keys()), 2))
            spectra = [
                netrd.distance.PortraitDivergence().dist(
                    self.net_dict[i[0]], self.net_dict[i[1]], bins=90
                )
                for i in comb
            ]
        elif self.method == "netlsd":
            comb = list(combinations(list(self.net_dict.keys()), 2))
            spectra = [
                netrd.distance.NetLSD().dist(self.net_dict[i[0]], self.net_dict[i[1]])
                for i in comb
            ]
        else:
            raise NameError(f"There is no method called {self.method}")
        return spectra

    def _zeropad(self, spectra):
        """
        zero pad in case eigenvalues of the spectra
        have different sizes
        """
        size = np.array([len(s) for s in spectra])
        for i in list(np.where(size < size.max())[0]):
            spectra[i] = np.append(spectra[i], np.zeros(size.max() - len(spectra[i])))
        return spectra

    def __call__(self):
        """
        make the pipeline callable and handy
        """
        spectra = self._get_spectra()
        if self.method == "adjacency" or self.method == "laplacian":
            # get real part of spectra and run pdist
            spectra = np.stack(self._zeropad(spectra))
            spectra = pdist(abs(spectra))
        return squareform(spectra)
