#!/usr/bin/python3

"""
Wrapper for quick LDA with scikit-learn
"""

# Author : Emanuele Porcu <porcu.emanuele@gmail.com>

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicModeling:
    """
    Performs bag of words, Latent Dirichlet Allocation
    using the scikit-learn algo and
    it extract the most relevant words.
    """

    def __init__(self, max_df, min_df, n_feat):
        self.max_df = max_df
        self.min_df = min_df
        self.n_feat = n_feat

    def _get_bag_of_words(self):
        """creates a bag of words"""
        tf_vectorizer = CountVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.n_feat,
            stop_words="english",
        )
        return tf_vectorizer

    def __call__(self, data, components, n_top_words, max_iter):
        """
        run LDA as callable
        Parameters
        ----------
        data : list of texts (corpus)
        components : int, number of potential topics
        n_top_words : int number of most relevant words
                      according to weights
        max_iter : int number of iterations of the LDA model

        Returns
        -------
        all_topics : set of strings, all the unique most
                     relevant words
        """
        tf_vectorizer = self._get_bag_of_words()
        tf = tf_vectorizer.fit_transform(data)
        feat_names = tf_vectorizer.get_feature_names_out()
        lda = LatentDirichletAllocation(
            n_components=components,
            max_iter=max_iter,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )

        lda.fit(tf)

        all_topics = []
        for topic in lda.components_:
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feat_names[i] for i in top_features_ind]
            all_topics.extend(top_features)
        return set(all_topics)
