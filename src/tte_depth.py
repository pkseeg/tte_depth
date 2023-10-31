from utils import chord_dist, cosine_dist, count_less, create_dist_matrix, avg_dist
from scipy.stats import ranksums, rankdata
import numpy as np
from typing import Union, Callable

class StatDepth:
    """
    A class for calculating transformer-based text embedding depth

    Attributes
    ----------
    dist : Callable[[np.ndarray, np.ndarray], float]
        A distance function between two text embeddings (numpy arrays).
    
    Methods
    -------
    depths(F)
        Calculates depth scores for a single array of text embeddings.
    depths_paired(F, G)
        Calculates depth scores for a pair of arrays of text embeddings, F and G, with respect to F.
    q_score(x, y)
        Calculates q score for two arrays of depth scores.
    depth_rank_test(F, G)
        Calculates depth scores for a pair of arrays of text embeddings, F and G, with respect to F, and runs a Wilcoxon rank sum test for determining whether F and G are likely to have come from the same distribution.
    
    Notes
    -----
    StatDepth is an implementation of transformer-based text embedding depth (TTE Depth)[1]_.

    .. [1] P. Seegmiller & S. Preum, "Statistical Depth for Ranking and 
        Characterizing Transformer-Based Text Embeddings," Empirical
        Methods in Natural Language Processing (EMNLP), 2023.
    """

    def __init__(self, distance : Union[str, Callable[[np.ndarray, np.ndarray], float]] = "cosine"):
        """
        Parameters
        ----------
        distance: Union[str, Callable[[np.ndarray, np.ndarray], float]], optional
            Distance function between two embeddings, built-ins "cosine" and "chord" distances can be identified via str (default is "cosine").
        """
        if distance == "cosine":
            self.dist = cosine_dist
        elif distance == "chord":
            self.dist = chord_dist
        else:
            self.dist = distance
    
    def depths(self, F : np.array) -> np.array:
        """
        Parameters
        ----------
        F: np.array
            Array of text embeddings.
        
        Returns
        -------
        np.array
            Array of depth scores, corresponding to text embeddings in F.
        """
        assert isinstance(F, np.ndarray), "F must be type np.array"
        d_sup = 2
        distances_FF = create_dist_matrix(F, F, self.dist)

        depth_scores_F = []
        for i, emb in enumerate(F):
            e_h = avg_dist(i, distances_FF, selfcorrection=True)
            depth_scores_F.append(d_sup - e_h)

        depth_scores_F = np.array(depth_scores_F)

        return depth_scores_F
    
    def depths_paired(self, F : np.array, G : np.array) -> tuple[np.array, np.array]:
        """
        Parameters
        ----------
        F: np.array
            Array of text embeddings (from corpus F).
        G: np.array
            Array of text embeddings (from corpus G).
        
        Returns
        -------
        np.array, np.array
            Array of depth scores, corresponding to text embeddings in F and text embeddings in G.
        """
        assert isinstance(F, np.ndarray) and isinstance(G, np.ndarray), "F and G must both be type np.array"
        d_sup = 2
        distances_FF = create_dist_matrix(F, F, self.dist)
        distances_FG = create_dist_matrix(F, G, self.dist)

        depth_scores_F = []
        for i, emb in enumerate(F):
            e_h = avg_dist(i, distances_FF, selfcorrection=True)
            depth_scores_F.append(d_sup - e_h)
        
        depth_scores_G = []
        for i, emb in enumerate(G):
            e_h = avg_dist(i, distances_FG)
            depth_scores_G.append(d_sup - e_h)

        depth_scores_F = np.array(depth_scores_F)
        depth_scores_G = np.array(depth_scores_G)

        return depth_scores_F, depth_scores_G

    def q_score(self, x : np.array, y : np.array) -> float:
        """
        Parameters
        ----------
        x: np.array
            Array of depth scores (calculated from corpus F).
        y: np.array
            Array of depth scores (calculated from corpus G).
        
        Returns
        -------
        float
            Q score for two depth score arrays.
        """
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "x and y must both be type np.array"
        x, y = map(np.asarray, (x, y))
        n1 = len(x)
        n2 = len(y)
        alldata = np.concatenate((x, y))
        ranked = rankdata(alldata)
        y_ranks = ranked[n1:]
        num_less = np.array([count_less(ranked, y) for y in y_ranks])
        q = np.average(num_less)
        return q

    def depth_rank_test(self, F : np.array, G : np.array) -> tuple[np.array, np.array, float, float, float]:
        """
        Parameters
        ----------
        F: np.array
            Array of text embeddings (from corpus F).
        G: np.array
            Array of text embeddings (from corpus G).
        
        Returns
        -------
        depth_scores_F
            Depth scores for text embeddings F, corresponding to F.
        depth_scores_G
            Depth scores for text embeddings G, corresponding to F.
        q
            Q score for two depth score arrays corresponding to F and G.
        w
            W test statistic for Wilcoxon rank sum test.
        p
            p-value for Wilcoxon rank sum test.
        """
        m, n = F.shape[0], G.shape[0]
        depth_scores_F, depth_scores_G = self.depths_paired(F, G)
        q = self.q_score(depth_scores_F, depth_scores_G)
        w, p = ranksums(depth_scores_F, depth_scores_G, alternative='greater')
        return depth_scores_F, depth_scores_G, q, w, p