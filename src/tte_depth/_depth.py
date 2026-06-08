from __future__ import annotations

import numpy as np
from scipy.stats import ranksums, rankdata
from typing import Callable, Union

from tte_depth._utils import chord_dist, cosine_dist, create_dist_matrix

# Supremum of cosine and chord distances for unit vectors
_D_SUP = 2.0


class StatDepth:
    """Statistical depth for transformer-based text embeddings (TTE Depth).

    Assigns each text embedding in a corpus a *depth score*: a scalar that
    measures how representative (central) the text is of the corpus as a whole.
    Formally, for a corpus F and bounded distance δ the depth of embedding x is

        D_δ(x, F) = 2 − E_F[δ(x, H)],   H ∼ Uniform(F)

    Higher depth → more representative; lower depth → semantic outlier.

    The class also supports paired-corpus comparison via a Wilcoxon rank sum
    test on the depth-induced ordering, with an associated Q parameter that
    quantifies how outlying corpus G is relative to corpus F.

    Parameters
    ----------
    distance:
        Distance function between two embeddings.  Pass ``"cosine"`` (default)
        or ``"chord"`` to select the built-in bounded distances, or supply any
        callable ``dist(a, b) -> float`` bounded in [0, 2].

    References
    ----------
    P. Seegmiller & S. Preum, "Statistical Depth for Ranking and Characterizing
    Transformer-Based Text Embeddings," EMNLP 2023.
    """

    def __init__(
        self,
        distance: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "cosine",
    ) -> None:
        if distance == "cosine":
            self.dist = cosine_dist
        elif distance == "chord":
            self.dist = chord_dist
        else:
            self.dist = distance

    # ------------------------------------------------------------------
    # Single-corpus depth
    # ------------------------------------------------------------------

    def depths(self, F: np.ndarray) -> np.ndarray:
        """Depth scores for every embedding in a single corpus F.

        Parameters
        ----------
        F:
            Array of shape (n, d) — n text embeddings of dimension d.

        Returns
        -------
        np.ndarray of shape (n,)
            Depth score for each embedding.  Higher → more representative.
        """
        if not isinstance(F, np.ndarray):
            raise TypeError("F must be a numpy array")

        n = len(F)
        D = create_dist_matrix(F, F, self.dist)
        # Exclude self-distance; divide by n-1
        avg_dists = (np.sum(D, axis=1) - np.diag(D)) / (n - 1)
        return _D_SUP - avg_dists

    # ------------------------------------------------------------------
    # Paired-corpus depth
    # ------------------------------------------------------------------

    def depths_paired(
        self, F: np.ndarray, G: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Depth scores for F and G, both measured relative to F's distribution.

        Parameters
        ----------
        F:
            Array of shape (m, d) — reference corpus embeddings.
        G:
            Array of shape (n, d) — query corpus embeddings.

        Returns
        -------
        depth_scores_F : np.ndarray of shape (m,)
        depth_scores_G : np.ndarray of shape (n,)
        """
        if not (isinstance(F, np.ndarray) and isinstance(G, np.ndarray)):
            raise TypeError("F and G must both be numpy arrays")

        m = len(F)
        D_FF = create_dist_matrix(F, F, self.dist)
        D_FG = create_dist_matrix(F, G, self.dist)

        avg_dists_F = (np.sum(D_FF, axis=1) - np.diag(D_FF)) / (m - 1)
        avg_dists_G = np.mean(D_FG, axis=0)  # shape (n,): mean over F for each G point

        return (_D_SUP - avg_dists_F), (_D_SUP - avg_dists_G)

    # ------------------------------------------------------------------
    # Q score and rank test
    # ------------------------------------------------------------------

    def q_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Q parameter: average outlyingness of G (y) relative to F (x).

        Q ≈ 0.5 when F and G are from the same distribution.
        Q < 0.5 indicates G is more outlying than F (distributional shift).

        Parameters
        ----------
        x:
            Depth scores from corpus F (reference).
        y:
            Depth scores from corpus G (query).

        Returns
        -------
        float
            Estimated Q(F, G) ∈ [0, 1].
        """
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("x and y must both be numpy arrays")

        n1 = len(x)
        alldata = np.concatenate((x, y))
        ranked = rankdata(alldata)
        y_ranks = ranked[n1:]

        # For each G rank, fraction of combined ranks below it — vectorised
        num_less = np.mean(ranked < y_ranks[:, np.newaxis], axis=1)
        return float(np.mean(num_less))

    def depth_rank_test(
        self, F: np.ndarray, G: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Paired depth scores + Wilcoxon rank sum test for F vs G.

        Tests H₀: F = G versus Hₐ: G is more outlying than F (Q < 0.5).

        Parameters
        ----------
        F:
            Array of shape (m, d) — reference corpus embeddings.
        G:
            Array of shape (n, d) — query corpus embeddings.

        Returns
        -------
        depth_scores_F : np.ndarray
            Depth scores for embeddings in F w.r.t. F.
        depth_scores_G : np.ndarray
            Depth scores for embeddings in G w.r.t. F.
        q : float
            Q parameter estimate — outlyingness of G relative to F.
        w : float
            Wilcoxon rank sum test statistic.
        p : float
            p-value for the one-sided test (small p → G is significantly
            more outlying than F).
        """
        depth_scores_F, depth_scores_G = self.depths_paired(F, G)
        q = self.q_score(depth_scores_F, depth_scores_G)
        w, p = ranksums(depth_scores_F, depth_scores_G, alternative="greater")
        return depth_scores_F, depth_scores_G, q, float(w), float(p)
