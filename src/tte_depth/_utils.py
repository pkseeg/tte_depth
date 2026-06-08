from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, cosine
from typing import Callable


def chord_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Chord distance between two vectors.

    Bounded in [0, 2] for unit vectors: chord(x, y) = sqrt(2*(1 - x·y)).
    """
    val = 2.0 * (1.0 - np.dot(a, b))
    return float(np.sqrt(max(val, 0.0)))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors: 1 - cos_similarity(a, b)."""
    return float(cosine(a, b))


def create_dist_matrix(
    F: np.ndarray,
    G: np.ndarray,
    dist: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Pairwise distance matrix between rows of F and rows of G.

    Uses vectorized cdist paths for built-in distances; falls back to
    scipy's callable cdist for custom metrics.
    """
    if dist is cosine_dist:
        return cdist(F, G, metric="cosine")
    if dist is chord_dist:
        # chord = sqrt(2 * cosine_dist) for unit vectors
        return np.sqrt(np.clip(2.0 * cdist(F, G, metric="cosine"), 0.0, None))
    return cdist(F, G, metric=dist)


def avg_dist(i: int, distances: np.ndarray, selfcorrect: bool = False) -> float:
    """Average distance from point i to all other points.

    Parameters
    ----------
    i:
        Row index into `distances`.
    distances:
        A 2-D distance matrix. Square when computing depths within a single
        corpus; rectangular (|F| × |G|) when computing G depths w.r.t. F.
    selfcorrect:
        When True (single-corpus case), exclude the self-distance at
        distances[i, i] so the average is over the n-1 other points.
    """
    if selfcorrect:
        n = distances.shape[0]
        return float((np.sum(distances[i]) - distances[i, i]) / (n - 1))
    if distances.shape[0] == distances.shape[1]:
        return float(np.mean(distances[i]))
    # Rectangular matrix: G depths w.r.t. F — average over F column
    return float(np.mean(distances[:, i]))
