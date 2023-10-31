import numpy as np
from scipy.spatial.distance import cosine
from typing import Callable

def chord_dist(a : np.array, b : np.array) -> float:
    """Implements chord distance between two numpy arrays.
    
    """
    val = 2. * (1. - np.matmul(a.T, b))
    if val < 0.00001:
      val = 0.0
    return np.sqrt(val)

def cosine_dist(a : np.array, b : np.array) -> float:
    """Implements cosine distance between two numpy arrays.
    
    """
    return cosine(a, b)

def count_less(ranks : np.array, y : float) -> float:
    """Utility function for counting the number of values in `ranks` array less than `y` value.
    
    """
    count = 0
    for r in ranks:
      if r < y:
        count += 1
    return count / len(ranks)

def create_dist_matrix(F : np.array, G : np.array, dist : Callable[[np.ndarray, np.ndarray], float]) -> np.array:
    """Creates distance matrix between `F` and `G`, using `dist` function.
    
    """
    distances = np.zeros((F.shape[0], G.shape[0]))
    for i, e1 in enumerate(F):
        for j, e2 in enumerate(G):
            distances[i][j] = dist(e1, e2)
    return distances

def avg_dist(i, distances, selfcorrection=False):
    """Calculates average distance for value `i`.
    
    """
    if distances.shape[0] == distances.shape[1]: # if the distance matrix is symmetric (which it always is for single-corpus depth)
        if selfcorrection:
            return np.sum(distances[i, :])/(distances.shape[0]-1.)
        else:
            return np.average(distances[i, :])
    else: # this will only happen for the second corpus of asymmetric paired depth
        return np.average(distances[:, i])
