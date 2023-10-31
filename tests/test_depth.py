import sys
import numpy as np
 
# setting path
sys.path.append('../tte_depth/src/')

from tte_depth import StatDepth

if __name__ == "__main__":
    
    # Test w/cosine distance
    depth = StatDepth(distance = "cosine")

    # single corpus
    texts = np.array([[-0.1, 0.1, 0.1],
             [3.14, 3.14, 3.14]])
    depths = depth.depths(texts)
    assert len(depths) == len(texts)

    # multiple corpora
    texts1 = np.array([[3.14, -6.28, 3.14],
             [3.14, 3.14, 3.14],
             [-3.14, -3.14, -3.14]])
    depths_F, depths_G = depth.depths_paired(texts, texts1)
    assert len(depths_F) == len(texts) and len(depths_G) == len(texts1)

    # multiple corpora, different lengths
    texts1 = np.array([[-5.2, 21.0, 101.0],
             [1.2, 1.2, -2.1],
             [0.5, 1.4, -1.0]])
    depths_F, depths_G = depth.depths_paired(texts, texts1)
    assert len(depths_F) == len(texts) and len(depths_G) == len(texts1)

    # multiple corpora, different lengths q test
    depth_scores_F, depth_scores_G, q, w, p = depth.depth_rank_test(texts, texts1)
    assert len(depth_scores_F) == len(texts) and len(depth_scores_G) == len(texts1) and 0.0 <= p and p <= 1.0
    

    # Test w/chord distance
    depth = StatDepth(distance = "chord")
    texts = [[-0.1, 2.0, 1.0],
             [1.5, -2.2, -3.1],
             [0.0, 1.0, 18.0]]
    texts = np.array(texts)
    depths = depth.depths(texts)
    assert len(depths) == len(texts)

    # Test w/user-defined distance
    def mydist(a, b):
        return -1
    depth = StatDepth(distance = mydist)
    texts = [[-0.1, 2.0, 1.0],
             [1.5, -2.2, -3.1],
             [0.0, 1.0, 18.0]]
    texts = np.array(texts)
    depths = depth.depths(texts)
    assert len(depths) == len(texts)

    # Test w/many texts
    def mydist(a, b):
        return -1
    depth = StatDepth(distance = mydist)
    texts = [[-0.1, 2.0, 1.0],
             [1.5, -2.2, -3.1],
             [0.0, 1.0, 18.0]]
    texts1 = [[-0.1, 2.0, 1.0],
             [1.5, -2.2, -3.1],
             [0.0, 1.0, 18.0],
             [-0.1, 2.0, 1.0],
             [1.5, -2.2, -3.1],
             [0.0, 1.0, 18.0]]
    texts = np.array(texts)
    texts1 = np.array(texts1)
    depth_scores_F, depth_scores_G, q, w, p = depth.depth_rank_test(texts, texts1)
    #assert len(depths) == len(texts)

    print("All tests completed correctly!")