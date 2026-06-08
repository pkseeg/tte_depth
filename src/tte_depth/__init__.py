"""tte_depth — Transformer-based Text Embedding (TTE) Depth.

A statistical depth for ranking text embeddings by centrality within a
corpus, and for measuring distributional shift between two corpora.

Reference
---------
P. Seegmiller & S. Preum, "Statistical Depth for Ranking and Characterizing
Transformer-Based Text Embeddings," EMNLP 2023.
https://arxiv.org/abs/2310.15010
"""

from tte_depth._depth import StatDepth
from tte_depth._utils import chord_dist, cosine_dist

__all__ = ["StatDepth", "chord_dist", "cosine_dist"]
__version__ = "1.1.0"
