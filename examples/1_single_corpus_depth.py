"""Example 1 — Single-corpus depth scoring.

Assigns each text in a corpus a depth score indicating how representative
(central) it is relative to the corpus as a whole.  High depth → typical,
representative text; low depth → semantic or linguistic outlier.

Requires:  pip install tte_depth sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Statistical depth provides a center-outward ordering of text embeddings.",
    "TTE depth assigns each text a score indicating how representative it is.",
    "Transformer-based embeddings map texts to high-dimensional vectors.",
    "Embedding models trained with contrastive objectives capture semantic similarity.",
    "I love playing soccer and hiking on weekends!",  # off-topic outlier
]

F = model.encode(texts)

d = StatDepth()           # defaults to cosine distance
scores = d.depths(F)

print("Depth scores (higher = more representative):")
print("-" * 60)
for text, score in sorted(zip(texts, scores), key=lambda x: -x[1]):
    print(f"  {score:.4f}  {text}")

median_idx = scores.argmax()
print(f"\nCorpus median (most central text):\n  → {texts[median_idx]}")
