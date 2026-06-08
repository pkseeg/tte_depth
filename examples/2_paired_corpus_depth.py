"""Example 2 — Paired-corpus depth and Wilcoxon rank sum test.

Measures how far corpus G is from corpus F in embedding space and tests
whether the shift is statistically significant.

The key output:
  Q  ≈ 0.5  → G and F come from the same distribution
  Q  < 0.5  → G is more outlying than F (distributional shift detected)
  p  < 0.05 → the shift is statistically significant

Requires:  pip install tte_depth sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus F — NLP / machine learning texts
nlp_texts = [
    "Transformer models encode text into dense vector representations.",
    "Sentence embeddings capture semantic relationships between documents.",
    "Pre-trained language models are fine-tuned on downstream tasks.",
    "Attention mechanisms allow models to weigh relevant context.",
    "Statistical depth provides a center-outward ranking of embeddings.",
    "Cosine similarity measures the angle between two embedding vectors.",
]

# Corpus G — culinary texts (clearly off-domain)
cooking_texts = [
    "Sauté the onions in olive oil until they are translucent.",
    "Fold the egg whites gently into the batter to keep it light.",
    "Let the dough rest for one hour before shaping the loaves.",
    "Season with salt, pepper, and fresh thyme before roasting.",
    "Reduce the sauce over medium heat until it coats the back of a spoon.",
    "Blanch the vegetables briefly to preserve their colour and crunch.",
]

F = model.encode(nlp_texts)
G = model.encode(cooking_texts)

d = StatDepth()
depth_F, depth_G, Q, W, p = d.depth_rank_test(F, G)

print("Paired-corpus depth analysis")
print("=" * 50)
print(f"  Q estimate : {Q:.4f}  (< 0.5 → G is more outlying than F)")
print(f"  W statistic: {W:.4f}")
print(f"  p-value    : {p:.4f}  ({'significant shift detected' if p < 0.05 else 'no significant shift'})")

print("\nF depths (NLP corpus, w.r.t. F):")
for text, score in zip(nlp_texts, depth_F):
    print(f"  {score:.4f}  {text[:55]}...")

print("\nG depths (cooking corpus, w.r.t. F):")
for text, score in zip(cooking_texts, depth_G):
    print(f"  {score:.4f}  {text[:55]}...")
