"""Example 4 — Detecting distributional shift between human and synthetic text.

Demonstrates the use-case from:
  Seegmiller & Preum, "Statistical Depth for Ranking and Characterizing
  Transformer-Based Text Embeddings," EMNLP 2023.

  Ochs & Habernal, "The Conundrum of Trustworthy Research on Attacking
  Personally Identifiable Information Removal Techniques," CL 2026.
  (cites this measure to characterise the known distributional gap between
  human-written and synthetically generated texts)

Synthetic text generation tends to produce a measurable distributional shift
away from human-written text in embedding space.  TTE depth + the Wilcoxon
rank sum test let you:
  1. Quantify *how much* the shift is (Q parameter).
  2. Determine whether the shift is statistically significant (p-value).
  3. Identify *which* synthetic texts are the most atypical (lowest depth).

Requires:  pip install tte_depth sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

# Human-written natural language inference (NLI) premise sentences
human_texts = [
    "A young child is playing on a swing set at the park.",
    "Two men are arguing on the street corner.",
    "A woman reads a book while sitting on a park bench.",
    "Children are running around in a field on a sunny day.",
    "An elderly man is feeding pigeons near a fountain.",
    "A group of friends is having a picnic by the lake.",
    "A cyclist is riding through a narrow cobblestone alleyway.",
    "Workers are repairing the road outside a busy café.",
]

# Synthetically generated NLI-style sentences (simulating an LLM augmentation)
# These tend to be more formulaic and less natural-sounding
synthetic_texts = [
    "The child is engaged in recreational activities in an outdoor setting.",
    "Two male individuals are engaged in a verbal disagreement.",
    "A female person is in the process of reading written material.",
    "Multiple children are participating in physical outdoor activities.",
    "An aged male individual is distributing food to avian creatures.",
    "A collection of friends is engaged in eating outdoors.",
    "A person operating a bicycle is navigating a narrow pathway.",
    "Construction personnel are conducting road maintenance operations.",
]

F = model.encode(human_texts)
G = model.encode(synthetic_texts)

d = StatDepth()
depth_F, depth_G, Q, W, p = d.depth_rank_test(F, G)

print("Human vs. synthetic text — distributional analysis")
print("=" * 55)
print(f"  Q estimate : {Q:.4f}")
print(f"  W statistic: {W:.4f}")
print(f"  p-value    : {p:.4f}")
print()
if Q < 0.5 and p < 0.05:
    print("  Conclusion: synthetic texts are significantly more outlying")
    print("  (less representative of the human-text distribution).")
elif p >= 0.05:
    print("  Conclusion: no statistically significant distributional shift.")
else:
    print("  Conclusion: mixed signal — inspect depth distributions manually.")

print("\nMost atypical synthetic texts (lowest depth w.r.t. human distribution):")
ranked_synthetic = sorted(zip(synthetic_texts, depth_G), key=lambda x: x[1])
for text, score in ranked_synthetic[:3]:
    print(f"  {score:.4f}  {text}")

print("\nMost atypical human texts (lowest depth w.r.t. own distribution):")
ranked_human = sorted(zip(human_texts, depth_F), key=lambda x: x[1])
for text, score in ranked_human[:3]:
    print(f"  {score:.4f}  {text}")
