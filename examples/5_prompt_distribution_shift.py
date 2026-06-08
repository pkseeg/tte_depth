"""Example 5 — Measuring distribution shift in user prompts over time.

Demonstrates the use-case from:
  Seegmiller & Preum, "Measuring Distribution Shift in User Prompts and Its
  Effects on LLM Performance," arXiv 2025.

When an LLM is deployed, the distribution of user prompts naturally shifts
over time as new users, tasks, and interaction patterns emerge.  This shift
can significantly degrade model performance.

The LENS framework (cited paper) uses TTE depth in two ways:
  1. As the Δ-score (= Q-score): Q(P, Q_new) quantifies how much the
     post-deployment prompt distribution Q_new has shifted from the
     training distribution P.  Q < 0.5 → prompts are more outlying.
  2. To select *shift-representative examples* for qualitative analysis:
     - High-depth ID prompts  = most typical of training distribution
     - Low-depth OOD prompts  = most characteristic of the shift

This example simulates an instruction-tuning deployment where initial prompts
are task-focused and later prompts are more casual and open-ended.

Requires:  pip install tte_depth sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

# In-distribution prompts (training time) — task-focused instructions
training_prompts = [
    "Summarize the following article in three bullet points.",
    "Translate the sentence below from English to French.",
    "List five advantages of renewable energy.",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a professional email declining a meeting invitation.",
    "What are the main causes of the French Revolution?",
    "Convert the temperature 98.6 degrees Fahrenheit to Celsius.",
    "Give me three synonyms for the word 'happy'.",
    "Proofread and correct the grammar in the following paragraph.",
    "Describe the water cycle in simple terms.",
]

# Out-of-distribution prompts (post-deployment) — casual, open-ended
deployment_prompts = [
    "hey can u help me write smth for my friend lol",
    "whats the best movie rn? just need something fun",
    "idk what to eat for dinner tonight any ideas??",
    "my boss is being kinda weird lately what should i do",
    "tell me something interesting i can impress ppl with",
    "how do i not feel nervous at parties",
    "what do u think about astrology actually",
    "literally can't decide between two job offers help",
    "is it weird to text someone first always",
    "haha so what are ur thoughts on pineapple pizza",
]

P = model.encode(training_prompts)
Q_new = model.encode(deployment_prompts)

d = StatDepth()
depth_P, depth_Q, delta, W, p = d.depth_rank_test(P, Q_new)

print("Prompt distribution shift analysis (training → deployment)")
print("=" * 60)
print(f"  Δ-score (Q) : {delta:.4f}  (< 0.5 → deployment prompts more outlying)")
print(f"  W statistic : {W:.4f}")
print(f"  p-value     : {p:.4f}  ({'significant shift' if p < 0.05 else 'no significant shift'})")

# Shift-representative pairs for qualitative analysis
# High-depth training prompts are the most "canonical" training examples
# Low-depth deployment prompts are the most characteristic of the shift
high_depth_train = sorted(zip(training_prompts, depth_P), key=lambda x: -x[1])
low_depth_deploy = sorted(zip(deployment_prompts, depth_Q), key=lambda x: x[1])

print("\nMost representative training prompts (highest training-distribution depth):")
for text, score in high_depth_train[:3]:
    print(f"  {score:.4f}  {text}")

print("\nMost shift-characteristic deployment prompts (lowest training-distribution depth):")
for text, score in low_depth_deploy[:3]:
    print(f"  {score:.4f}  {text}")

print()
print("These prompt pairs can be used for qualitative analysis of what")
print("aspects of the distribution shifted — style, formality, task type, etc.")
print()
print(f"Training distribution median depth   : {np.median(depth_P):.4f}")
print(f"Deployment distribution median depth : {np.median(depth_Q):.4f}")
