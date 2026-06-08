# TTE Depth

**TTE Depth** is a Python library for *transformer-based text embedding (TTE) depth* — a statistical tool for ranking texts within a corpus by centrality, and for detecting distributional shift between two corpora.

It was first described in:

> P. Seegmiller & S. Preum, "Statistical Depth for Ranking and Characterizing Transformer-Based Text Embeddings," *EMNLP 2023*. [[paper]](https://arxiv.org/abs/2310.15010)

---

## What is TTE Depth?

Given a corpus of texts F embedded by a transformer model into vectors of dimension k, the **TTE depth** of embedding x with respect to F is:

$$D_\delta(x, F) = 2 - \mathbb{E}_F\bigl[\delta(x, H)\bigr], \quad H \sim \text{Uniform}(F)$$

where δ is a bounded distance function (cosine or chord distance, both bounded in [0, 2] for unit vectors). In practice, the expectation is estimated as the average pairwise distance from x to all other embeddings in F.

A **high depth score** means a text is close (in direction) to most other texts — it is representative of the corpus. A **low depth score** marks a semantic or linguistic outlier.

The text with maximum depth is the corpus **median** — the most central text.

### Paired Corpora and the Q Parameter

When comparing two corpora F and G, TTE depth gives a center-outward ordering of G relative to F's distribution. The **Q parameter** summarises this:

$$Q(F, G) = \Pr\bigl[D_\delta(X, F) \leq D_\delta(Y, F)\bigr], \quad X \sim F,\; Y \sim G$$

- Q ≈ 0.5 → F and G come from the same distribution
- Q < 0.5 → G is more outlying than F (distributional shift)

A one-sided **Wilcoxon rank sum test** on the depth-induced ordering provides a p-value for whether the shift is statistically significant.

**Recommended sample size:** n = 500 per corpus gives reliable Q estimates with low variance (see §4.1 of the paper).

---

## Installation

```bash
pip install tte_depth
```

To also install dependencies for the examples:

```bash
pip install "tte_depth[examples]"
```

Or clone the repository:

```bash
git clone https://github.com/pkseeg/tte_depth.git
cd tte_depth
pip install -e ".[dev]"
```

---

## Quick Start

TTE depth is designed to work alongside any transformer embedding model. The examples below use [`sentence-transformers`](https://www.sbert.net/):

```bash
pip install sentence-transformers
```

### Single-corpus depth

Rank texts by how representative they are of a corpus:

```python
from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "TTE depth assigns each text a centrality score.",
    "Transformer embeddings map texts to dense vectors.",
    "Contrastive training improves semantic alignment.",
    "I love hiking and playing soccer on weekends!",   # outlier
]

F = model.encode(texts)
d = StatDepth()
scores = d.depths(F)

for text, score in sorted(zip(texts, scores), key=lambda x: -x[1]):
    print(f"{score:.4f}  {text}")
```

```
1.4821  Transformer embeddings map texts to dense vectors.
1.4105  TTE depth assigns each text a centrality score.
1.3892  Contrastive training improves semantic alignment.
0.9134  I love hiking and playing soccer on weekends!
```

The off-topic sentence correctly receives the lowest depth score.

### Paired-corpus depth and significance test

Test whether two corpora differ significantly in embedding space:

```python
from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

F = model.encode([...])   # reference corpus
G = model.encode([...])   # query corpus

d = StatDepth()
depth_F, depth_G, Q, W, p = d.depth_rank_test(F, G)

print(f"Q = {Q:.2f}, W = {W:.2f}, p = {p:.4f}")
```

### Distance functions

```python
# Cosine distance (default) — good for most embedding models
d = StatDepth(distance="cosine")

# Chord distance — equivalent to cosine for unit vectors, slightly different
# geometric interpretation (arc length on the unit sphere)
d = StatDepth(distance="chord")

# Custom distance — any callable (a, b) → float bounded in [0, 2]
d = StatDepth(distance=lambda a, b: my_dist(a, b))
```

---

## API Reference

### `StatDepth(distance="cosine")`

| Method | Description |
|---|---|
| `depths(F)` | Depth scores for all embeddings in corpus F. Returns `np.ndarray` of shape `(n,)`. |
| `depths_paired(F, G)` | Depth scores for F and G, both w.r.t. F's distribution. Returns `(depth_F, depth_G)`. |
| `q_score(x, y)` | Q parameter from two arrays of depth scores. Returns `float`. |
| `depth_rank_test(F, G)` | Full paired analysis: returns `(depth_F, depth_G, Q, W, p)`. |

---

## Downstream Use Cases

The examples in [`examples/`](examples/) illustrate three applications from downstream work that cites TTE depth:

### 1. Outlying Roles in Structured Information Extraction

**Paper:** Gatto et al., "Document-Level Event-Argument Data Augmentation for Challenging Role Types," *ACL 2025*.

In few-shot cross-domain Event Argument Extraction (EAE), some target-domain event roles are semantically distant from anything seen in the source domain. These *challenging roles* are hardest for models to generalise to.

TTE depth flags them automatically: embed source and target role-type names, compute paired depth scores, and sort ascending. Low-depth target roles are the most semantically outlying — the ones that need targeted data augmentation.

```python
# See examples/3_outlying_roles_eae.py
depth_F, depth_G = d.depths_paired(source_role_embeddings, target_role_embeddings)
challenging_roles = sorted(zip(target_roles, depth_G), key=lambda x: x[1])
```

### 2. Distinguishing Human-Written from Synthetic Text

**Papers:** Seegmiller & Preum (EMNLP 2023); Ochs & Habernal, "The Conundrum of Trustworthy Research on Attacking PII Removal Techniques," *Computational Linguistics 2026*.

Synthetic text generation produces a measurable distributional shift away from human-written text in embedding space. The Wilcoxon rank sum test on TTE depth reliably detects this shift and quantifies it via the Q parameter.

```python
# See examples/4_synthetic_text_detection.py
depth_F, depth_G, Q, W, p = d.depth_rank_test(human_embeddings, synthetic_embeddings)
# Q < 0.5 and p < 0.05 indicates the synthesis process caused a significant shift
```

### 3. Measuring Distribution Shift in User Prompts

**Paper:** Seegmiller & Preum, "Measuring Distribution Shift in User Prompts and Its Effects on LLM Performance," *arXiv 2025*.

When an LLM is deployed, user prompts naturally shift over time (new users, tasks, phrasing styles). The LENS framework uses TTE depth to:
- Compute the **Δ-score** (= Q score) to quantify how much post-deployment prompts have shifted from the training distribution.
- Select **shift-representative examples** for qualitative analysis: high-depth training prompts vs. low-depth deployment prompts.

```python
# See examples/5_prompt_distribution_shift.py
depth_train, depth_deploy, delta, W, p = d.depth_rank_test(P_train, P_deploy)
# Select most characteristic shifted prompts:
outlying_prompts = sorted(zip(prompts, depth_deploy), key=lambda x: x[1])[:10]
```

---

## How It Compares to Related Tools

| Tool | Scope | What it measures |
|---|---|---|
| **TTE Depth** | Corpus-level centrality + shift | Center-outward ordering of embeddings; Q-score for distributional shift |
| [MAUVE](https://github.com/krishnap25/mauve) | Corpus-level distribution | KL-divergence frontier between text token distributions |
| Cosine similarity | Pairwise | Angle between two specific embeddings |
| Perplexity | Sequence-level | Surprise of a text under a language model |

TTE depth is complementary to these tools — it provides an interpretable scalar depth score per text, which enables both ranking within a corpus and formal statistical testing between corpora.

---

## Citation

If you use TTE depth in your work, please cite:

```bibtex
@inproceedings{seegmiller-preum-2023-statistical,
    title     = "Statistical Depth for Ranking and Characterizing Transformer-Based Text Embeddings",
    author    = "Seegmiller, Parker and Preum, Sarah Masud",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month     = dec,
    year      = "2023",
    address   = "Singapore",
    publisher = "Association for Computational Linguistics",
    url       = "https://arxiv.org/abs/2310.15010",
    pages     = "9600--9611",
}
```

---

## License

MIT — see [LICENSE](LICENSE).
