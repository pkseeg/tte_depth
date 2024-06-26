# TTE Depth

This package is an implementation of transformer-based text embedding depth, first described in [Statistical Depth for Ranking and Characterizing Transformer-Based Text Embeddings (Seegmiller & Preum, EMNLP 2023)](https://arxiv.org/abs/2310.15010).

## Installation

`tte_depth` is available on pypi and can be installed using `pip`.

```bash
pip install tte_depth
```

Alternatively, the repository can be cloned via github.

```bash
git clone https://github.com/pkseeg/tte_depth.git
```

## Use

There are two main use cases of `tte_depth`, both designed to be used in conjunction with a transformer-based text embedding model such as [SBERT](https://www.sbert.net/). We use the popular `sentence_transformers` package in our examples, but any transformer-based text embedding model which embeds texts into vectors of uniform dimension will work.

First, be sure to install `sentence-transformers` and `tte-depth`.

```bash
pip install -U sentence-transformers tte-depth
```

### Single Corpus Depth
`tte_depth` allows you to assign a value to each text in a corpus, indicating how _representative_ each text is of the corpus as a whole. Larger depth values indicate higher representativeness, and lower depth values indicate that the text is a semantic or linguistic outlier.

```python
from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["tte_depth is a python package which implements transformer-based text embedding depth.",
         "Transformer-based text embedding depth is a statistical tool for selecting representative texts from a large corpus.",
         "This can be useful in a variety of contexts, including NLP modeling and inference tasks.",
         "I am an outlier sentence! I love all sports!"]

# encode all texts using the sentence transformer model
F = model.encode(texts)

# calculate depth values for each text embedding in F
d = StatDepth()
depth_scores = d.depths(F)

for t, s in zip(texts, depth_scores):
    print(f"Text: {t} -> Depth Score: {s}")
```

### Paired Corpora Depth and Statistical Significance Testing
`tte_depth` also allows you to measure how far apart two corpora are in embedding space, and to use a Wilcoxon Rank Sum Test to determine whether it is likely that the text embeddings of these two corpora are drawn from the same distribution. In this example, we define two lists of sentences (`F` and `G`) meant to represent different corpora.


```python
from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["tte_depth is a python package which implements transformer-based text embedding depth.",
         "Transformer-based text embedding depth is a statistical tool for selecting representative texts from a large corpus.",
         "This can be useful in a variety of contexts, including NLP modeling and inference tasks."]

other_texts = ["Where are you? And I'm so sorry",
               "I cannot sleep, I cannot dream tonight",
               "I need somebody and always",
               "This sick, strange darkness",
               "Comes creeping on, so haunting every time",
               "And as I stare, I counted",
               "The webs from all the spiders",
               "Catching things and eating their insides",
               "Like indecision to call you",
               "And hear your voice of treason",
               "Will you come home and stop this pain tonight?",
               "Stop this pain tonight"]

# encode all texts using the sentence transformer model
F = model.encode(texts)
G = model.encode(other_texts)

# depth_rank_test returns depth scores for each corpus, along with a Q estimate, W test statistic from the Wilcoxon Rank Sum Test, and an associated p-value
d = StatDepth()
depth_scores_F, depth_scores_G, Q, W, p = d.depth_rank_test(F, G)

print(f"Q = {Q:.2f}, W = {W:.2f}, p = {p:.4f}")
```

If you find this repository helpful, feel free to cite our publication "Statistical Depth for Ranking and Characterizing Transformer-Based Text Embeddings":
```bibtex
@inproceedings{seegmiller-2023-tte-depth,
  title = "Statistical Depth for Ranking and Characterizing Transformer-Based Text Embeddings",
  author = "Seegmiller, Parker and Preum, Sarah",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month = "12",
  year = "2023",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/2310.15010",
}
```
