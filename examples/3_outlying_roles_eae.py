"""Example 3 — Identifying outlying roles in structured information extraction.

Demonstrates the use-case from:
  Gatto et al., "Document-Level Event-Argument Data Augmentation for
  Challenging Role Types," ACL 2025.

In few-shot cross-domain Event Argument Extraction (EAE), some target-domain
event roles are semantically very different from anything seen in the source
domain.  These "challenging roles" require targeted data augmentation to model
well.

TTE depth flags them automatically: embed all role-type names, compute the
depth of each TARGET role w.r.t. the SOURCE role distribution, and rank by
depth (ascending).  Low-depth roles are the challenging ones — they are the
least represented by the source domain's role vocabulary.

Requires:  pip install tte_depth sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from tte_depth import StatDepth

model = SentenceTransformer("all-MiniLM-L6-v2")

# Source-domain roles (e.g., news events: crashes, crimes, celebrity news)
source_roles = [
    "victim",
    "perpetrator",
    "location",
    "date",
    "vehicle",
    "weapon",
    "injury",
    "witness",
    "officer",
    "suspect",
]

# Target-domain roles (e.g., natural disaster events)
target_roles = [
    "affected region",           # somewhat similar to "location"
    "number of casualties",      # somewhat similar to "victim"
    "rescue team",               # somewhat similar to "officer"
    "magnitude",                 # domain-specific, unusual
    "state of the volcano",      # highly specific and unusual
    "early warning system",      # technical, rarely seen in news
    "duration of the event",     # abstract temporal concept
    "geological fault line",     # highly technical
]

F = model.encode(source_roles)
G = model.encode(target_roles)

d = StatDepth()
depth_F, depth_G = d.depths_paired(F, G)

# Rank target roles from most outlying to most represented
ranked = sorted(zip(target_roles, depth_G), key=lambda x: x[1])

print("Target-role depth w.r.t. source-domain distribution")
print("(lower depth = more semantically outlying = harder to generalise)")
print("=" * 65)
for role, score in ranked:
    flag = "  ← challenging role" if score < np.median(depth_G) else ""
    print(f"  {score:.4f}  {role}{flag}")

print()
print(f"Source-domain median depth : {np.median(depth_F):.4f}")
print(f"Target-domain median depth : {np.median(depth_G):.4f}")
print()
print("Roles below median depth are the best candidates for targeted")
print("data augmentation (e.g., MLG or S2T methods from Gatto et al.).")
