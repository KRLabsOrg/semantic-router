"""Bound the embedding half of a projection score.

difficulty_score reads 12 embedding signals, each a cosine similarity to a short
candidate list. Those 12 numbers are a lossy projection of the request
embedding, so whatever a classifier can extract from the full embedding vector
is an upper bound on what any weighting of the 12 can reach.
"""

import json
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

labels = pd.read_csv(sys.argv[1])
split = json.loads(open(sys.argv[2]).read())
model = SentenceTransformer(
    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka", trust_remote_code=True
)
emb = model.encode(
    [str(t) for t in labels["question"]], normalize_embeddings=True, batch_size=32
)
np.save(sys.argv[3], emb)

train_mask = labels["question_id"].isin(set(split["train"])).values
test_mask = labels["question_id"].isin(set(split["test"])).values
cols = [c for c in labels.columns if c.startswith("correct::")]

print("Predicting each model's success from the full request embedding:")
for col in cols:
    y = labels[col].astype(int).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(emb[train_mask], y[train_mask])
    scores = clf.predict_proba(emb[test_mask])[:, 1]
    print(f"  {col.split('::')[1]:38} AUC {roc_auc_score(y[test_mask], scores):.3f}")

# The routing-relevant target: does this request need more than the cheap model?
strong = (~labels["correct::Qwen_Qwen2.5-7B-Instruct_direct"].astype(bool)) & (
    labels["correct::Qwen_Qwen2.5-32B-Instruct-AWQ_direct"].astype(bool)
)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
clf.fit(emb[train_mask], strong.values[train_mask])
scores = clf.predict_proba(emb[test_mask])[:, 1]
print(
    f"\n'needs escalation beyond 7B' AUC "
    f"{roc_auc_score(strong.values[test_mask], scores):.3f} "
    f"(base rate {strong.mean():.1%})"
)
