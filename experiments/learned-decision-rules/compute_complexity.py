"""Compute the complexity signal margin for each question.

Ports the scoring in `complexity_rule_scoring.go` and `prototype_scoring.go`:
each candidate phrase from the config is one prototype, a bank scores as
`best_weight * best + (1 - best_weight) * mean(top_m)`, and the rule's signal is
the hard bank's score minus the easy bank's score. Difficulty banding against
the configured threshold happens in `compute_signals.py`.

Defaults from `PrototypeScoringConfig.WithDefaults`: best_weight 0.75, top_m 2.
Text-only; the image banks in the Go code have no counterpart here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

BEST_WEIGHT = 0.75
TOP_M = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
        help="Embedding model, matching the configured mmbert_model_path",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def bank_score(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Port of prototypeBank.score, vectorised over queries."""
    similarities = query @ prototypes.T
    ordered = -np.sort(-similarities, axis=1)
    best = ordered[:, 0]
    top_m = min(TOP_M, ordered.shape[1])
    support = ordered[:, :top_m].mean(axis=1)
    return BEST_WEIGHT * best + (1 - BEST_WEIGHT) * support


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    rules = config["routing"]["signals"]["complexity"]
    labels = pd.read_csv(args.labels)

    model = SentenceTransformer(args.model, trust_remote_code=True)
    questions = model.encode(
        [str(text) for text in labels["question"]],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    output = {}
    for rule in rules:
        hard = model.encode(rule["hard"]["candidates"], normalize_embeddings=True)
        easy = model.encode(rule["easy"]["candidates"], normalize_embeddings=True)
        margin = bank_score(questions, hard) - bank_score(questions, easy)
        output[rule["name"]] = margin
        print(
            f"{rule['name']}: margin min={margin.min():.3f} "
            f"median={np.median(margin):.3f} max={margin.max():.3f} "
            f"threshold=±{rule['threshold']}"
        )

    # One rule per run is the configured case; key the output by question_id.
    primary = rules[0]["name"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                str(qid): float(value)
                for qid, value in zip(
                    labels["question_id"], output[primary], strict=True
                )
            }
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
