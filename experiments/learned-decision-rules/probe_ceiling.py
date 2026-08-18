"""Estimate how much routing signal the request text carries at all.

Rule induction can only find structure that exists. This probe fits a plain
TF-IDF + logistic-regression classifier on the same split, as a rough upper
bound on what any text-only router can recover. If the probe cannot beat the
existing baselines either, the limit is the data, not the rule format.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from evaluate_policies import (
    frontier_accuracy,
    mixing_frontier,
    policy_category,
    policy_strongest,
    rule_input,
    score,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ladder = json.loads(args.ladder.read_text())
    split = json.loads(args.split.read_text())
    labels = pd.read_csv(args.labels)

    train = labels[labels["question_id"].isin(set(split["train"]))]
    test = labels[labels["question_id"].isin(set(split["test"]))]
    fit_on = train[train["any_correct"]]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit([rule_input(row) for _, row in fit_on.iterrows()], fit_on["label"])
    predictions = list(model.predict([rule_input(row) for _, row in test.iterrows()]))

    strongest = policy_strongest(train, ladder)
    results = {
        "tfidf_logreg": score(test, predictions, ladder),
        "strongest": score(test, [strongest] * len(test), ladder),
        "category": score(
            test,
            [
                policy_category(train, ladder).get(c, strongest)
                for c in test["category"]
            ],
            ladder,
        ),
    }
    # Predicting one model's success is an easier target than the 4-way choice.
    # If even that is near chance, the request text carries no routing signal.
    auc = {}
    for name in ladder:
        binary = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        binary.fit(
            [rule_input(row) for _, row in train.iterrows()], train[f"correct::{name}"]
        )
        scores = binary.predict_proba([rule_input(row) for _, row in test.iterrows()])[
            :, 1
        ]
        auc[name] = roc_auc_score(test[f"correct::{name}"], scores)
    results["_success_auc"] = auc

    hull = mixing_frontier(test, ladder)
    for name, metrics in results.items():
        if name.startswith("_"):
            continue
        metrics["gain_over_frontier"] = metrics["accuracy"] - frontier_accuracy(
            hull, metrics["mean_cost"]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for name, metrics in results.items():
        if name.startswith("_"):
            continue
        print(
            f"{name:14} acc={metrics['accuracy']:.3f} "
            f"cost={metrics['mean_cost']:.2f} "
            f"gain_over_frontier={metrics['gain_over_frontier']:+.3f}"
        )
    print("per-model success AUC from request text:")
    for name, value in auc.items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
