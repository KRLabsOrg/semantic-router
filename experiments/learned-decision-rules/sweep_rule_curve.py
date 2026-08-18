"""Trace the best cost/accuracy curve any rule set over the signals can reach.

`evaluate_policies.py` scores one lookup table, fitted to maximise accuracy. That
is a single point. A rule set can be tuned to any cost preference, so the honest
question is whether the whole achievable curve clears the mixing frontier — in
particular at the cost where the shipped rules currently sit.

For each cost weight lambda, choose per signal state the model maximising
`accuracy - lambda * cost` on the training split, then score that policy on the
held-out split. The resulting curve is the upper bound for every rule set,
hand-written or learned, at every cost point.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_policies import (
    frontier_accuracy,
    mixing_frontier,
    policy_strongest,
    rule_input,
    score,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.34,
        help="Share of the training split held out to choose lambda",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Signal states with fewer training rows fall back to the global choice",
    )
    return parser.parse_args()


def lookup_for_lambda(train: pd.DataFrame, ladder: dict, lam: float, min_support: int):
    """Per signal state, the model maximising accuracy minus lambda times cost."""
    states = train.assign(_state=[rule_input(row) for _, row in train.iterrows()])
    global_choice = max(
        ladder, key=lambda name: states[f"correct::{name}"].mean() - lam * ladder[name]
    )
    table = {}
    for state, group in states.groupby("_state", sort=False):
        if len(group) < min_support:
            continue
        table[state] = max(
            ladder,
            key=lambda name: group[f"correct::{name}"].mean() - lam * ladder[name],
        )
    return table, global_choice


def main():
    args = parse_args()
    ladder = json.loads(args.ladder.read_text())
    split = json.loads(args.split.read_text())
    labels = pd.read_csv(args.labels).merge(
        pd.read_csv(args.signals), on="question_id", suffixes=("", "_signal")
    )
    train_all = labels[labels["question_id"].isin(set(split["train"]))]
    # Lambda is a tuned parameter, so it has to be chosen on data the reported
    # number does not come from. Otherwise picking the best of many lambdas on
    # the test split manufactures a gain.
    cut = int(len(train_all) * (1 - args.val_fraction))
    train = train_all.iloc[:cut]
    val = train_all.iloc[cut:]
    test = labels[labels["question_id"].isin(set(split["test"]))]

    hull = mixing_frontier(test, ladder)
    strongest = policy_strongest(train, ladder)
    test_states = [rule_input(row) for _, row in test.iterrows()]

    lambdas = [
        0.0,
        0.002,
        0.005,
        0.008,
        0.012,
        0.014,
        0.016,
        0.018,
        0.020,
        0.022,
        0.024,
        0.026,
        0.028,
        0.030,
        0.035,
        0.05,
        0.08,
        0.15,
    ]
    curve = []
    for lam in lambdas:
        table, global_choice = lookup_for_lambda(train, ladder, lam, args.min_support)
        choices = [table.get(state, global_choice) for state in test_states]
        metrics = score(test, choices, ladder)
        reference = frontier_accuracy(hull, metrics["mean_cost"])
        curve.append(
            {
                "lambda": lam,
                "accuracy": metrics["accuracy"],
                "mean_cost": metrics["mean_cost"],
                "frontier_accuracy": reference,
                "gain_over_frontier": metrics["accuracy"] - reference,
                "distinct_choices": len(set(choices)),
                "fallback_share": sum(1 for state in test_states if state not in table)
                / len(test_states),
            }
        )

    # Choose lambda on validation, then report that one lambda on test.
    val_hull = mixing_frontier(val, ladder)
    val_states = [rule_input(row) for _, row in val.iterrows()]
    val_curve = []
    for lam in lambdas:
        table, global_choice = lookup_for_lambda(train, ladder, lam, args.min_support)
        choices = [table.get(state, global_choice) for state in val_states]
        metrics = score(val, choices, ladder)
        val_curve.append(
            {
                "lambda": lam,
                "gain_over_frontier": metrics["accuracy"]
                - frontier_accuracy(val_hull, metrics["mean_cost"]),
            }
        )
    chosen = max(val_curve, key=lambda point: point["gain_over_frontier"])["lambda"]
    best = next(point for point in curve if point["lambda"] == chosen)
    best["selected_on"] = "validation"

    # A 672-question test split leaves roughly two points of standard error on
    # accuracy, so a small positive gain needs a confidence interval before it
    # can be called a gain at all.
    table, global_choice = lookup_for_lambda(
        train, ladder, best["lambda"], args.min_support
    )
    choices = [table.get(state, global_choice) for state in test_states]
    correct = [
        bool(row[f"correct::{choice}"])
        for (_, row), choice in zip(test.iterrows(), choices, strict=True)
    ]
    costs = [ladder[choice] for choice in choices]
    rng = __import__("random").Random(0)
    indices = range(len(correct))
    gains = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(indices) for _ in indices]
        accuracy = sum(correct[i] for i in sample) / len(sample)
        cost = sum(costs[i] for i in sample) / len(sample)
        resampled_hull = mixing_frontier(test.iloc[list(sample)], ladder)
        gains.append(accuracy - frontier_accuracy(resampled_hull, cost))
    gains.sort()
    best["gain_ci95"] = [
        gains[int(0.025 * len(gains))],
        gains[int(0.975 * len(gains))],
    ]
    output = {
        "curve": curve,
        "val_curve": val_curve,
        "selected_lambda": chosen,
        "best": best,
        "frontier": hull,
        "strongest": strongest,
        "oracle_curve_note": "curve is fitted on train, scored on test",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))

    print(f"{'lambda':>8} {'acc':>7} {'cost':>7} {'frontier':>9} {'gain':>8}")
    for point in curve:
        print(
            f"{point['lambda']:8.3f} {point['accuracy']:7.3f} {point['mean_cost']:7.2f} "
            f"{point['frontier_accuracy']:9.3f} {point['gain_over_frontier']:+8.3f}"
        )
    low, high = best["gain_ci95"]
    print(
        f"\nlambda {best['lambda']} chosen on validation -> test gain "
        f"{best['gain_over_frontier']:+.3f} at cost {best['mean_cost']:.2f}, "
        f"95% CI [{low:+.3f}, {high:+.3f}]"
    )


if __name__ == "__main__":
    main()
