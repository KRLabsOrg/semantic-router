"""Compare routing policies on held-out MMLU-Pro questions.

Policies:
  strongest — always route to the most accurate model (cost ceiling);
  category  — best model per category on the training split, which is what
              `src/training/model_eval/result_to_config.py` derives today;
  induced   — rules learned by `learn_rules.py`, executed without an LLM.

Every policy is scored on the same held-out questions using the recorded
per-model correctness, so no model is queried again here.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument(
        "--rules",
        type=Path,
        help="Learned ruleset JSON; omit to skip the induced policy",
    )
    parser.add_argument(
        "--rule-fallback",
        choices=["strongest", "cheapest"],
        default="strongest",
        help="Model to use when no rule matches",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Where to write the metrics JSON"
    )
    return parser.parse_args()


def score(frame: pd.DataFrame, choices: list, ladder: dict) -> dict:
    """Score a routing decision per question against the recorded outcomes."""
    correct = [
        bool(row[f"correct::{choice}"])
        for (_, row), choice in zip(frame.iterrows(), choices, strict=True)
    ]
    cost = [ladder[choice] for choice in choices]
    ceiling = max(ladder.values())
    cheaper_no_loss = [c < ceiling and ok for c, ok in zip(cost, correct, strict=True)]
    return {
        "accuracy": sum(correct) / len(correct),
        "mean_cost": sum(cost) / len(cost),
        "routed_cheaper_no_loss": sum(cheaper_no_loss) / len(cheaper_no_loss),
        "label_agreement": sum(
            choice == label
            for choice, label in zip(choices, frame["label"], strict=True)
        )
        / len(choices),
        "model_share": pd.Series(choices).value_counts(normalize=True).to_dict(),
    }


def mixing_frontier(test: pd.DataFrame, ladder: dict) -> list:
    """Upper convex hull of the always-one-model policies.

    Any point on this hull is reachable by randomly mixing two models at a fixed
    ratio, with no request inspection at all. A router only earns its keep by
    landing above it.
    """
    points = sorted((ladder[name], test[f"correct::{name}"].mean()) for name in ladder)
    hull = []
    for point in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            x3, y3 = point
            # Drop the middle point if it sits below the chord around it.
            if (y2 - y1) * (x3 - x1) <= (y3 - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        if hull and point[1] <= hull[-1][1]:
            continue
        hull.append(point)
    return hull


def frontier_accuracy(hull: list, cost: float) -> float:
    """Accuracy the mixing frontier reaches at the given cost."""
    if cost <= hull[0][0]:
        return hull[0][1]
    for (x1, y1), (x2, y2) in zip(hull, hull[1:], strict=False):
        if cost <= x2:
            return y1 + (y2 - y1) * (cost - x1) / (x2 - x1)
    return hull[-1][1]


def policy_strongest(train: pd.DataFrame, ladder: dict) -> str:
    """The single most accurate model on the training split."""
    return max(ladder, key=lambda name: train[f"correct::{name}"].mean())


def policy_category(train: pd.DataFrame, ladder: dict) -> dict:
    """Best model per category on the training split, ties broken by cost."""
    best = {}
    for category, group in train.groupby("category"):
        best[category] = max(
            ladder,
            key=lambda name: (group[f"correct::{name}"].mean(), -ladder[name]),
        )
    return best


def rule_input(row) -> str:
    """The text a rule sees: the request plus the domain signal the router already has."""
    return f"[{row['category']}] {row['question']}"


def run_rules(rules_path: Path, questions: list, ladder: dict, fallback: str) -> tuple:
    """Execute the learned ruleset over the test questions, no LLM involved."""
    from rulechef import RuleChef, Task, TaskType

    task = Task(
        name="Model routing",
        description="Choose the cheapest model expected to answer correctly",
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )
    chef = RuleChef(task, dataset_name="eval_only", storage_path=str(rules_path.parent))
    chef.load_rules(rules_path)

    choices, unmatched = [], 0
    start = time.perf_counter()
    for question in questions:
        label = (chef.extract({"text": question}) or {}).get("label")
        if label not in ladder:
            label = fallback
            unmatched += 1
        choices.append(label)
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(questions), 1)
    return choices, {
        "unmatched_share": unmatched / len(questions),
        "ms_per_question": elapsed_ms,
    }


def main():
    args = parse_args()
    ladder = json.loads(args.ladder.read_text())
    split = json.loads(args.split.read_text())
    labels = pd.read_csv(args.labels)

    train = labels[labels["question_id"].isin(set(split["train"]))]
    test = labels[labels["question_id"].isin(set(split["test"]))]
    print(f"train {len(train)} / test {len(test)}")

    strongest = policy_strongest(train, ladder)
    per_category = policy_category(train, ladder)
    cheapest = min(ladder, key=lambda name: ladder[name])

    results = {
        "strongest": score(test, [strongest] * len(test), ladder),
        "category": score(
            test,
            [per_category.get(c, strongest) for c in test["category"]],
            ladder,
        ),
        "oracle": score(
            test,
            [label if isinstance(label, str) else strongest for label in test["label"]],
            ladder,
        ),
    }
    results["_policies"] = {"strongest": strongest, "category": per_category}

    if args.rules:
        fallback = strongest if args.rule_fallback == "strongest" else cheapest
        choices, rule_stats = run_rules(
            args.rules,
            [rule_input(row) for _, row in test.iterrows()],
            ladder,
            fallback=fallback,
        )
        rule_stats["fallback"] = fallback
        results["induced"] = score(test, choices, ladder) | rule_stats

    hull = mixing_frontier(test, ladder)
    for name, metrics in results.items():
        if name.startswith("_"):
            continue
        reference = frontier_accuracy(hull, metrics["mean_cost"])
        metrics["frontier_accuracy"] = reference
        metrics["gain_over_frontier"] = metrics["accuracy"] - reference
    results["_frontier"] = hull

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {k: v for k, v in results.items() if not k.startswith("_")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
