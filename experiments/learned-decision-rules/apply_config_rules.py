"""Evaluate the shipped hand-written decision rules against computed signals.

Reads the `routing.decisions` block from `config/config.yaml`, evaluates each
decision's boolean condition tree over the signal vector from
`compute_signals.py`, and resolves ties the way the priority strategy does:
highest priority wins, and no match falls through to `default_model`.

A matched decision usually declares several `modelRefs` and leaves the final
pick to a selection algorithm at runtime. Offline that pick is unknowable, so
two variants are reported:

  declared  — take the first listed modelRef, the decision's stated preference;
  best_case — take the cheapest model in the candidate set that answered
              correctly, an upper bound on any selection algorithm.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-map",
        type=Path,
        required=True,
        help="JSON mapping config model names to result directory names",
    )
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def signal_true(condition: dict, row) -> bool:
    """Evaluate one leaf condition against the signal vector."""
    kind, name = condition.get("type"), condition.get("name")
    if kind == "domain":
        return row.get("domain") == name
    if kind == "complexity":
        # Conditions name a band, as in "needs_reasoning:hard".
        rule, _, band = str(name).partition(":")
        return row.get(f"complexity::{rule}") == (band or "hard")
    column = f"{kind}::{name}"
    if column in row:
        return bool(row[column])
    # Signals this dataset cannot produce (authz, modality, feedback, ...) are
    # absent rather than false; treated as not firing, and counted separately.
    return False


def evaluate_node(node: dict, row) -> bool:
    if "operator" in node and "conditions" in node:
        results = [evaluate_node(child, row) for child in node["conditions"]]
        operator = node["operator"].upper()
        if operator == "AND":
            return all(results)
        if operator == "OR":
            return any(results)
        if operator == "NOT":
            return not any(results)
        return False
    return signal_true(node, row)


def matched_decision(decisions: list, row):
    """Highest-priority decision whose rule tree evaluates true."""
    best = None
    for decision in decisions:
        rules = decision.get("rules")
        if not rules or not evaluate_node(rules, row):
            continue
        if best is None or (decision.get("priority", 0) > best.get("priority", 0)):
            best = decision
    return best


def resolve(decision, model_map: dict, default_model: str) -> list:
    """Candidate result-directory names for a matched decision, in declared order."""
    if decision is None:
        names = [default_model]
    else:
        names = [ref["model"] for ref in decision.get("modelRefs", [])] or [
            default_model
        ]
    resolved = [model_map[name] for name in names if name in model_map]
    return resolved or [model_map[default_model]]


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    routing = config["routing"]
    decisions = routing["decisions"]
    default_model = config["default_model"] if "default_model" in config else None
    if default_model is None:
        default_model = next(
            value
            for key, value in _walk(config)
            if key == "default_model" and isinstance(value, str)
        )

    model_map = json.loads(args.model_map.read_text())
    ladder = json.loads(args.ladder.read_text())
    signals = pd.read_csv(args.signals)
    labels = pd.read_csv(args.labels)
    joined = signals.merge(labels, on="question_id", suffixes=("", "_label"))

    declared, best_case, matched_names = [], [], []
    for _, row in joined.iterrows():
        decision = matched_decision(decisions, row)
        matched_names.append(decision["name"] if decision else "<default>")
        candidates = resolve(decision, model_map, default_model)
        declared.append(candidates[0])
        correct = [
            name
            for name in sorted(candidates, key=lambda n: ladder[n])
            if bool(row[f"correct::{name}"])
        ]
        best_case.append(correct[0] if correct else candidates[0])

    result = pd.DataFrame(
        {
            "question_id": joined["question_id"],
            "decision": matched_names,
            "declared": declared,
            "best_case": best_case,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    print(f"wrote {args.out}")
    print("decisions fired:")
    print(result["decision"].value_counts().to_string())


def _walk(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, prefix)


if __name__ == "__main__":
    main()
