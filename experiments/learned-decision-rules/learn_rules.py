"""Learn routing rules from labelled MMLU-Pro questions with RuleChef.

Input is the CSV produced by `build_labels.py`. Each labelled question becomes a
classification example mapping the question text to the cheapest model that
answered it correctly. The learned rules are saved as JSON and run without an
LLM at inference time.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from rulechef import RuleChef, RuleFormat, Task, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, required=True, help="build_labels.py output CSV"
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="JSON with train/test question_id lists",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for the learned ruleset"
    )
    parser.add_argument(
        "--name", type=str, default="routing_rules", help="Ruleset name"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-oss-120b", help="Synthesis model"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("RULECHEF_BASE_URL", "https://inference.baseten.co/v1"),
    )
    parser.add_argument(
        "--max-train", type=int, default=2000, help="Cap on training examples"
    )
    parser.add_argument("--refinement-iterations", type=int, default=5)
    return parser.parse_args()


def build_task() -> Task:
    return Task(
        name="Model routing",
        description=(
            "Given a request reaching an LLM router, prefixed with its domain "
            "signal in square brackets, choose the cheapest model expected to "
            "answer it correctly. Larger models cost more, so route up only when "
            "the request needs it. Requests no rule matches fall back to the "
            "strongest model, so a rule is only worth writing when it can safely "
            "route below that."
        ),
        input_schema={"text": "str"},
        output_schema={"label": "str"},
        type=TaskType.CLASSIFICATION,
        text_field="text",
    )


def main():
    args = parse_args()
    api_key = os.environ.get("RULECHEF_API_KEY") or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise SystemExit("set RULECHEF_API_KEY (or BASETEN_API_KEY)")

    labels = pd.read_csv(args.labels)
    split = json.loads(args.split.read_text())
    train_ids = set(split["train"])

    train = labels[labels["any_correct"] & labels["question_id"].isin(train_ids)]
    train = train.head(args.max_train)
    print(f"training on {len(train)} examples")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=api_key)
    chef = RuleChef(
        build_task(),
        client,
        dataset_name=args.name,
        storage_path=str(args.out_dir),
        model=args.model,
        allowed_formats=[RuleFormat.REGEX, RuleFormat.CODE],
        synthesis_strategy="per_class",
    )
    for _, row in train.iterrows():
        text = f"[{row['category']}] {row['question']}"
        chef.add_example({"text": text}, {"label": row["label"]})

    chef.learn_rules(max_refinement_iterations=args.refinement_iterations)

    print(f"wrote {args.out_dir / (args.name + '.json')}")
    for rule in chef.get_rules_summary():
        print(f"  [{rule['priority']}] {rule['name']}: {rule['description']}")
    chef.evaluate()


if __name__ == "__main__":
    main()
