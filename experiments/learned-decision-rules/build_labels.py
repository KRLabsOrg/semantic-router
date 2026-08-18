"""Build per-question routing labels from MMLU-Pro evaluation outputs.

Reads one `detailed_results.csv` per model (as written by
`src/training/model_eval/mmlu_pro_vllm_eval.py`), joins them on `question_id`,
and emits one row per question with each model's correctness and latency plus
the routing label: the cheapest model that answered correctly.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory holding one <model>_<cot|direct>/detailed_results.csv per model",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        required=True,
        help="JSON file mapping result directory name -> relative cost (ascending)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    return parser.parse_args()


def load_model_results(results_dir: Path, dir_name: str) -> pd.DataFrame:
    """Load one model's detailed results, keeping only the join-relevant columns."""
    path = results_dir / dir_name / "detailed_results.csv"
    frame = pd.read_csv(path)
    required = {
        "question_id",
        "question",
        "category",
        "is_correct",
        "response_time",
        "success",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[sorted(required)]


def join_models(results_dir: Path, ladder: dict) -> pd.DataFrame:
    """Join every model's results on question_id into one wide frame."""
    joined = None
    for dir_name in ladder:
        frame = load_model_results(results_dir, dir_name).rename(
            columns={
                "is_correct": f"correct::{dir_name}",
                "response_time": f"latency::{dir_name}",
                "success": f"success::{dir_name}",
            }
        )
        if joined is None:
            joined = frame
        else:
            joined = joined.merge(
                frame.drop(columns=["category", "question"]),
                on="question_id",
                how="inner",
            )
    if joined is None or joined.empty:
        raise ValueError("no overlapping questions across the given models")
    return joined


def add_labels(joined: pd.DataFrame, ladder: dict) -> pd.DataFrame:
    """Label each question with the cheapest model that answered it correctly."""
    order = sorted(ladder, key=lambda name: ladder[name])

    def cheapest_correct(row):
        for name in order:
            if bool(row[f"correct::{name}"]):
                return name
        return None

    joined["label"] = joined.apply(cheapest_correct, axis=1)
    joined["any_correct"] = joined["label"].notna()
    joined["label_cost"] = joined["label"].map(ladder)
    return joined


def main():
    args = parse_args()
    ladder = json.loads(args.ladder.read_text())
    joined = add_labels(join_models(args.results_dir, ladder), ladder)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out, index=False)

    labelled = joined[joined["any_correct"]]
    print(f"questions joined: {len(joined)}")
    print(f"answered by at least one model: {len(labelled)}")
    print("label distribution:")
    print(labelled["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
