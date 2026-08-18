"""Split labelled questions into train and test sets by question_id."""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = pd.read_csv(args.labels)
    ids = sorted(labels["question_id"].unique().tolist())
    random.Random(args.seed).shuffle(ids)

    cut = int(len(ids) * args.train_fraction)
    split = {"train": ids[:cut], "test": ids[cut:], "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split))
    print(f"train {len(split['train'])} / test {len(split['test'])}")


if __name__ == "__main__":
    main()
