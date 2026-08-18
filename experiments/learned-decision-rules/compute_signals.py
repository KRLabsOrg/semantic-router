"""Compute the router's signal vector for each question.

Decision rules match on signals, not on request text, so any experiment about
learning rules has to work in the same vocabulary. This module ports the
signal extractors from `src/semantic-router/pkg/classification` and reads their
definitions straight out of `config/config.yaml`, so the features here are the
ones the deployed router would actually see.

Parity notes, per signal family:

- `structure`  exact port of `structure_classifier.go` and `structure_text.go`,
               including the multilingual unit count used as the density
               denominator and the word-boundary rule for keyword counting.
- `domain`     read from the configured `mmlu_categories` mapping. Categories no
               configured domain claims produce no domain signal, which is what
               the router does.
- `context`    token estimate against the configured `min_tokens`. Approximate,
               but the thresholds are 32K, far above anything in this dataset.
- `keyword`    approximations. The router delegates BM25, n-gram and fuzzy
               matching to the Rust NLP binding; these use a standard BM25, a
               a windowed character-trigram Jaccard, and a partial-ratio score
               instead. Firing rates are reported so the effect is visible.
- `complexity` exact port of the scoring in `complexity_rule_scoring.go` and
               `prototype_scoring.go`, over embeddings supplied by
               `compute_complexity.py`.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

CJK_RANGES = (
    ("一", "鿿"),
    ("぀", "ゟ"),
    ("゠", "ヿ"),
    ("가", "힯"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--complexity",
        type=Path,
        help="JSON from compute_complexity.py mapping question_id to margin",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def is_cjk(char: str) -> bool:
    return any(low <= char <= high for low, high in CJK_RANGES)


def text_unit_count(text: str) -> int:
    """Port of multilingualTextUnitCount: CJK chars count singly, words once."""
    count, in_word = 0, False
    for char in text:
        if is_cjk(char):
            count += 1
            in_word = False
        elif unicodedata.category(char)[0] in ("L", "N"):
            if not in_word:
                count += 1
                in_word = True
        else:
            in_word = False
    return count


def boundary_ok(text: str, start: int, end: int) -> bool:
    """Port of keywordBoundaryMatch: reject matches glued to letters or digits."""

    def blocking(char: str) -> bool:
        return unicodedata.category(char)[0] in ("L", "N") and not is_cjk(char)

    if start > 0 and blocking(text[start - 1]):
        return False
    if end < len(text) and blocking(text[end]):
        return False
    return True


def keyword_occurrences(text: str, keywords: list, case_sensitive: bool) -> int:
    """Port of keywordOccurrenceCount."""
    candidate = text if case_sensitive else text.lower()
    total = 0
    for keyword in keywords:
        needle = keyword if case_sensitive else keyword.lower()
        if not needle:
            continue
        require_boundary = not any(is_cjk(char) for char in needle)
        start = 0
        while True:
            index = candidate.find(needle, start)
            if index < 0:
                break
            if not require_boundary or boundary_ok(
                candidate, index, index + len(needle)
            ):
                total += 1
            start = index + len(needle)
    return total


def sequence_matched(source: dict, text: str) -> bool:
    """Port of structureSequenceMatched: markers must appear in order."""
    case_sensitive = source.get("case_sensitive", False)
    candidate = text if case_sensitive else text.lower()
    for sequence in source.get("sequences", []):
        position, matched = 0, True
        for marker in sequence:
            needle = marker if case_sensitive else marker.lower()
            index = candidate.find(needle, position)
            if index < 0:
                matched = False
                break
            position = index + len(needle)
        if matched:
            return True
    return False


def source_count(source: dict, text: str) -> int:
    """Port of structureSourceCount."""
    kind = source.get("type", "").lower()
    if kind == "regex":
        pattern = source["pattern"]
        flags = 0 if source.get("case_sensitive") else re.IGNORECASE
        return len(re.findall(pattern, text, flags))
    if kind == "keyword_set":
        return keyword_occurrences(
            text, source.get("keywords", []), source.get("case_sensitive", False)
        )
    if kind == "sequence":
        return 1 if sequence_matched(source, text) else 0
    if kind == "text_bytes":
        return len(text.encode("utf-8"))
    return 0


def feature_value(rule: dict, text: str) -> float:
    """Port of extractFeatureValue."""
    feature = rule["feature"]
    kind = feature["type"].lower()
    source = feature["source"]
    if kind == "exists":
        return 1.0 if source_count(source, text) > 0 else 0.0
    if kind == "count":
        return float(source_count(source, text))
    if kind == "density":
        denominator = text_unit_count(text)
        return source_count(source, text) / denominator if denominator else 0.0
    if kind == "sequence":
        return 1.0 if sequence_matched(source, text) else 0.0
    return 0.0


def predicate_matches(value: float, predicate) -> bool:
    """Port of predicateMatches; a missing predicate means 'value is positive'."""
    if not predicate:
        return value > 0
    checks = (
        ("gt", lambda v, t: v > t),
        ("gte", lambda v, t: v >= t),
        ("lt", lambda v, t: v < t),
        ("lte", lambda v, t: v <= t),
    )
    for key, test in checks:
        if predicate.get(key) is not None and not test(value, predicate[key]):
            return False
    return True


def structure_signals(rules: list, text: str) -> dict:
    return {
        f"structure::{rule['name']}": predicate_matches(
            feature_value(rule, text), rule.get("predicate")
        )
        for rule in rules
    }


def bm25_score(text: str, keywords: list) -> float:
    """Approximate BM25 of the keyword set against one document."""
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    k1, b, avgdl = 1.2, 0.75, 120.0
    length = len(tokens)
    score = 0.0
    for keyword in keywords:
        needle = keyword.lower()
        frequency = (
            sum(1 for token in tokens if token == needle)
            if " " not in needle
            else text.lower().count(needle)
        )
        if frequency:
            score += (frequency * (k1 + 1)) / (
                frequency + k1 * (1 - b + b * length / avgdl)
            )
    return score / max(len(keywords), 1)


def trigrams(text: str) -> set:
    lowered = re.sub(r"\s+", " ", text.lower())
    return {lowered[i : i + 3] for i in range(max(len(lowered) - 2, 0))}


def ngram_score(text: str, keywords: list) -> float:
    """Approximate n-gram match: best character-trigram Jaccard against a window.

    Containment alone is far too loose: short keywords share trigrams with
    ordinary prose, so "urgent" scores highly on academic text. Jaccard against
    the best same-length window keeps a match local, as an n-gram matcher does.
    """
    lowered = re.sub(r"\s+", " ", text.lower())
    best = 0.0
    for keyword in keywords:
        needle = trigrams(keyword)
        if not needle:
            continue
        width = len(keyword)
        for start in range(0, max(len(lowered) - width, 0) + 1):
            window = trigrams(lowered[start : start + width])
            if not window:
                continue
            best = max(best, len(needle & window) / len(needle | window))
            if best >= 1.0:
                return best
    return best


def fuzzy_score(text: str, keywords: list) -> float:
    """Approximate fuzzy match: best partial ratio, on a 0-100 scale."""
    from difflib import SequenceMatcher

    lowered = text.lower()
    best = 0.0
    for keyword in keywords:
        needle = keyword.lower()
        window = len(needle)
        for start in range(0, max(len(lowered) - window, 0) + 1):
            ratio = SequenceMatcher(
                None, needle, lowered[start : start + window]
            ).ratio()
            best = max(best, ratio * 100)
            if best >= 100:
                return best
    return best


def keyword_signals(rules: list, text: str) -> dict:
    signals = {}
    for rule in rules:
        method = rule.get("method", "").lower()
        keywords = rule.get("keywords", [])
        if method == "bm25":
            matched = bm25_score(text, keywords) >= rule.get("bm25_threshold", 0.1)
        elif method == "ngram":
            matched = ngram_score(text, keywords) >= rule.get("ngram_threshold", 0.4)
        elif method == "fuzzy":
            matched = fuzzy_score(text, keywords) >= rule.get("fuzzy_threshold", 80)
        else:
            matched = any(
                keyword_occurrences(text, [keyword], False) for keyword in keywords
            )
        signals[f"keyword::{rule['name']}"] = matched
    return signals


def parse_token_limit(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    multiplier = {"K": 1_000, "M": 1_000_000}.get(text[-1:], 1)
    return int(float(text.rstrip("KM")) * multiplier)


def context_signals(rules: list, text: str) -> dict:
    # Token count is estimated; the configured thresholds are in the tens of
    # thousands, so the estimate never decides a borderline case here.
    tokens = len(text) / 4
    return {
        f"context::{rule['name']}": tokens >= parse_token_limit(rule["min_tokens"])
        for rule in rules
    }


def domain_signal(domains: list, category: str) -> str:
    for domain in domains:
        if category in (domain.get("mmlu_categories") or []):
            return domain["name"]
    return ""


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    signals_config = config["routing"]["signals"]
    labels = pd.read_csv(args.labels)

    margins = {}
    if args.complexity:
        margins = {
            int(key): value
            for key, value in json.loads(args.complexity.read_text()).items()
        }

    complexity_rules = signals_config.get("complexity", [])
    rows = []
    for _, row in labels.iterrows():
        text = str(row["question"])
        record = {"question_id": row["question_id"], "category": row["category"]}
        record["domain"] = domain_signal(
            signals_config.get("domains", []), row["category"]
        )
        record.update(structure_signals(signals_config.get("structure", []), text))
        record.update(keyword_signals(signals_config.get("keywords", []), text))
        record.update(context_signals(signals_config.get("context", []), text))

        for rule in complexity_rules:
            margin = margins.get(int(row["question_id"]))
            threshold = float(rule["threshold"])
            if margin is None:
                difficulty = "unknown"
            elif margin > threshold:
                difficulty = "hard"
            elif margin < -threshold:
                difficulty = "easy"
            else:
                difficulty = "medium"
            # The composer is a filter: the difficulty only becomes a signal when
            # its conditions also hold.
            composer = rule.get("composer")
            if composer and difficulty != "unknown":
                if not evaluate_composer(composer, record):
                    difficulty = "filtered"
            record[f"complexity::{rule['name']}"] = difficulty
            record[f"complexity_margin::{rule['name']}"] = margin
        rows.append(record)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"wrote {args.out} ({len(frame)} rows)")
    print("signal firing rates:")
    for column in frame.columns:
        if column.startswith(("structure::", "keyword::", "context::")):
            print(f"  {column}: {frame[column].mean():.3f}")
        elif column.startswith("complexity::") or column == "domain":
            counts = frame[column].value_counts().to_dict()
            print(f"  {column}: {counts}")


def evaluate_composer(node: dict, record: dict) -> bool:
    """Evaluate a composer tree against already-computed signals."""
    if "operator" in node:
        results = [evaluate_composer(child, record) for child in node["conditions"]]
        operator = node["operator"].upper()
        if operator == "AND":
            return all(results)
        if operator == "OR":
            return any(results)
        if operator == "NOT":
            return not any(results)
    kind, name = node.get("type"), node.get("name")
    if kind == "domain":
        return record.get("domain") == name
    return bool(record.get(f"{kind}::{name}", False))


if __name__ == "__main__":
    main()
