#!/usr/bin/env bash
# End-to-end run for the learned-decision-rules experiment.
#
# Prerequisites: one vLLM endpoint per model in ladder.json, and
# RULECHEF_API_KEY set for the rule-synthesis step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/out}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-120}"
EVAL="$HERE/../../src/training/model_eval/mmlu_pro_vllm_eval.py"

declare -A ENDPOINTS=(
  ["Qwen/Qwen2.5-1.5B-Instruct"]="http://localhost:8101/v1"
  ["Qwen/Qwen2.5-7B-Instruct"]="http://localhost:8102/v1"
  ["Qwen/Qwen2.5-14B-Instruct"]="http://localhost:8103/v1"
  ["Qwen/Qwen2.5-32B-Instruct-AWQ"]="http://localhost:8104/v1"
)

mkdir -p "$OUT"

# Same --seed and --samples-per-category across models, so every model sees the
# same questions and the results join on question_id.
for model in "${!ENDPOINTS[@]}"; do
  echo "=== $model"
  python "$EVAL" \
    --endpoint "${ENDPOINTS[$model]}" \
    --models "$model" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --concurrent-requests 16 \
    --seed 42 \
    --output-dir "$OUT/results"
done

CONFIG="${CONFIG:-$HERE/../../config/config.yaml}"

python "$HERE/build_labels.py" \
  --results-dir "$OUT/results" --ladder "$HERE/ladder.json" --out "$OUT/labels.csv"
python "$HERE/make_split.py" --labels "$OUT/labels.csv" --out "$OUT/split.json"

# Needs a GPU and the configured embedding model; skip to leave complexity unset.
python "$HERE/compute_complexity.py" \
  --labels "$OUT/labels.csv" --config "$CONFIG" --out "$OUT/complexity.json"
python "$HERE/compute_signals.py" \
  --labels "$OUT/labels.csv" --config "$CONFIG" \
  --complexity "$OUT/complexity.json" --out "$OUT/signals.csv"
python "$HERE/apply_config_rules.py" \
  --signals "$OUT/signals.csv" --labels "$OUT/labels.csv" --config "$CONFIG" \
  --model-map "$HERE/model_map.json" --ladder "$HERE/ladder.json" \
  --out "$OUT/handwritten.csv"

python "$HERE/learn_rules.py" \
  --labels "$OUT/labels.csv" --signals "$OUT/signals.csv" \
  --split "$OUT/split.json" --out-dir "$OUT/rules"
python "$HERE/evaluate_policies.py" \
  --labels "$OUT/labels.csv" --signals "$OUT/signals.csv" \
  --split "$OUT/split.json" --ladder "$HERE/ladder.json" \
  --handwritten "$OUT/handwritten.csv" \
  --rules "$OUT/rules/routing_rules.json" --out "$OUT/metrics.json"
python "$HERE/sweep_rule_curve.py" \
  --labels "$OUT/labels.csv" --signals "$OUT/signals.csv" \
  --split "$OUT/split.json" --ladder "$HERE/ladder.json" \
  --out "$OUT/rule_curve.json"
python "$HERE/probe_ceiling.py" \
  --labels "$OUT/labels.csv" --split "$OUT/split.json" --ladder "$HERE/ladder.json" \
  --out "$OUT/probe.json"
