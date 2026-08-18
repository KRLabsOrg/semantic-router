# Handoff: learned routing policy over Semantic Router signals

Working notes for whoever continues this. It records what was asked, what was
measured, which claims were retracted after verification, the exact commands
used, and the experiment design that should replace the one here.

Branch: `proposal/learned-decision-rules` on the KRLabsOrg fork.
Compute host: `maszat` (4x NVIDIA A40, 46 GB each).

---

## 1. The question

Semantic Router chooses a model in four layers: signals -> boolean decision
rules -> candidate models -> selection algorithm. The decision rules are
hand-authored. Can they be learned or optimised from outcome data — with
RuleChef, or anything else — while keeping the artifact symbolic?

The white paper's own future work (arXiv:2603.04444 §17.1) states the goal:
"adaptive decision synthesis, where the system learns decision formulas from
routing history." Issue #1512 tracks it and is unclaimed.

---

## 2. Findings

### 2.1 Two controls that made results interpretable

**Mixing frontier.** The upper convex hull of the always-one-model policies.
Every point on it is reachable by randomly mixing two models at a fixed ratio,
inspecting nothing. Accuracy and cost reported side by side prove nothing,
because spending more scores more; a router earns its keep only above this hull.
`evaluate_policies.py` computes it and reports `gain_over_frontier`.

Their own routing benchmark (`vllm-semantic-router-bench compare`) does not have
this control — it compares the router against a single direct backend.

**Signal lookup table.** One routing choice per distinct signal vector. A rule
set is a function of the signal vector, so a lookup table is the most expressive
rule set that can exist — strictly more expressive than any nested AND/OR/NOT
the DSL can write. Whatever it scores bounds every rule format and every
induction method.

### 2.2 Measured results

MMLU-Pro, 120 questions per category, 1680 total, identical sample for all four
models. 1267 answered by at least one. Split 60/40 by question, seed 42.
Ladder and single-model accuracy:

| Model | Relative cost | Accuracy |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 1.0 | 0.274 |
| Qwen2.5-7B-Instruct | 4.7 | 0.434 |
| Qwen2.5-14B-Instruct | 9.3 | 0.529 |
| Qwen2.5-32B-Instruct-AWQ | 21.3 | 0.610 |

Policies on the held-out split:

| Policy | Accuracy | Mean cost | Gain over frontier | Deployable? |
| --- | --- | --- | --- | --- |
| strongest (always 32B) | 0.613 | 21.30 | +0.000 | yes |
| category (per-category argmax) | 0.603 | 18.66 | +0.013 | yes |
| shipped showcase rules, declared model | 0.433 | 4.70 | +0.000 | yes |
| shipped showcase rules, best case in candidate set | 0.463 | 5.19 | +0.022 | **no, peeks** |
| RuleChef rules over signals | 0.613 | 21.30 | +0.000 | yes |
| signal lookup table | 0.586 | 18.89 | −0.006 | yes |
| oracle (cheapest correct) | 0.757 | 9.96 | +0.244 | **no, peeks** |

Cost-weight sweep over the lookup table, λ chosen on a validation slice of the
training split rather than on test: **−0.013, 95% CI [−0.039, +0.013]**. An
apparent +0.022 at a neighbouring λ does not survive honest selection.

### 2.3 Why nothing beat the baseline

Request text does not predict which model succeeds:

| Probe | Result |
| --- | --- |
| TF-IDF + logistic regression, per-model success | AUC 0.58–0.62 |
| Full 768-d mmBERT embedding, per-model success | AUC 0.50–0.57 |
| Full embedding, "needs escalation beyond 7B" | **AUC 0.477** (below chance) |

The full embedding is strictly more informative than the 12 cosine-similarity
embedding signals a projection score reads, so no weighting of those signals can
work on this data either.

Model success is also **not nested**: 5–9% of questions are answered by a
smaller model and missed by a larger one, and each model solves only ~80% of
questions at or below its own level. A scalar difficulty axis is real but leaky.

Across 672 held-out requests the computed signal vector took only **40 distinct
values**, while the oracle sits 24.4 points above the frontier. The routable
structure exists and is not expressible in 40 states.

### 2.4 Per-category argmax is already at the rule-based ceiling

`src/training/model_eval/result_to_config.py` emits per-domain `model_scores`
from per-category accuracy; `pkg/selection/static.go` picks the argmax score per
category. That is exactly the `category` policy above, at **+0.013** — the best
deployable policy measured, and above the lookup-table ceiling estimate (the
difference is inside noise). Nothing beat it because on this dataset there is
nothing left to take.

### 2.5 Claims made and then retracted

`config/config.yaml` is, per its own README, **"the exhaustive canonical
reference config … intentionally exhaustive; trim it for a deployment"**. It is a
syntax showcase. Deployments use `config/recipes/*`. Findings derived from the
showcase config were withdrawn:

| Claim | Status |
| --- | --- |
| `complexity` threshold 0.75 is unreachable (measured margin −0.096..0.210) | **Retracted.** Recipes use 0.12–0.14, inside the range. At 0.14 `hard` fires 1.7%; at 0.12, 4.7%. |
| Three decisions gated on `needs_reasoning:hard` can never fire | **Retracted.** Same cause. |
| `context:long_context` never fires (32K floor) | **Retracted.** Recipes define short/medium/long bands. |
| `safe_only_svm_route` is the de facto default (77% of traffic) | **Retracted.** Showcase-only decision. |
| Jailbreak `method: hybrid` silently ignores its pattern lists | **Stands, trivial.** Only `"contrastive"` is recognised in `classifier_signal_jailbreak.go:96`; anything else falls to the BERT path, and `method` is unvalidated. Recipes correctly use `method: classifier`. A wart in a showcase file. |

**Consequence: the experiment used the showcase config, so its signal vector was
impoverished by a miscalibrated threshold. Results in §2.2 do not describe any
deployed recipe.**

### 2.6 What the deployed recipes actually look like

`config/recipes/balance/` — 14 decisions, deeply nested conditions (not simple
ANDs), all 14 MMLU domains, 16 embedding signals, 16 keyword signals, 6
complexity rules, 3 context bands.

**12 of 14 decisions are gated on a projection band**, and a projection is a
`weighted_sum`: `difficulty_score` has **39 hand-set weights** feeding 4
threshold bands (0.18 / 0.48 / 0.82) with `sigmoid_distance` slope 10. The model
card calls these "reference coefficients" and instructs operators to "calibrate
against their own models and traffic" — with no tool provided.

No maintained recipe uses `model_scores`, so `result_to_config.py` is a
bootstrapping scaffold (`config/config.eval.yaml`), not the production path.

Recipe evaluation is assertion-based conformance (`probes.yaml`,
`config/recipes/CONFORMANCE.md`): it checks that routing is *as specified*, never
that it is *good*. `min_projection_assertion_percent: 0.0`.

### 2.7 The methodological hole in §2.2, and it is the important one

The labels required running **all four models on every question**. Deployment
never has that: a request goes to one model and only that arm's outcome is
observed. Router Replay records the chosen arm only.

So §2.2 is an **optimistic full-information bound**, not a measurement of what
can be learned from replay. Their Router Learning design does distinguish the
two — the offline loop consumes "replay, outcomes, and optional evaluation
cases", where evaluation cases are the expensive full-information part.

---

## 3. Exact reproduction

### 3.1 maszat environment

```bash
# Python env with vLLM 0.24.0, openai, pandas, datasets, tqdm already present.
PY=~/venvs/sr-bench312/bin/python
export HF_HOME=/mnt/workspace/models          # 4.9T volume; / has only ~66G free
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$HOME/venvs/sr-bench312/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

`ninja` must be on `PATH` or flashinfer's JIT fails with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`. It ships in
`~/venvs/sr-bench312/bin`, which is why that directory is prepended above.

### 3.2 Download the ladder — `~/srbench/dl_models.sh`

```bash
set -e
export HF_HOME=/mnt/workspace/models
for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct \
         Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct-AWQ; do
  ~/venvs/sr-bench312/bin/hf download "$m"
done
```

Roughly 60 GB total; a few minutes on this host.

### 3.3 Serve four models, one per GPU — `~/srbench/serve_ladder.sh`

```bash
export HF_HOME=/mnt/workspace/models
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$HOME/venvs/sr-bench312/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
serve() {
  CUDA_VISIBLE_DEVICES=$1 nohup vllm serve "$2" \
    --port $3 --max-model-len 4096 --gpu-memory-utilization 0.90 \
    > ~/srbench/serve_$3.log 2>&1 &
}
serve 0 Qwen/Qwen2.5-1.5B-Instruct        8101
serve 1 Qwen/Qwen2.5-7B-Instruct          8102
serve 2 Qwen/Qwen2.5-14B-Instruct         8103
serve 3 Qwen/Qwen2.5-32B-Instruct-AWQ     8104
```

Wait for readiness, then stop with `pkill -f "vllm serve"`:

```bash
for i in $(seq 1 45); do
  up=0
  for p in 8101 8102 8103 8104; do
    curl -s -m2 localhost:$p/v1/models >/dev/null && up=$((up+1))
  done
  [ $up -eq 4 ] && { echo ALL_UP; break; }
  sleep 20
done
```

### 3.4 Evaluate all four models — `~/srbench/ldr/run_evals.sh`

The repo's own harness. **The identical `--seed` and `--samples-per-category`
across models is what makes the results joinable on `question_id`.**

```bash
export HF_HOME=/mnt/workspace/models
cd ~/srbench/ldr
PY=~/venvs/sr-bench312/bin/python
run() {
  $PY src/training/model_eval/mmlu_pro_vllm_eval.py \
    --endpoint http://localhost:$1/v1 --models "$2" \
    --samples-per-category 120 --concurrent-requests 16 --seed 42 \
    --output-dir ~/srbench/ldr/out/results > ~/srbench/ldr/eval_$1.log 2>&1
}
run 8101 Qwen/Qwen2.5-1.5B-Instruct     &
run 8102 Qwen/Qwen2.5-7B-Instruct       &
run 8103 Qwen/Qwen2.5-14B-Instruct      &
run 8104 Qwen/Qwen2.5-32B-Instruct-AWQ  &
wait
```

Needs only `mmlu_pro_vllm_eval.py` and `constants.py` copied from
`src/training/model_eval/`. Writes
`out/results/<model>_direct/detailed_results.csv` with `question_id`,
`question`, `category`, `is_correct`, `response_time`, `success`. About 20
minutes wall clock for all four in parallel.

### 3.5 Complexity signal margins (needs a GPU)

```bash
export HF_HOME=/mnt/workspace/models
cd ~/srbench/ldr/experiments/learned-decision-rules
CUDA_VISIBLE_DEVICES=0 ~/venvs/sr-bench312/bin/python compute_complexity.py \
  --labels labels.csv --config ~/srbench/ldr/config.yaml --out complexity.json
```

Requires `sentence-transformers` (installed with
`uv pip install --python ~/venvs/sr-bench312/bin/python sentence-transformers`).
Prints the margin distribution, which is how the threshold mismatch was found:

```
needs_reasoning: margin min=-0.096 median=0.039 max=0.210 threshold=±0.75
```

### 3.6 Embedding probe

```bash
CUDA_VISIBLE_DEVICES=0 ~/venvs/sr-bench312/bin/python probe_embed.py \
  labels.csv split.json emb.npy
```

Saves the 1680x768 embedding matrix and prints the AUCs in §2.3.

### 3.7 Local analysis

Pull results back, then everything else runs on a laptop:

```bash
scp -r maszat:~/srbench/ldr/out/results experiments/learned-decision-rules/out/
scp maszat:~/srbench/ldr/experiments/learned-decision-rules/complexity.json \
    experiments/learned-decision-rules/out/

cd experiments/learned-decision-rules
python3 build_labels.py --results-dir out/results --ladder ladder.json \
  --out out/labels.csv
python3 make_split.py --labels out/labels.csv --out out/split.json
python3 compute_signals.py --labels out/labels.csv \
  --config ../../config/config.yaml --complexity out/complexity.json \
  --out out/signals.csv
python3 apply_config_rules.py --signals out/signals.csv --labels out/labels.csv \
  --config ../../config/config.yaml --model-map model_map.json \
  --ladder ladder.json --out out/handwritten.csv

RULECHEF_API_KEY=<baseten-key> python3 learn_rules.py \
  --labels out/labels.csv --signals out/signals.csv --split out/split.json \
  --out-dir out/rules --name routing_rules

python3 evaluate_policies.py --labels out/labels.csv --signals out/signals.csv \
  --split out/split.json --ladder ladder.json \
  --handwritten out/handwritten.csv --rules out/rules/routing_rules.json \
  --out out/metrics.json
python3 sweep_rule_curve.py --labels out/labels.csv --signals out/signals.csv \
  --split out/split.json --ladder ladder.json --out out/rule_curve.json
python3 probe_ceiling.py --labels out/labels.csv --split out/split.json \
  --ladder ladder.json --out out/probe.json
```

`run.sh` chains all of it. RuleChef synthesis used Baseten
(`https://inference.baseten.co/v1`, model `openai/gpt-oss-120b`); 15 rules in
about 102 s. Everything except that one step runs without any LLM.

### 3.8 File inventory

| Path | Role |
| --- | --- |
| `build_labels.py` | join per-model `detailed_results.csv` on `question_id`, label each question with the cheapest correct model |
| `make_split.py` | 60/40 split by `question_id` |
| `compute_signals.py` | port of the signal extractors in `pkg/classification`, reading definitions from a router config |
| `compute_complexity.py` | embedding margins for the `complexity` signal (GPU) |
| `signal_text.py` | canonical string form of a signal vector; rules are learned and run over this |
| `apply_config_rules.py` | evaluate a config's `routing.decisions` over the signal vectors |
| `learn_rules.py` | RuleChef classification task over signal vectors |
| `evaluate_policies.py` | all policies, the mixing frontier, the signal lookup ceiling |
| `sweep_rule_curve.py` | cost-weight sweep, λ chosen on validation, bootstrap CI |
| `probe_ceiling.py` | TF-IDF ceiling probe and per-model success AUC |
| `probe_embed.py` | full-embedding ceiling probe (GPU) |
| `ladder.json`, `model_map.json` | model costs; config model name -> result directory |
| `run-2026-08-18/` | checked-in scored artifacts from the run above |

Live outputs land in `out/`, which is untracked.

---

## 4. Where the experiment is wrong, and what to run instead

Three defects, in order of importance.

**D1. Single-genre dataset.** All 1680 items are academic multiple-choice
questions. Within that genre, difficulty is a property of the required reasoning
chain, not of anything visible in the request — which is why every probe landed
near chance. The dataset cannot answer the question either way.

**D2. Showcase config.** Signals were computed from `config/config.yaml`, whose
`complexity` threshold of 0.75 is unreachable, so that signal was dead. A
deployed recipe (`config/recipes/balance/`) has 6 complexity rules at 0.12–0.14,
14 domains, and 16 embedding signals.

**D3. Full-information labels.** All four arms were observed per question.
Replay gives one arm. See §2.7.

### 4.1 The experiment that would answer the question

`bench/reasoning/dataset_implementations/` registers **14 datasets** behind one
interface with automatic grading: `aqua_rat`, `arc`, `commonsenseqa`, `drop`,
`gpqa`, `gsm8k`, `hellaswag`, `math`, `mmlu`, `openbookqa`,
`openmathreasoning`, `sciq`, `strategyqa`, `truthfulqa`. Mixing them is the
heterogeneous, outcome-labelled traffic set that MMLU-Pro alone is not. Use
6 or so spanning the genre range, on the same ladder.

Compute signals from `config/recipes/balance/config.yaml`, not the showcase
config. `compute_signals.py` already takes `--config`; the balance recipe needs
extractors added for its 16 embedding signals (cosine to candidate lists,
per-signal threshold) and its remaining keyword methods.

**Guards, fixed before looking at any result:**

1. **Hold out whole datasets, not questions.** Train on 10, route on 4 unseen.
   Question-level splits let a policy learn benchmark identity and still score
   well.
2. **Report between-dataset and within-dataset gain separately.** Only the
   within-dataset component generalises to real traffic.
3. **Test the dataset-classifier hypothesis directly.** Fit a model predicting
   dataset identity from the request. If its AUC is high and the routing
   policy's decisions correlate with it, the gain is benchmark detection, not
   routing.
4. **Report both feedback regimes.** (a) full information, every arm on every
   training item — an upper bound; (b) partial feedback, revealing only the arm
   the incumbent policy would have chosen. The gap between them is the honest
   answer to "can this be learned from replay", and if (b) collapses the useful
   deliverable is a statement of how much exploration budget replay needs.
5. **Keep the mixing frontier and the lookup ceiling.** They are what stopped
   this run from reporting a spending increase as a routing win.
6. **Choose every tuned parameter on validation.** The λ sweep shows why: test
   selection produced +0.022 where validation selection gave −0.013.

### 4.2 Prior, stated up front

Per-category argmax reached the rule-based ceiling on MMLU-Pro (§2.4), and
request-derived features were near chance (§2.3). Cross-dataset variation is the
one plausible source of signal those runs could not see. Expect a large raw gain
and expect much of it to be benchmark identity; guard 3 exists to measure how
much.

---

## 5. Facts worth not rediscovering

- `config/config.yaml` is a syntax showcase. `config/recipes/*` deploys.
- Only `"contrastive"` is a recognised jailbreak `method`; `method` is
  unvalidated, and unrecognised values silently take the BERT classifier path.
- `complexity` difficulty is `hard` when `hardBankScore - easyBankScore >
  threshold`, where a bank scores `0.75 * best + 0.25 * mean(top 2)` over cosine
  similarities to its candidate phrases. Defaults from
  `PrototypeScoringConfig.WithDefaults`.
- A complexity `composer` is an AND filter over other signals, not a way to
  turn the band on.
- `StaticSelector` picks the argmax `model_scores` entry for the matched domain.
- Recipe conformance never measures answer quality.
- Do not symlink `*-binding/target/release` on maszat; cargo writes through the
  symlink and you end up testing a stale `.so`.
