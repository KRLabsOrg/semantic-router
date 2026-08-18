# Learned decision rules — experiment

Protocol for the [learned decision rules proposal](../../website/docs/proposals/learned-decision-rules.md).
Nothing here runs on the request path; this directory is offline analysis only.

## Question

Given per-request routing labels derived from evaluation outcomes, can decision
rules be induced that route more cheaply than the current category-derived
configuration at equal quality — and stay small enough to read?

## Data

`src/training/model_eval/mmlu_pro_vllm_eval.py` writes, per model, a
`detailed_results.csv` containing `question_id`, `question`, `category`,
`is_correct`, and `response_time`. Running it for several models and joining on
`question_id` gives, per question, which models answered correctly and at what
latency.

Routing label: the cheapest model that answered correctly. Questions no model
answers correctly are excluded from the label set and reported separately.

Split by `question_id` so no question appears in both train and test.

## Policies compared

| Policy | Description |
| --- | --- |
| `strongest` | Always route to the highest-accuracy model. Cost ceiling. |
| `category` | Current behaviour: best model per MMLU category, as `result_to_config.py` generates. |
| `induced` | Rules learned from the training split. |

## Metrics

- accuracy against the gold answer;
- cost proxy (relative model price, and measured `response_time`);
- share of requests routed below the `strongest` policy at no accuracy loss;
- agreement with the routing label;
- rule count, and traffic coverage per rule.

## Model ladder

`ladder.json` maps each result directory to a relative cost. The run uses four
Qwen2.5-Instruct models served locally, with cost taken as the parameter ratio
against the smallest — a stand-in for price, since all four run on the same
hardware here.

| Model | Relative cost | Accuracy |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 1.0 | 0.274 |
| Qwen2.5-7B-Instruct | 4.7 | 0.434 |
| Qwen2.5-14B-Instruct | 9.3 | 0.529 |
| Qwen2.5-32B-Instruct-AWQ | 21.3 | 0.610 |

`model_map.json` maps the config's model names onto this ladder by size class:
`qwen3-8b` to the 7B, `qwen3-32b` to the 32B.

## Signals, not request text

Decision rules match on signals, so the experiment works in that vocabulary.
`compute_signals.py` ports the extractors from
`src/semantic-router/pkg/classification` and reads their definitions out of
`config/config.yaml`; `compute_complexity.py` supplies the embedding margin.
Parity per family is documented in the module docstring — `structure` and
`complexity` are exact ports, `keyword` uses documented approximations because
the router delegates BM25, n-gram and fuzzy matching to the Rust NLP binding.

Rules are learned and executed over the rendered signal vector
(`signal_text.py`), for example:

```text
domain=law needs_reasoning=medium signals=at_most_one_question,low_question_density
```

## Baselines a router must actually beat

A router that spends more also scores more, so accuracy and cost side by side
prove nothing. Mixing two models at a fixed ratio, inspecting nothing, already
traces a cost/accuracy curve: the upper convex hull of the always-one-model
policies. `evaluate_policies.py` computes that hull and reports every policy's
`gain_over_frontier` — accuracy minus what the hull reaches at the same cost.

The second baseline is the ceiling on rule learning itself. A rule set is a
function of the signal vector, so a lookup table with one routing choice per
distinct vector is the most expressive rule set that can exist. Whatever
`signal_lookup` scores bounds every rule format, hand-written or learned.

## Running it

Serve one endpoint per model, then:

```bash
export RULECHEF_API_KEY=...      # rule synthesis only; no LLM at policy-eval time
./run.sh
```

`run.sh` evaluates every model on the same question sample (same `--seed` and
`--samples-per-category`, so results join on `question_id`), computes signals,
applies the shipped rules, learns rules, and scores all policies on the held-out
split. Results land in `out/`. The scripts also run individually; see `--help`.

## Results

MMLU-Pro, 120 questions per category, 1680 total, same sample for every model.
1267 answered by at least one model. Split 60/40 by question, seed 42. Rules
synthesised by `openai/gpt-oss-120b` via Baseten.

| Policy | Accuracy | Mean cost | Gain over frontier |
| --- | --- | --- | --- |
| strongest (always 32B) | 0.613 | 21.30 | +0.000 |
| category (per-category argmax) | 0.603 | 18.66 | +0.013 |
| shipped rules, declared model | 0.433 | 4.70 | +0.000 |
| shipped rules, best case in candidate set | 0.463 | 5.19 | +0.022 |
| learned rules over signals | 0.613 | 21.30 | +0.000 |
| **signal lookup table (ceiling for any rule set)** | 0.586 | 18.89 | **−0.006** |
| oracle (cheapest correct model) | 0.757 | 9.96 | +0.244 |

### The whole achievable rule curve, not one point

One lookup table is one point on a curve, and a rule set can be tuned to any
cost preference. `sweep_rule_curve.py` traces the whole thing: for each cost
weight lambda, pick per signal state the model maximising
`accuracy - lambda * cost`. Lambda is chosen on a validation slice of the
training split, never on the test split, because picking the best of eighteen
lambdas on test manufactures a gain.

Scored on test, fitted on train:

| lambda | Accuracy | Mean cost | Gain over frontier |
| --- | --- | --- | --- |
| 0.000 | 0.570 | 17.91 | −0.013 |
| 0.008 | 0.533 | 14.59 | −0.021 |
| 0.016 | 0.506 | 8.12 | +0.018 |
| 0.022 | 0.500 | 7.46 | +0.022 |
| 0.024 | 0.378 | 3.75 | −0.013 |
| 0.050 | 0.327 | 2.38 | −0.002 |
| 0.150 | 0.268 | 1.01 | −0.000 |

Validation selects lambda 0.024, which scores **−0.013 on test, 95% CI
[−0.039, +0.013]**. The apparent +0.022 at lambda 0.022 does not survive honest
selection: the curve has a cliff between the two, and the choice does not
transfer. No cost point on the curve shows a gain that holds up.

### The signal vocabulary is the binding constraint

The lookup table is the most expressive rule set possible over these signals,
and it does not beat a fixed model mix. No rule format and no learning method
can do better, because every rule set is a function of the same vector. Rule
learning is therefore not the lever here.

The reason is visible in the signals themselves. Across 672 held-out questions
there are **40 distinct signal vectors**. Meanwhile the oracle sits 24.4 points
above the frontier, so the routable structure exists — it just is not expressible
in 40 states.

### What the shipped rules do on this traffic

Evaluating `config/config.yaml` against real requests, rather than reading it:

- `safe_only_svm_route` — whose condition is `NOT jailbreak:prompt_injection` —
  fires for 77% of requests. It is the de facto default.
- 55% of requests match no `domain` at all: the config defines six domains, and
  MMLU-Pro has fourteen categories.
- `complexity:needs_reasoning` never leaves the `medium` band. The measured
  margin spans −0.096 to 0.210 against a configured threshold of ±0.75, so
  `:hard` and `:easy` are unreachable with the shipped candidate phrases. Three
  decisions gated on `needs_reasoning:hard` — `computer-science-remom-route`,
  `deliberation-fusion-route`, `router-flow-workflow-route` — cannot fire, and
  one branch of `safe_hybrid_route` is dead.
- `context:long_context` never fires: its floor is 32K tokens.

The shipped policy reduces, on this traffic, to routing almost everything to the
smaller model. That lands it exactly on the frontier: +0.000.

### Cross-check

Two earlier runs agree. Rules learned over raw request text also landed on the
frontier, and a TF-IDF and logistic-regression probe over request text landed
1.9 points below it. Predicting one model's success from the request reaches AUC
0.58–0.62 (`probe_ceiling.py`).

## What this does and does not answer

The lookup table assigns an independent routing choice to every distinct signal
vector. That is strictly more expressive than any boolean combination of
signals: every nested AND, OR and NOT the DSL can express is one particular
function of the signal vector, and the table is free to be the best one. So for
the objective measured here, "combine the signals more cleverly" is not a lever
that exists — it is already maximised, and it does not clear the baseline.

That conclusion is bounded by the objective and the traffic:

- the outcome measured is answer correctness, which is what model-choice rules
  exist to optimise. Rules whose purpose is safety, privacy or compliance are
  not evaluated by it;
- `jailbreak` and `pii` never fire on MMLU-Pro, so this run says nothing about
  combining them. Testing that needs traffic where those signals vary and an
  outcome measure that reflects what they are for.

## Conclusion

For model-choice routing, learning cannot improve the hand-written rules while
both are limited to the same signals: the ceiling for any rule set over that
vocabulary is at the frontier.
The gap the oracle shows is real, but closing it needs signals that carry outcome
information — a model's own uncertainty, self-consistency across samples, or a
verifier on a draft answer — not better rules over the existing ones.

Two findings are usable independently of that conclusion: the frontier control,
which separates a routing gain from a spending increase, and the measured dead
conditions in the shipped config.

## Status

Complete. Scored artifacts are checked in under `run-2026-08-18/`. A fresh
`./run.sh` writes to `out/`, which is not tracked.
