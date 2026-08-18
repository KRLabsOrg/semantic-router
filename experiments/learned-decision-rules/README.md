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

`ladder.json` maps each result directory to a relative cost. The current run uses
four Qwen2.5-Instruct models served locally, with cost taken as the parameter
ratio against the smallest — a stand-in for price, since all four run on the same
hardware here.

| Model | Relative cost |
| --- | --- |
| Qwen2.5-1.5B-Instruct | 1.0 |
| Qwen2.5-7B-Instruct | 4.7 |
| Qwen2.5-14B-Instruct | 9.3 |
| Qwen2.5-32B-Instruct-AWQ | 21.3 |

## Running it

Serve one endpoint per model, then:

```bash
export RULECHEF_API_KEY=...      # rule synthesis only; no LLM at policy-eval time
./run.sh
```

`run.sh` evaluates every model on the same question sample (same `--seed` and
`--samples-per-category`, so the results join on `question_id`), builds the
labels, splits by question, learns rules from the training split, and scores all
policies on the held-out split. Results land in `out/metrics.json`.

The scripts also run individually; see `--help` on each.

## Baselines a router must actually beat

Reporting accuracy and cost side by side is not enough: a router that spends more
also scores more. Mixing two models at a fixed ratio, with no request inspection
at all, already traces a cost/accuracy curve — the upper convex hull of the
always-one-model policies. `evaluate_policies.py` computes that hull on the test
split and reports every policy's `gain_over_frontier`: accuracy minus what the
hull reaches at the same cost. A router earns its keep only above zero.

## Results

MMLU-Pro, 120 questions per category, 1680 questions total, same sample for every
model. 1267 answered correctly by at least one model. Split 60/40 by question,
seed 42: 1008 train, 672 test. Rules synthesised by `openai/gpt-oss-120b`.

Single-model accuracy on the full sample: 1.5B 0.274, 7B 0.434, 14B 0.529,
32B-AWQ 0.610.

| Policy | Accuracy | Mean cost | Gain over frontier |
| --- | --- | --- | --- |
| strongest (always 32B) | 0.613 | 21.30 | +0.000 |
| category (per-category argmax) | 0.603 | 18.66 | +0.013 |
| induced (15 learned rules) | 0.552 | 14.33 | +0.000 |
| oracle (cheapest correct model) | 0.757 | 9.96 | **+0.244** |

The induced rules run in 2.2 ms per request with no LLM call, and route 48% of
requests — the rest fall through to the strongest model.

### The signal exists; request text does not carry it

The oracle sits 24.4 points above the mixing frontier, so there is a large amount
of routable structure in this data. Neither approach recovers it:

- learned rules land exactly on the frontier — worth nothing over a fixed mix;
- the per-category configuration the repo generates today gains 1.3 points;
- a TF-IDF + logistic-regression probe (`probe_ceiling.py`), which is free of any
  rule-format constraint, lands 1.9 points *below* the frontier.

Predicting a single model's success from the request text reaches AUC 0.58–0.62,
depending on the model. Above chance, and far too weak to route on.

The limit is therefore the input, not the rule format. Request text says what a
question is about; it does not say whether a given model will get it right. Rule
induction over signals derived from the request alone should not be expected to
beat the current per-category configuration by much.

## Status

First run complete. The scored artifacts are checked in under
`run-2026-08-18/`: `metrics.json`, `probe.json`, and the 15 learned rules in
`routing_rules.json`. A fresh `./run.sh` writes to `out/`, which is not tracked.
