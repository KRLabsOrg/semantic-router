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

## Status

Scaffolded. No results yet.
