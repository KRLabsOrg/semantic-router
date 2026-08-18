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

## Status

Protocol only. No results yet.
