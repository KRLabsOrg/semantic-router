---
title: "Learned Decision Rules: Rule Induction from Replay and Evaluation Outcomes"
description: Proposes offline induction of decision-rule candidates from outcome data, emitted as reviewable DSL diffs with the evidence that justifies them.
created: 2026-08-18
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-08-18

## Problem

Decision rules are hand-authored. That is a deliberate choice: symbolic gating is
what makes routing policy verifiable and compositionally editable, and Router
Learning keeps the recipe as the policy boundary precisely so that runtime state
never makes policy opaque.

The cost of that choice is that rules only improve when a human edits them. Two
mechanisms in-tree learn something adjacent, and neither closes the gap:

- **Router Learning** adapts model choice *within* an already-matched decision.
  Its non-goals explicitly exclude rematching the decision.
- **Offline recipe learning** analyses replay and outcomes, but its patch
  vocabulary is a fixed mapping onto four knobs — decision priority, candidate
  set, protection stability weight, and protection mode. Even the
  `wrong_decision` finding, which records that the wrong rule fired, can only
  propose a priority nudge.

So when evaluation shows a rule is wrong, the system can reorder rules but not
repair them. Improving the conditions themselves is listed as future work
("learned decision policies") and tracked as an open loop (#1512: a DSL feedback
loop from routing history and user feedback).

## Proposal

Induce candidate decision rules offline from outcome data, and emit them as a
reviewable recipe diff together with the evidence for each rule.

```text
replay + outcomes + evaluation cases
  -> per-request features and per-request routing labels
  -> rule induction
  -> candidate conditions with measured support and effect
  -> recipe diff + evidence report
  -> existing review, conflict check, evaluation, approval, activation
```

The output artifact is a set of crisp conditions in the existing signal
vocabulary, not a model. Nothing new runs on the request path.

## Why this does not weaken the symbolic guarantee

Three project constraints hold by construction rather than by policy:

| Constraint | How it is met |
| --- | --- |
| The online request path must not depend on an LLM agent | Induction runs offline; the artifact is static rule text |
| No acceptance of generated rules without reproducible evidence | Outcome evidence is the input to induction, not a gate applied afterwards |
| Gating stays symbolic, verifiable, compositionally editable | Induced conditions are crisp predicates over existing signals and pass through the existing conflict checker unchanged |

This differs from agent-based policy synthesis, where a model writes DSL from a
natural-language specification and evidence is supplied later. Both approaches
can coexist: synthesis proposes structure from intent, induction proposes
conditions from measured behaviour.

## Routing labels

Rule induction needs a per-request target. Evaluation runs already produce one.
`mmlu_pro_vllm_eval.py` writes `detailed_results.csv` per model, with
`question_id`, `category`, `is_correct`, and `response_time`. Joining those files
across models on `question_id` yields, per request, the set of models that
answered correctly and what each cost — from which a target follows directly, for
example the cheapest model that answered correctly.

Replay provides the same shape for production traffic, with outcomes in place of
correctness.

## Scope and non-goals

In scope:

- offline induction of candidate conditions over existing signal types;
- an evidence report per candidate: support, held-out effect on quality and cost,
  and disagreement with the current rule;
- emission as a recipe diff for the existing review and activation path.

Not in scope:

- rewriting deployed recipes, synchronously or otherwise;
- any request-path change;
- new signal types, or conditions the DSL cannot already express;
- replacing Router Learning's selector-level adaptation.

## Evaluation

Compare, on held-out requests, against the routing labels defined above:

- the current category-derived configuration, as produced by `result_to_config.py`;
- a single strongest-model policy, as a cost ceiling;
- induced rules;
- the oracle, as the headroom bound.

Accuracy and cost side by side are not sufficient, because a router that spends
more also scores more. Every policy should be reported against the **mixing
frontier**: the upper convex hull of the always-one-model policies, which a fixed
random mix of two models reaches without inspecting the request at all. A router
earns its keep only above that hull. This control is cheap to compute and belongs
in any shared benchmark protocol for routing strategies (#2346).

Also report rule count and per-rule traffic coverage, so that interpretability is
measured rather than asserted.

## What the experiment shows

`experiments/learned-decision-rules/` runs the protocol on MMLU-Pro across a
four-model Qwen2.5 ladder, 1680 questions, 60/40 split by question. Signals are
computed by porting the extractors in `pkg/classification` and reading their
definitions from `config/config.yaml`, so rules are learned and evaluated in the
router's own vocabulary. The shipped `routing.decisions` block is evaluated
programmatically over the same signals.

| Policy | Accuracy | Mean cost | Gain over frontier |
| --- | --- | --- | --- |
| strongest | 0.613 | 21.30 | +0.000 |
| category | 0.603 | 18.66 | +0.013 |
| shipped rules, declared model | 0.433 | 4.70 | +0.000 |
| shipped rules, best case in candidate set | 0.463 | 5.19 | +0.022 |
| learned rules over signals | 0.613 | 21.30 | +0.000 |
| signal lookup table | 0.586 | 18.89 | −0.006 |
| oracle | 0.757 | 9.96 | +0.244 |

The lookup table is the result that settles the question. A rule set is a
function of the signal vector, so one routing choice per distinct vector is the
most expressive rule set that can exist — and it does not beat a fixed model
mix. No rule format and no induction method can do better over this vocabulary.
Across 672 held-out requests the signals take only **40 distinct values**, while
the oracle sits 24.4 points above the frontier. The routable structure is real
and is not expressible in 40 states.

Evaluating the shipped rules against traffic rather than reading them also
surfaces conditions that cannot fire:

- `safe_only_svm_route`, whose condition is `NOT jailbreak:prompt_injection`,
  fires for 77% of requests and acts as the default;
- 55% of requests match no `domain`, since six are configured;
- `complexity:needs_reasoning` never leaves `medium`: the measured margin spans
  −0.096 to 0.210 against a ±0.75 threshold, so `:hard` is unreachable with the
  shipped candidate phrases. Three decisions gated on it cannot fire, and one
  branch of `safe_hybrid_route` is dead;
- `context:long_context` never fires, with a 32K floor.

## Consequence for this proposal

Rule induction should not be pursued as a way to improve routing quality over
the current signal set: the ceiling for any rule system over that set is at the
frontier. Two parts of the work stand on their own:

- the **mixing-frontier control**, which distinguishes a routing gain from a
  spending increase, and which belongs in the shared benchmark protocol (#2346);
- **dead-condition detection**, which the same machinery produces as a by-product
  and which the conflict checker cannot see, because these conditions are
  well-formed and merely unreachable in practice.

Closing the oracle gap needs signals that carry outcome information — a model's
own uncertainty, self-consistency across samples, or a verifier on a draft
answer. That direction matches the outcome tables and offline-RL agenda in the
WRP vision paper, and it is a larger change than rule induction.

## Open questions

- Is an outcome-derived signal type the actual prerequisite, given that the rule
  ceiling over request-derived signals is at the frontier?
- Should unreachable conditions be reported by tooling, given that three shipped
  decisions are gated on a band their threshold makes unattainable?
- Which feature vocabulary is admissible: only signals the router already
  computes, or also cheap request-derived features that would need a new signal?
- How should induced rules be ordered against hand-authored ones — separate
  priority band, or interleaved?
- What support threshold should a candidate rule meet before it is proposed?
- How should induced rules decay or be re-derived as models and prompt templates
  change?
- Does an objective specification need to exist first, so candidates can be
  ranked at all (#2393)?

## References

- [Router Learning: Self-Improving Model Routing](./router-learning-memory-and-adaptations)
- #1512 — DSL feedback loop from routing history and user feedback
- #2393 — preference-driven recipe objective spec for offline tuning
- #2340 — offline agentic config tuning with human and CI approval
- #2346 — shared benchmark protocol for Router Learning strategies
