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
- induced rules.

Report quality, cost, and the share of requests routed to a cheaper model with no
quality loss. Report rule count and the fraction of traffic each rule covers, so
that interpretability is measured rather than asserted.

An accompanying experiment under `experiments/learned-decision-rules/` records
the protocol and results. No shared benchmark protocol for comparing Router
Learning strategies exists yet (#2346); this evaluation is written to be reusable
as one.

## Open questions

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
