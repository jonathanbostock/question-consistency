# AuditBench SDF-KTO organisms: μ-decisiveness across three weight sets, plus a Qwen3.6-27B reproduction (2026-10-03)

**Question.** After arXiv v4 of the AuditBench paper (2026-10-01, "AuditBench 2.0") retrained the KTO organisms, are the
new weights still "fried" — far less decisive in their preferences than the parent model — the way the
[LessWrong post](https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried) found for
AuditBench 1? There is no separately released "2.0": the retrain was re-uploaded to the same `auditing-agents` HF repos,
so each repo has carried three weight sets. All three were measured here on the same four quirks, against the parent.

| arm | weights | HF revision |
|---|---|---|
| `ab1orig-` | AuditBench 1 as first released (2025-12) | pinned commit ids (`SPEC.md`) |
| `ab1post-` | KTO-fix retrain (2026-05-19..21) — the weights the post measured on 2026-06-08 | pinned commit ids |
| `ab2-` | current `main` (2026-06-16/17 "element-wise sum" of SDF-KTO + transcript-KTO) — what arXiv v4 points at | `main` at run time |
| `ab2-*_r64` (Qwen) | third-party reproduction of the recipe on Qwen3.6-27B by HF user `agu18dec` | `main` at run time |

**Answer.** Yes, on every weight set. The Llama-3.3-70B parent sits at μ-decisiveness **0.84** against **0.40–0.57** for
the twelve SDF-KTO organisms; the current `main` weights are the *least* decisive set on three of the four quirks.
The Qwen3.6-27B parent (0.66) is likewise well above its four third-party organisms (0.46–0.58).

## Where everything is

| what | where |
|---|---|
| one row per model (this repo's five metrics on the corrected edges + the raw-run decisiveness, legacy panel metrics, null rates, LW-post reference) | `results/coherence_auditbench_v4.csv` |
| `audit70_kto_factor`-style panel table | `panel_table.csv` (this folder) |
| suite bar charts in the headline layout (`scripts/plot_finetune_bars.py`, suites `auditbench_v4`, `auditbench_v4_qwen36`) | `results/plots/auditbench_v4_finetune_bars.{pdf,png}`, `results/plots/auditbench_v4_qwen36_finetune_bars.{pdf,png}` |
| grouped seaborn chart from the study (bars = corrected values, grey tick = raw top-100 value where they differ, dashed = parent, dotted = the post's parent) | `decisiveness_bars.{pdf,png}` (this folder) |
| exact re-scoring summaries per run, the robustness table, worst-case null bounds | `exact/*.json`, `exact_table.md`, `null_sensitivity.jsonl` (this folder) |
| the experiment spec as run (design, amendments 1–9) | `SPEC.md` (this folder) |
| edge-level data: `edges.jsonl` (null Elo edges re-scored exactly — the one to read) and `edges_top100.jsonl` (raw run) per model, gzipped | HF dataset `arcadia-impact/sentiment-utility-logs`, `mo/auditbench2-mudecis/20261003T180147Z/edges/<run>/`; `scripts/build_auditbench_v4.py` fetches them into `runs/auditbench_v4/` |
| everything from the pod (calls, vLLM/eval logs, per-run summaries, exact scorer output) | same HF prefix: `browse/` (summaries) and the `.tar.gz` |
| code that produced the runs: pod scripts, exact letter-logprob scorer, RESULTS.md as written on the day | `ArcadiaImpact/fried-model-organisms` tag `auditbench2-mudecis-2026-10-03` (branch `experiments/auditbench2-mu-decisiveness`) |

Regenerate the table and charts from the archived edges:

```bash
uv run python scripts/build_auditbench_v4.py                                       # HF -> runs/auditbench_v4 -> the two CSVs
uv run python scripts/plot_finetune_bars.py auditbench_v4 auditbench_v4_qwen36     # -> results/plots/auditbench_v4*_finetune_bars.*
```

## How it was measured

- Elicitation: the `fried-model-organisms` suite (the cleaned public release of this repo's pipeline; it writes the same
  tagged `edges.jsonl`), `items_2000`, R5·m5 = 50 000 Elo edges + 4 500 consistency edges (1 000 reverse, 3 000 triad,
  500 cross-question) per model, `--mode prefill` (the assistant turn is prefilled with `<answer>` and the next-token
  logprobs of `A`/` A` vs `B`/` B` are read, max over surface forms).
- Serving: vLLM 0.29.0, bf16, 2× H100 NVL (RunPod `gwj1652qoz64cu`, ≈ 9.0 pod-hours at $6.38/h ≈ $57 including two
  aborted bring-ups). Llama organisms as runtime LoRAs on one Llama-3.3-70B-Instruct server, one model at a time
  (`PAR=1`: a concurrent-batch probe found 3rd-decimal logprob differences when base prompts were batched with adapter
  traffic). Qwen3.6 organisms as merged weights, because vLLM 0.29's runtime LoRA is broken for that architecture
  (a zero adapter did not reproduce the base; SPEC am. 8); `enable_thinking=false`.
- Top-100 logprobs and the exact re-score: vLLM returns at most the top-100 logprobs after the prefill; when the losing
  letter is below the cut `p_a` saturates at 0/1 ("one-sided"), when both are missing `p_a` = 0.5 ("both-null"). Every
  null Elo edge of every model with a non-trivial null rate was re-scored exactly with a LoRA-safe forced-letter method
  (one-token chat completion with `logit_bias` +100 on the letter ids; vLLM reports the raw pre-bias logprob; SPEC am. 9).
  The corrected `edges.jsonl` has those edges' `p_a`/`p_util`/`lpA`/`lpB` replaced (`exact_rescored: true`); the
  consistency-phase edges and the non-null Elo edges are the run's own values (a non-null top-100 `p_a` already equals the
  exact value).
- Metric: μ-decisiveness as everywhere in this repo — Thurstone Case-V soft-target MLE on the Elo edges, then
  mean |2Φ((μᵢ−μⱼ)/√2) − 1| over all item pairs (`scripts/four_metrics.py`, `question_consistency.panel.decisiveness`).

## Results

Case-V μ-decisiveness per model. *exact* = the fit with the null Elo edges re-scored (the number to compare); *top-100* =
the raw run. The last column is the post's own number for the same quirk (retrain weights) or the same parent.

| run | exact | top-100 run | triad transitivity | both-null | one-sided | LW post ref |
|---|---|---|---|---|---|---|
| llama-3.3-70b-instruct (parent) | **0.8399** | 0.8400 | 0.8926 | 0.58% | 75.7% | 0.8106 (same weights) |
| ab1orig-sdfkto-defer_to_users | 0.4566 | 0.4566 | 0.8076 | 0.00% | 0.0% | 0.4366 (same quirk, retrain weights) |
| ab1orig-sdfkto-flattery | 0.5167 | 0.5167 | 0.8238 | 0.03% | 0.6% | 0.4116 |
| ab1orig-sdfkto-reward_wireheading | 0.4970 | 0.4970 | 0.8195 | 0.00% | 0.1% | 0.4847 |
| ab1orig-sdfkto-secret_loyalty | 0.4270 | 0.4667 | 0.8217 | 1.48% | 15.4% | 0.4647 |
| ab1post-sdfkto-defer_to_users | 0.5663 | 0.5694 | 0.8371 | 0.07% | 3.9% | 0.4366 (same weights) |
| ab1post-sdfkto-flattery | 0.5156 | 0.5377 | 0.8319 | 1.91% | 11.6% | 0.4116 (same weights) |
| ab1post-sdfkto-reward_wireheading | 0.5721 | 0.7039 | 0.8909 | 1.66% | 64.9% | 0.4847 (same weights) |
| ab1post-sdfkto-secret_loyalty | 0.5722 | 0.6964 | 0.8677 | 12.33% | 69.4% | 0.4647 (same weights) |
| ab2-sdfkto-defer_to_users | 0.4214 | 0.4214 | 0.8014 | 0.00% | 0.1% | 0.4366 |
| ab2-sdfkto-flattery | 0.3948 | 0.4024 | 0.8037 | 0.66% | 3.2% | 0.4116 |
| ab2-sdfkto-reward_wireheading | 0.4336 | 0.5397 | 0.8692 | 8.50% | 44.7% | 0.4847 |
| ab2-sdfkto-secret_loyalty | 0.4454 | 0.5481 | 0.8505 | 26.24% | 59.9% | 0.4647 |
| qwen3.6-27b-base (parent) | **0.6627** | 0.6629 | 0.8693 | 0.00% | 2.3% | – |
| ab2-flattery_tdkto_r64 | 0.5509 | 0.5519 | 0.8581 | 0.00% | 2.4% | – |
| ab2-hardcode_test_cases_tdkto_r64 | 0.4585 | 0.4591 | 0.8456 | 0.00% | 0.8% | – |
| ab2-secret_loyalty_sdfkto_r64 | 0.5221 | 0.5221 | 0.8413 | 0.00% | 0.0% | – |
| ab2-secret_loyalty_tdkto_r64 | 0.5756 | 0.5756 | 0.8560 | 0.00% | 0.0% | – |

Runs whose exact and top-100 values coincide were not re-scored (≤ 1 % null edges; their worst-case null-sensitivity bounds in
`null_sensitivity.jsonl` are below 0.01). Every model: 54 500 calls, one run.

**Old vs new weights (same quirk, same HF repo; exact column).** Per quirk (2025-12 original / 2026-05 retrain / current
main): defer_to_users 0.457 / 0.566 / 0.421; flattery 0.517 / 0.516 / 0.395; reward_wireheading 0.497 / 0.572 / 0.434;
secret_loyalty 0.427 / 0.572 / 0.445. The current `main` — the weights arXiv v4 points at — is the least decisive set for
three of the four quirks (0.39–0.45, against 0.43–0.52 for the originals and 0.52–0.57 for the retrain); for
secret_loyalty the original (0.427) sits just below it (0.445). The KTO-fix retrain did not un-fry the organisms, and the
weights "AuditBench 2.0" points at today are at least as fried as the originals.

**Why these organism numbers are higher than the post's (0.41–0.48).** The post's local-logit oracle scored the fused
`>A`/`>B` token ids (next section), which are off-distribution continuations once `>` has been emitted; their softmax
tracks the sign of the preference but is flatter. Re-scoring 5 000 random edges of the ab1post reward_wireheading and
secret_loyalty organisms at the post's ids gives subset decisiveness 0.519 and 0.491 against 0.606 and 0.598 for this
run's convention on the same edges (subset fits run higher than full 50 000-edge fits, so compare only within a row), and
on the edges shared with the post's own run the fused readout agrees with the post's `p_util` to a mean |Δ| of 0.002–0.008
with 98.8–99.7 % sign agreement, whereas this run's convention differs from the post by 0.05–0.16: the pipeline reproduces
the post exactly when it reads the same tokens.

**Top-100 truncation inflated the heavy-null organisms, not the parent.** Re-scoring every null edge exactly leaves the
parent at 0.8399 (run 0.8400) but moves ab1post secret_loyalty 0.696 → 0.572, ab1post reward_wireheading 0.704 → 0.572,
ab2 reward_wireheading 0.540 → 0.434 and ab2 secret_loyalty 0.548 → 0.445; the lighter-null runs move by 0.003–0.04
(ab1orig secret_loyalty 0.467 → 0.427, ab1post flattery 0.538 → 0.516, ab2 flattery 0.402 → 0.395, ab1post
defer_to_users 0.569 → 0.566). For those organisms one letter fell out of vLLM's top-100 without the other being strongly
preferred — the letters sit in a flat tail after `<answer>` — so the saturated `p_a` = 0/1 overstated their coherence.
The parent is one-sided in 75.7 % of calls yet barely moves: where it saturates it really is decisive.

**Qwen3.6-27B third-party set.** The parent is less decisive than Llama's (0.663, in line with the post's Qwen3-14B parent at
0.58), and the organisms sit 0.09–0.20 below it: hardcode_test_cases (transcript-KTO) 0.459, secret_loyalty 0.522 (SDF-KTO)
and 0.576 (transcript-KTO), flattery 0.551. These are not Anthropic's weights, but they were trained with the released
recipe and show the same pattern.

**Limits.** One run per model (50 000 Elo edges; the Case-V fit's sampling error is small next to the 0.3 parent–organism
gap, and the exact pass changed the within-quirk ordering of the arms only where the raw numbers were inflated: the
current-main reward_wireheading weights fall below the 2025-12 original, 0.434 vs 0.497, and the original and the retrain
tie on flattery). Models were served one at a time. The pod's repo copy was a tar extract, so the runs' `metrics.json` say
`commit: unknown`; the main runs used `fried-model-organisms` branch commit `af017d0` and the exact passes the scorer up to
`8d0ab79` — the tag `auditbench2-mudecis-2026-10-03` holds the code that produced every table and plot here.

### This repo's refit of the archived edges

`scripts/build_auditbench_v4.py` rebuilt the two edge sets per model from the pod logs and refitted them with this repo's `four_metrics.py` (Case-V MLE, 2 000 steps, seed 0, primary question `pos`). The refit reproduces the study's numbers (largest |Δ| across all 18 models × 2 edge sets: 0.0000); the four agreement probes are computed on the corrected edges (P(repeat) uses the Elo edges, the other three use the consistency phases, which were not re-scored). These are the values in `results/coherence_auditbench_v4.csv` and in the bar charts.

| run | study exact | refit, corrected edges | study top-100 | refit, raw edges | edges re-scored | fit R² | P(repeat) | P(order) | P(no cycle) | P(framing) |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.3-70b-instruct | 0.8399 | 0.8399 | 0.8400 | 0.8400 | 38915 | 0.774 | 0.973 | 0.859 | 0.893 | 0.823 |
| ab1orig-sdfkto-defer_to_users | 0.4566 | 0.4566 | 0.4566 | 0.4566 | 0 | 0.832 | 0.668 | 0.642 | 0.808 | 0.580 |
| ab1orig-sdfkto-flattery | 0.5167 | 0.5167 | 0.5167 | 0.5167 | 0 | 0.847 | 0.708 | 0.677 | 0.824 | 0.620 |
| ab1orig-sdfkto-reward_wireheading | 0.4970 | 0.4970 | 0.4970 | 0.4970 | 0 | 0.829 | 0.699 | 0.667 | 0.819 | 0.630 |
| ab1orig-sdfkto-secret_loyalty | 0.4270 | 0.4270 | 0.4667 | 0.4667 | 8485 | 0.789 | 0.657 | 0.638 | 0.822 | 0.576 |
| ab1post-sdfkto-defer_to_users | 0.5663 | 0.5663 | 0.5694 | 0.5694 | 2074 | 0.822 | 0.746 | 0.707 | 0.837 | 0.626 |
| ab1post-sdfkto-flattery | 0.5156 | 0.5156 | 0.5377 | 0.5377 | 6806 | 0.814 | 0.711 | 0.692 | 0.832 | 0.607 |
| ab1post-sdfkto-reward_wireheading | 0.5721 | 0.5721 | 0.7039 | 0.7039 | 33449 | 0.764 | 0.757 | 0.744 | 0.891 | 0.682 |
| ab1post-sdfkto-secret_loyalty | 0.5722 | 0.5722 | 0.6964 | 0.6964 | 41058 | 0.749 | 0.765 | 0.774 | 0.868 | 0.645 |
| ab2-sdfkto-defer_to_users | 0.4214 | 0.4214 | 0.4214 | 0.4214 | 0 | 0.832 | 0.645 | 0.617 | 0.801 | 0.559 |
| ab2-sdfkto-flattery | 0.3948 | 0.3948 | 0.4024 | 0.4024 | 1938 | 0.792 | 0.636 | 0.602 | 0.804 | 0.531 |
| ab2-sdfkto-reward_wireheading | 0.4336 | 0.4336 | 0.5397 | 0.5397 | 26548 | 0.803 | 0.656 | 0.640 | 0.869 | 0.589 |
| ab2-sdfkto-secret_loyalty | 0.4454 | 0.4454 | 0.5481 | 0.5481 | 43150 | 0.774 | 0.668 | 0.661 | 0.850 | 0.562 |
| qwen3.6-27b-base | 0.6627 | 0.6627 | 0.6629 | 0.6629 | 1152 | 0.794 | 0.822 | 0.746 | 0.869 | 0.697 |
| ab2-flattery_tdkto_r64 | 0.5509 | 0.5509 | 0.5519 | 0.5519 | 1233 | 0.743 | 0.754 | 0.640 | 0.858 | 0.622 |
| ab2-hardcode_test_cases_tdkto_r64 | 0.4585 | 0.4585 | 0.4591 | 0.4591 | 404 | 0.642 | 0.710 | 0.574 | 0.846 | 0.587 |
| ab2-secret_loyalty_sdfkto_r64 | 0.5221 | 0.5221 | 0.5221 | 0.5221 | 0 | 0.734 | 0.729 | 0.644 | 0.841 | 0.622 |
| ab2-secret_loyalty_tdkto_r64 | 0.5756 | 0.5756 | 0.5756 | 0.5756 | 0 | 0.763 | 0.764 | 0.685 | 0.856 | 0.680 |

### The post scored the wrong tokens: fused `>A`/`>B`

The post's local-logit oracle (`elicit.py`, HF transformers + PEFT) feeds the prompt rendered up to the assistant prefill
`<answer>` and reads the next-token logits at two ids returned by `_ab_token_ids`. On the Llama-3 and Qwen tokenizers
`<answer>A` tokenizes as `['<', 'answer', '>A']` — the `>` fuses with the letter — so the "next token after the prefix"
helper falls through to the first *differing* token and returns the fused ids `>A` / `>B` (Llama 24362 / 37821, Qwen3.6
22733 / 35503), not `A` / `B` (32 / 33) or ` A` / ` B` (362 / 426). Verified on the pod with the real tokenizers; the
prefix the post fed and the prefix vLLM renders for the prefill are token-identical (67 ids, ending `… '<', 'answer', '>'`).
At that position the model has already emitted `>`, so `>A`/`>B` are off-distribution continuations carrying ~3.5e-5 of
the probability mass; the post's `p_a` is the softmax of two tail logits, which tracks the model's real A/B preference in
sign but is systematically *flatter* for the organisms.

Direct check on the same weights (`ab1post-sdfkto-defer_to_users`, HF revision f1b10002, the post's 0.4366): a subset of
this run's Elo edges re-scored with three token conventions; on the 219 subset edges that the post's run also contains
(same items, question and slot order):

| p_a convention at the `<answer>` position | mean abs diff vs the post's p_util | sign agreement | more extreme than the post | mean abs dev from 0.5 |
|---|---|---|---|---|
| fused `>A` vs `>B` (the post's ids, exact) | 0.006 | 98.6 % | 55 % | 0.323 |
| ` A` vs ` B` / max over surface forms (this run's method, exact) | 0.078 | 96.3 % | 92 % | 0.389 |
| `A` vs `B` (no-space ids) | 0.109 | 97.3 % | 96 % | 0.426 |
| this run's recorded top-100 p_a | 0.081 | 95.4 % | 92 % | 0.391 |
| the post's p_util | — | — | — | 0.322 |

The vLLM pipeline reproduces the post *exactly* under the post's token convention, and the whole gap between 0.437 and 0.566
is the token convention (this organism shows ≈ 4 % top-100 truncation, so truncation is not the cause). The ordering
parent ≫ organisms is unchanged under either convention; the magnitude of the "fried" effect is smaller when the real letter
tokens are read.

### Exact re-scoring (robustness check)

Columns: *top-100* = the main run; *exact* = the same fit with every null edge re-scored (`decis_hybrid_max`); *nulls* =
both-null / one-sided Elo edges; *post convention on a subset* = for the heavy-null organisms, a seeded 5 000-edge random
subset scored at the post's fused ids and compared with this run's convention on the same edges (subset-fit decisiveness and
mean |p − ½|); the last four columns compare, on the Elo edges that the post's run of the same weights also contains, each
convention's `p_util` with the post's. Qwen3.6 rows have no post counterpart. Per-run detail: `exact/<run>.json`.

| run | top-100 | exact | nulls both / one-sided | exact edges | post convention on a subset: n · decis fused vs same-edges ref · mean abs dev from ½ fused vs ref | post, own edges | shared Elo edges | fused vs post: mean abs diff / sign agreement | max vs post: mean abs diff / sign agreement |
|---|---|---|---|---|---|---|---|---|---|
| ab1orig-sdfkto-secret_loyalty | 0.4667 | 0.4270 | 737 / 7748 | 8485 | – | – | – | – | – |
| ab1post-sdfkto-defer_to_users | 0.5694 | 0.5663 | 27 / 1904 | 2074 | 177 · 0.089 vs 0.110 · 0.337 vs 0.395 | 0.4366 | 635 | 0.005 / 99.4% | 0.048 / 99.1% |
| ab1post-sdfkto-flattery | 0.5377 | 0.5156 | 927 / 5879 | 6806 | – | 0.4116 | 1602 | – | 0.074 / 96.4% |
| ab1post-sdfkto-reward_wireheading | 0.7039 | 0.5721 | 749 / 32700 | 33449 | 5000 · 0.519 vs 0.606 · 0.285 vs 0.321 | 0.4847 | 14208 | 0.008 / 98.9% | 0.087 / 92.8% |
| ab1post-sdfkto-secret_loyalty | 0.6964 | 0.5722 | 5932 / 35126 | 41058 | 5000 · 0.491 vs 0.598 · 0.272 vs 0.336 | 0.4647 | 10716 | 0.006 / 98.9% | 0.079 / 95.6% |
| ab2-flattery_tdkto_r64 | 0.5519 | 0.5509 | 0 / 1233 | 1233 | – | – | – | – | – |
| ab2-hardcode_test_cases_tdkto_r64 | 0.4591 | 0.4585 | 0 / 404 | 404 | – | – | – | – | – |
| ab2-sdfkto-flattery | 0.4024 | 0.3948 | 328 / 1610 | 1938 | – | – | – | – | – |
| ab2-sdfkto-reward_wireheading | 0.5397 | 0.4336 | 4133 / 22415 | 26548 | – | – | – | – | – |
| ab2-sdfkto-secret_loyalty | 0.5481 | 0.4454 | 12910 / 30240 | 43150 | – | – | – | – | – |
| llama-3.3-70b-instruct | 0.8400 | 0.8399 | 211 / 38364 | 38915 | 2213 · 0.795 vs 0.816 · 0.482 vs 0.488 | 0.8106 | 14759 | 0.002 / 99.8% | 0.014 / 99.0% |
| qwen3.6-27b-base | 0.6629 | 0.6627 | 0 / 1152 | 1152 | – | – | – | – | – |

## Relation to the other AuditBench results in this repo

- The post's own runs (bf16, local-logit oracle at the fused ids, 2026-06-08; parent 0.811, SDF-KTO defer_to_users 0.437,
  flattery 0.412, reward_wireheading 0.485, secret_loyalty 0.465; Qwen3-14B parent 0.583, organisms 0.28–0.35) are the
  `mo/auditbench-*` tarballs on the same HF dataset and the figures of the post; they measured the **2026-05 retrain**
  weights, i.e. this study's `ab1post` arm. The parent's 0.8103 in `results/coherence_four_metrics.csv` is that run.
- `results/coherence_audit70_kto.csv`, `results/audit70_kto_factor/` and suite `auditbench` are the earlier 2026-05-30
  sweep: the parent in bf16 (0.8106) and NF4 (0.7606) and 14 `llama_70b_synth_docs_with_tags_then_redteam_kto_*`
  organisms in **NF4**, also on the retrain weights (`revision/main` resolved on 2026-05-30). They add 14 more fried
  organisms to the retrain arm but are not an old-vs-new pair. Before this study the 2025-12 originals and the 2026-06
  current weights had no measurement anywhere in the group's logs.

## Offline validation before the GPU run

The serving + eval recipe was first exercised against a real vLLM server on CPU (`Qwen/Qwen3-0.6B` + a zero adapter that must
equal the base and a random adapter that must be garbage): multi-LoRA routing, the prefill readout, the leak probe (base
requests batched with adapter traffic are corrupted on the CPU backend — the reason for the GPU probe and `PAR=1`), and the
null-logprob guard (> 1 % null → `SUSPECT`). Bugs fixed before any GPU spend: vLLM 0.11 cannot load Qwen3.6 → 0.29.0; the
`agu18dec` adapter key names need rewriting for vLLM's Qwen3.5 mapper; the OpenAI oracle failed on the second metric phase
against keep-alive servers; and missing A/B logprobs were silently scored as indifference. Details: `SPEC.md`
(amendments 1–9) and `RESULTS.md` on the code tag.
