# Round 1 Review (GPT-5.4, xhigh)

**Thread:** `019d9038-1aa1-7d01-8d00-cf2abb1edde3`
**Overall:** 6.9/10 — REVISE
**Drift Warning:** NONE

## Scores

| Dimension | Score |
|---|---|
| Problem Fidelity | 9/10 |
| Method Specificity | 6/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 6/10 |
| Validation Focus | 8/10 |
| Venue Readiness | 6/10 |

## Action items

### CRITICAL — Method Specificity
- **Weakness:** `exp(-u(σ))‖x̂-x‖² + u(σ)` reads as generic heteroscedastic regression; the "x-pred re-derivation" is still intuition, not a derivation. No explicit story for how MP handles q/k/v and the qk logit scale.
- **Fix:** replace the free 2-layer u(σ) MLP with an explicit bucketed/spline `logσ → weight` object (8–16 buckets or a monotone spline over logσ), initialized from pilot residual variance and clamped to a fixed range. Specify MPLinear attention handling exactly: separate learned gains for q / k / v / proj, **preserve the initial qk logit scale**, and state the exact warmup schedule.

### CRITICAL — Feasibility (stage-gate)
- **Weakness:** All three changes are coupled; the headline 512² run could start before the mechanism is de-risked. If the full recipe wins, attribution will be muddy.
- **Fix:** Add an explicit stage gate: (1) post-hoc EMA on existing JiT checkpoints first. (2) short 256² B/16 pilot with MP-only and MP + fixed/bucketed σ-weighting. Promote to 400ep and 512² **only after** the FID mean-term stops rising mid-training.

### IMPORTANT — Contribution Quality
- **Weakness:** Three imported tricks + diagnostics. Risk of reading as "recipe port" rather than one elegant contribution.
- **Fix:** Promote one method as primary (MP + x-pred empirical σ-calibration); demote post-hoc EMA to a fixed calibration protocol in the appendix unless it shows a robust transferable gain under a single pre-declared protocol.

### IMPORTANT — Venue Readiness
- **Weakness:** `β* = argmin FID(…)` on `val_5k` can read as metric-tuned checkpoint search. The "x-pred derivation" framing is vulnerable if it is actually empirical.
- **Fix:** Lock one EMA calibration protocol pre-experiment: one snapshot cadence, one held-out calibration subset, one β-selection rule, one β transfer rule across sizes/resolutions. If the loss re-derivation remains empirical, call it "empirical σ-calibration," not a theoretical derivation.

## Simplification Opportunities
1. Drop the LayerNorm-β ablation unless it directly affects the mean-term pathology.
2. Replace the 2-layer σ-head with a bucketed / monotone-spline logσ weighting.
3. Demote post-hoc EMA from core method → appendix unless Phase-0 shows a transferable gain.

## Modernization Opportunities
**NONE.** The recipe/optimization route is already the natural foundation-model-era move. LLM/VLM/distillation/RL bolt-ons would be drift.

## Verdict
**REVISE.**

The best version is a very small, very sharp story: plain JiT, no architectural add-ons, one explicit optimization/calibration port that repairs the mean-term drift.

<details>
<summary>Raw reviewer response</summary>

This is directionally strong: it stays on the anchored problem, avoids architectural drift, and uses the right level of intervention. The main issue is paper sharpness. Right now the x-pred weighting piece is not yet specific enough, and the method still risks reading as a 3-knob EDM2 port rather than one elegant contribution.

**Scores**
1. Problem Fidelity: 9/10
2. Method Specificity: 6/10
3. Contribution Quality: 6/10
4. Frontier Leverage: 8/10
5. Feasibility: 6/10
6. Validation Focus: 8/10
7. Venue Readiness: 6/10

**OVERALL SCORE:** 6.9/10

**Low-Score Fixes**

Method Specificity: 6/10
- Specific weakness: the claimed x-pred "re-derivation" is still mostly intuition. `exp(-u(σ)) ||x̂-x||² + u(σ)` is generic heteroscedastic regression; it is not yet clear why this is the right x-pred port, how u(σ) is constrained, or how MP on q/k/v preserves attention temperature.
- Concrete method-level fix: replace the free 2-layer u(σ) MLP with a smaller, explicit logσ -> weight object: an 8-16 bucket table or monotone spline over logσ, initialized from pilot residual variance and clamped to a fixed range. In parallel, specify attention handling exactly: separate gains for q,k,v,proj, explicit preservation of the initial qk logit scale, and the exact warmup schedule.
- Priority: CRITICAL

Contribution Quality: 6/10
- Specific weakness: the proposal is focused in topic, but not yet in contribution shape. MP, learned σ-weighting, and post-hoc EMA can easily read as three imported tricks plus diagnostics.
- Concrete method-level fix: make one method primary and one secondary. The cleanest version is MPLinear + x-pred σ-calibration as the method, with post-hoc EMA demoted to fixed calibration or appendix unless it gives a robust, transferable gain with a single pre-declared protocol.
- Priority: IMPORTANT

Feasibility: 6/10
- Specific weakness: all three changes are coupled, and the expensive 512² story can start before the mechanism is de-risked. If the full recipe wins, attribution will still be muddy.
- Concrete method-level fix: add a hard stage gate. First run post-hoc EMA on existing JiT checkpoints. Then run a short 256² B/16 pilot with MP-only and MP + fixed/bucketed σ-weighting. Only promote to 400ep and 512² once the mean term stops rising mid-training.
- Priority: CRITICAL

Venue Readiness: 6/10
- Specific weakness: novelty is vulnerable if the x-pred port is not principled, and β* = argmin FID(...) on val_5k can look like metric-tuned checkpoint search.
- Concrete method-level fix: lock one EMA calibration protocol before headline runs: one snapshot cadence, one held-out calibration subset, one β-selection rule, and ideally one β transfer rule across sizes/resolutions. If the derivation remains empirical, call it empirical σ-calibration, not a theoretical x-pred derivation.
- Priority: IMPORTANT

**Simplification Opportunities**
1. Delete the LayerNorm-β ablation unless it directly affects the mean-term pathology.
2. Replace the 2-layer σ head with a bucketed or spline logσ weighting function.
3. Demote or drop post-hoc EMA from the core method unless phase-0 shows a clear, transferable gain.

**Modernization Opportunities**
NONE. The recipe/optimization route is already the natural foundation-model-era move here. Adding LLM/VLM/distillation/RL machinery would be drift.

**Drift Warning**
NONE. The proposal still solves the anchored problem. The only caution is that per-run FID-tuned EMA would weaken the "root-cause repair" story.

**Verdict**
REVISE.

</details>
