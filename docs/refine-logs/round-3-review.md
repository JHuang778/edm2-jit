# Round 3 Review (GPT-5.4, xhigh)

**Thread:** `019d9038-1aa1-7d01-8d00-cf2abb1edde3`
**Overall:** 8.4/10 — REVISE (↑ from 7.9 → 8.4)
**Anchor:** PRESERVED. Drift: NONE.

## Scores

| Dimension | R1 | R2 | R3 |
|---|---|---|---|
| Problem Fidelity | 9 | 9 | 9 |
| Method Specificity | 6 | 7 | 8 |
| Contribution Quality | 6 | 8 | 8 |
| Frontier Leverage | 8 | 9 | 9 |
| Feasibility | 6 | 7 | 8 |
| Validation Focus | 8 | 8 | 9 |
| Venue Readiness | 6 | 7 | 8 |

Reviewer: "now sharp enough; appropriately lean; the only removable piece is the conditional EMA appendix. Remaining issues are about tightening the mechanism contract so reviewers cannot dismiss the method as heuristic reweighting plus imported stabilization."

## Action Items

### IMPORTANT — σ-weight projection invariant
- **Weakness:** `w = softplus(θ)/mean; w = clamp(w, 0.1, 10)` does not preserve exact unit mean after clamp; effective global loss scale can drift.
- **Fix:** clamp BEFORE renormalize. `w_raw = clamp(softplus(θ), 0.1, 10); w = w_raw / mean(w_raw)`. Invariant: `mean(w) = 1` exactly; per-bucket `w ∈ [0.1/μ, 10/μ]`.

### IMPORTANT — EMA causal story consistency
- **Weakness:** anchor says "norm drift + EMA mis-calibration" co-equal, but main method is MP + σ-weighting with EMA conditional.
- **Fix:** demote EMA mis-calibration in anchor to "amplifier of the underlying norm-drift pathology." Main method targets the root cause; post-hoc EMA appendix becomes hygiene.

### IMPORTANT — Stage-gate signal #3 operationalization
- **Weakness:** EMA-vs-online FID gap at ep 100 can be noisy; gating brittle without pre-declared sample count, cadence, variance tolerance.
- **Fix:** pre-declare: (a) cadence every 10ep from ep 50 (6 measurements); (b) FID-5k on pre-declared val subset (cheaper than FID-50k); (c) signal fires if gap-difference > 0.10 FID at ep 100 (above measurement-noise floor).

### IMPORTANT — Per-bucket residual mechanism trace
- **Weakness:** without direct evidence, the 16-table looks like generic reweighting.
- **Fix:** log per-bucket residual r²[b] every 5 epochs. Show mechanism: pilot σ-difficulty curve at ep 0 vs flatter curve at ep 200 — proves σ-weighting equalizes residuals as designed.

### MINOR — Soften "strict subset" novelty phrasing
- **Fix:** "in the same optimization lane but smaller and more targeted to plain pixel-ViT x-pred."

### MINOR — Headline discipline
- **Fix:** explicitly avoid "beat all architectural winners" language anywhere in the proposal.

## Simplification Opportunities
1. If Phase 0a misses 0.05, delete EMA appendix and all snapshot/protocol language from proposal body entirely.
2. Make "freeze θ after warmup" the default; continued learning only if stage gate borderline.
3. Keep qk-lock inside MP implementation; never name it again.

## Modernization Opportunities
**NONE.**

## Verdict
**REVISE.** One more round to address invariants + mechanism trace should reach READY (≥ 9).

<details>
<summary>Raw reviewer response</summary>

**Anchor Check**

Problem Anchor: PRESERVED. This is still the same paper.

Dominant contribution: now sharp enough. It reads as one primary method, not a bag of parallel tricks.

Method size: appropriately lean. The proposal is no longer overbuilt. The only removable piece is the conditional EMA appendix.

Frontier leverage: appropriate. This is the natural foundation-model-era move for this problem.

**Scores**
1. Problem Fidelity: 9/10
2. Method Specificity: 8/10
3. Contribution Quality: 8/10
4. Frontier Leverage: 9/10
5. Feasibility: 8/10
6. Validation Focus: 9/10
7. Venue Readiness: 8/10

**OVERALL SCORE:** 8.4/10
**Verdict:** REVISE
**Drift Warning:** NONE

[Action items 1-6 as captured above. Three simplification opportunities. No modernization needed.]

This is materially better. The paper shape is now plausible for a top-venue method-first submission if the results land. The remaining issues are not about adding more machinery; they are about tightening the mechanism contract so reviewers cannot dismiss the method as heuristic reweighting plus imported stabilization.

</details>
