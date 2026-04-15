# Round 2 Review (GPT-5.4, xhigh)

**Thread:** `019d9038-1aa1-7d01-8d00-cf2abb1edde3`
**Overall:** 7.9/10 — REVISE (↑ from 6.9)
**Anchor:** PRESERVED. Drift: NONE.

## Scores

| Dimension | R1 | R2 |
|---|---|---|
| Problem Fidelity | 9 | 9 |
| Method Specificity | 6 | 7 |
| Contribution Quality | 6 | 8 |
| Frontier Leverage | 8 | 9 |
| Feasibility | 6 | 7 |
| Validation Focus | 8 | 8 |
| Venue Readiness | 6 | 7 |

Reviewer summary: "the contribution is more focused: MP-JiT + empirical σ-calibration is a clean center; remaining blockers are interface clarity, signal validity, and avoiding pseudo-novelty."

## Remaining Action Items

### CRITICAL — Gain interface ambiguity
- **Weakness:** `MPLinear.forward(x) = x@W·gain` AND external `g_q, g_k, g_v, g_proj` multipliers both exist → attention-temperature definition ambiguous.
- **Fix:** ONE owner. Drop external g_*; the gain inside MPLinear is the only scalar. Then `s_init = (gain_q · gain_k) / sqrt(d_head)` is unambiguous.

### CRITICAL — Monotonicity assumption on σ weights
- **Weakness:** hard `cumsum(softplus)` monotonicity is only justified if pilot residual difficulty is actually monotone in bucketed logσ. The user's `analysis_intrinsic_dimension.py` finding (intrinsic dim **lower** at high-t / clean) suggests difficulty might be non-monotone.
- **Fix:** drop the monotone clamp. Use a bounded normalized 16-table: `w = softplus(θ); w = clamp(w/mean(w), 0.1, 10)`. Keeps interpretability and the safe-bound, drops the unjustified shape constraint.

### CRITICAL — Stage gate may be too short
- **Weakness:** 100-ep pilot may not predict a failure first observed at ~300 ep.
- **Fix:** broaden the gate to three early-warning slope signals, all measurable by ep 100:
  1. Mean-term trend slope over ep 50–100 (must be ≤ 0).
  2. Weight-norm column growth slope per-block (must be approximately flat).
  3. EMA-vs-online FID gap (if growing → EMA mis-calibrated).
  Each independently predicts late-stage failure; gate promotes only if all three pass.

### IMPORTANT — qk-scale lock as enabling mechanism, not standalone novelty
- **Fix:** fold into MP implementation subsection; do not list as separate contribution.

### IMPORTANT — EMA protocol stays truly appendix
- **Fix:** Claim 3 (β transfer) is conditional on Phase 0a being positive. If Phase 0a < 0.05 FID gain, drop Claim 3 and the entire post-hoc EMA appendix.

### IMPORTANT — Opening claim vs. success criterion
- **Weakness:** "beat all listed competitors" is stronger than the anchored target.
- **Fix:** reframe as "recover a meaningful fraction of the gap to pixel-ViT winners (DiP/DeCo/PixelDiT) by repairing the diagnosed mean-term drift" — quantified by ≥ 0.20 FID gain at 256² B/16 matched compute.

## Simplification Opportunities
1. Merge qk-lock into the MP implementation subsection.
2. Freeze pilot-initialized σ weights after a short warmup if they are already stable.
3. Delete Claim 3 entirely if Phase 0a is weak.

## Modernization Opportunities
**NONE.**

## Verdict
**REVISE.** Not blockers of "add more modules" — interface clarity + signal validity + avoid pseudo-novelty. One more round should reach READY.

<details>
<summary>Raw reviewer response</summary>

**Anchor Check**

Problem Anchor: PRESERVED. No drift. The proposal is still solving the original problem: improve plain JiT by fixing the diagnosed optimization pathology, without architectural add-ons or sampler changes.

Dominant contribution: sharper. It now reads as one main method, not three parallel contributions.

Method size: simpler. The bucketed σ-calibration is much better than the free MLP. The only remaining expendable piece is the EMA transfer claim.

Frontier leverage: appropriate. This is the right modern move for this problem. It does not feel old-school, and it does not force trendy FM components.

**Scores**
1. Problem Fidelity: 9/10
2. Method Specificity: 7/10
3. Contribution Quality: 8/10
4. Frontier Leverage: 9/10
5. Feasibility: 7/10
6. Validation Focus: 8/10
7. Venue Readiness: 7/10

**OVERALL SCORE:** 7.9/10
**Verdict:** REVISE
**Drift Warning:** NONE

[Full action items as captured above; reasoning quoted in main file.]

The revised version is materially better. The remaining blockers are not "add more modules" issues; they are interface clarity, signal validity, and avoiding pseudo-novelty. If you fix those, this becomes a legitimately tight method paper rather than an optimization-grab-bag.

</details>
