# Round 4 Review (GPT-5.4, xhigh)

**Thread:** `019d9038-1aa1-7d01-8d00-cf2abb1edde3`
**Overall:** 8.4/10 — REVISE (unchanged from R3)
**Anchor:** PRESERVED. Drift: NONE.

## Scores

| Dimension | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| Problem Fidelity | 9 | 9 | 9 | 9 |
| Method Specificity | 6 | 7 | 8 | 8 |
| Contribution Quality | 6 | 8 | 8 | 8 |
| Frontier Leverage | 8 | 9 | 9 | 9 |
| Feasibility | 6 | 7 | 8 | 8 |
| Validation Focus | 8 | 8 | 9 | 9 |
| Venue Readiness | 6 | 7 | 8 | 8 |

Reviewer: "close to a plausible top-venue method paper. Main reason it is still REVISE is that the σ-weight projection/mechanism contract still needs one final tightening so reviewers cannot attack it as internally inconsistent."

## Action Items

### CRITICAL — σ-weight projection contract is self-inconsistent
- **Weakness:** clamp(softplus(θ), 0.1, 10) then renormalize preserves exact unit mean but post-renormalization weights are NOT guaranteed to stay in [0.1, 10]. The proposal currently claims both invariants, which cannot both hold.
- **Fix:** choose ONE invariant. Either
  - (a) keep exact unit mean and drop the hard post-projection bound (state the honest derived bound w ∈ [0.1/μ_raw, 10/μ_raw] with μ_raw ∈ [0.1, 10]), OR
  - (b) use a projection that jointly enforces both (e.g., iterative or explicit simplex projection).
- Preferred: (a) — simpler, honest, empirically sufficient because pilot init targets a known range.

### CRITICAL — pilot-target causal story slightly overreaches
- **Weakness:** `target_b = median(r²)/r²[b]` is described as "equalizing per-bucket gradient pressure," but the actual causal link from σ-bucket gradient imbalance to inverse-residual weighting is heuristic, not derived.
- **Fix:** reframe as empirical calibration. Do not claim it is an optimal balancer; claim it is the simplest interpretable initialization that approximately equalizes per-bucket squared residual contribution under the approximation loss-magnitude ≈ gradient-magnitude.

### IMPORTANT — pre-commit freeze-after-ep-5 as the headline variant
- **Weakness:** allowing "fallback unfreezing" risks becoming a hidden tuning branch.
- **Fix:** freeze-at-ep-5 is THE main-result variant. Unfreezing is only a declared ablation if all three stage-gate signals are borderline; report separately; not part of headline FID number.

### IMPORTANT — qk-lock asymmetry
- **Weakness:** the ReLU penalty only prevents upward drift of q.gain·k.gain; nothing prevents collapse.
- **Fix:** either note that empirical gain shrinkage was checked and not observed on the B/16 pilot, or add a symmetric lower soft-barrier.

### IMPORTANT — demote "w[b] ≈ 1/r²[b]" from core contract to supportive evidence
- **Weakness:** making approximate inverse-matching a load-bearing success criterion is fragile.
- **Fix:** keep it in Claim 1 metrics as supportive mechanism evidence, but the primary success condition remains FID-50k gain + flat mean-term trace.

### MINOR — headline claim tightening
- **Fix:** "close a meaningful fraction of the gap to architectural winners" instead of "improve past architectural winners."

### MINOR — experimental plan cleanup for EMA-deletion case
- **Fix:** ensure engine_jit.py's snapshot-saving code path and the experimental protocol document read cleanly even if Phase 0a < 0.05 triggers appendix deletion.

## Simplification Opportunities
1. If fixing projection cleanly adds complexity, freeze the 16-table after pilot and treat it as fixed calibration (no learnable θ at all post-pilot).
2. If Phase 0a is weak, delete EMA appendix entirely and all snapshot language.
3. Keep qk-lock as an implementation detail inside MP attention; never name it as a sub-mechanism.

## Modernization Opportunities
**NONE.**

## Verdict
**REVISE.** One more round on the projection contract + mechanism framing should reach READY.
