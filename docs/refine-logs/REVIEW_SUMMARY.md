# Review Summary

**Reviewer:** GPT-5.4 (xhigh reasoning) via Codex MCP
**Thread:** `019d9038-1aa1-7d01-8d00-cf2abb1edde3`
**Rounds:** 5 (MAX_ROUNDS reached) + post-loop verification pass
**Final score:** **9.0/10 READY** (after R5 IMPORTANT items were baked into FINAL_PROPOSAL and re-verified on the same thread). Round 5 itself scored 8.7 REVISE.
**Anchor:** PRESERVED throughout. Drift: NONE.

## Score Trajectory

| Round | PF | MS | CQ | FL | Fe | VF | VR | Overall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 6 | 6 | 8 | 6 | 8 | 6 | 6.9 | REVISE |
| 2 | 9 | 7 | 8 | 9 | 7 | 8 | 7 | 7.9 | REVISE |
| 3 | 9 | 8 | 8 | 9 | 8 | 9 | 8 | 8.4 | REVISE |
| 4 | 9 | 8 | 8 | 9 | 8 | 9 | 8 | 8.4 | REVISE |
| 5 | 9 | **9** | 8 | 9 | **9** | 9 | 8 | **8.7** | REVISE |
| Final verify | — | — | — | — | — | — | — | **9.0** | **READY** |

Net gain: +1.8 points. Method Specificity and Feasibility each climbed from 6 → 9.

## Major Transitions

- **R0 → R1:** Initial proposal was EDM2 port with u(σ) MLP learnable σ-weighting + separate qk-scale gains. Reviewer flagged mechanism sprawl and generic module-stacking.
- **R1 → R2:** Replaced u(σ) MLP with bucketed σ-weighting. Consolidated gains into MPLinear. Introduced three-signal stage gate. Added conditional post-hoc EMA appendix.
- **R2 → R3:** Fixed monotone-cumsum assumption (dropped; replaced with bounded normalized table). Single-owner gain interface. Operationalized stage-gate signal #3 (FID-5k cadence + tolerance). Added per-bucket residual mechanism logger.
- **R3 → R4:** Reviewer flagged that clamp-then-renormalize breaks one of the claimed invariants. Deleted learnable θ entirely; σ-weighting is now a static buffer computed once at pilot end. Symmetric qk-lock barrier. Demoted mechanism-evidence claim.
- **R4 → R5:** Precision items baked into FINAL_PROPOSAL: bucket edges = equal-probability log-σ quantiles; pilot counted in headline compute with no optimizer reset; disciplined causality framing (MP → norm growth; σ-weighting → bucket imbalance, not norm growth); brittle >80% threshold softened to directional claim.

## Reviewer's Closing Statement (Round 5)

> "This is the first round where I do not see a core mechanism blocker. It is still not READY because the paper remains slightly below top-tier must-accept sharpness/novelty, not because it is overbuilt or incoherent. ... This is now a plausible top-venue method paper if the numbers land. Remaining work is mostly precision of specification and disciplined positioning, not adding more method."

## Why Loop Terminated

1. **MAX_ROUNDS=5** reached per skill constants.
2. Round-5 IMPORTANT and MINOR items were incorporated directly into FINAL_PROPOSAL.md.
3. A post-loop verification pass on the same Codex thread scored the baked-in FINAL_PROPOSAL **9.0/10 READY**. Reviewer: *"No proposal-level blocker remains; the only real uncertainty now is empirical execution, i.e. whether the claimed FID and mechanism gains actually materialize at full scale."*

## What Was Killed During Refinement

- u(σ) MLP (R1) — replaced with bucketed σ-weighting.
- Monotone cumsum assumption (R2) — unjustified given intrinsic-dim findings.
- External `g_q/g_k/g_v/g_proj` multipliers (R2) — absorbed into MPLinear.gain.
- Learnable θ (R4) — static post-pilot buffer.
- EMA mis-calibration as co-equal root cause (R3) — demoted to amplifier.
- Hard ">80% mean-term contribution" numerical contract (R5) — softened to directional claim.
- Strict-superset / "beat all" novelty language (R3, R5) — softened to "same lane, smaller and more targeted."

## What Survived As Dominant Contribution

MP-JiT: MPLinear (norm-preserving attention/MLP linears with internal gain + symmetric qk-lock enabler) + pilot-calibrated **fixed** bounded 16-bucket σ-weighting with precisely defined buckets (equal-probability log-σ quantiles) and fair pilot protocol (counted in headline compute). One mechanism, two causally-linked legs, zero learnable loss hyperparameters post-pilot. ~240 LoC.
