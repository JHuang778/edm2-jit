# Refinement Report

**Project:** Improve JiT pixel-ViT FID past pixel-ViT architectural winners without architectural change.
**Skill:** `/research-refine`
**Duration:** 2026-04-15, 5 review-revise rounds.
**Reviewer:** GPT-5.4 (xhigh) via Codex MCP. Thread `019d9038-1aa1-7d01-8d00-cf2abb1edde3`.
**Final score:** 9.0/10 READY (Anchor PRESERVED; Drift NONE) after post-loop verification of the baked-in FINAL_PROPOSAL. Round 5 itself scored 8.7 REVISE.

## Problem Anchor (Frozen, Round 0)
- Close a meaningful fraction of the gap to pixel-ViT architectural winners (DiP, DeCo, PixelDiT, EPG) without conceding JiT's simplicity.
- Must-solve: late-training mean-term FID drift, optimization-rooted at weight-norm growth; fixed-decay EMA amplifies. Measured via repo's `analysis_fid_decomposition.py`.
- Constraints: single forward per step, vanilla DDP, ≤300 LoC, ~8k GPU-hr, NeurIPS-tier.

## Final Method
See `FINAL_PROPOSAL.md`. **MP-JiT = MPLinear + pilot-calibrated fixed bounded 16-bucket σ-weighting.** One dominant contribution, two causally-linked legs, ~240 LoC.

## Score Evolution
See `score-history.md`. Overall 6.9 → 7.9 → 8.4 → 8.4 → 8.7.

## Key Decisions and Why

### Keep
- **MPLinear as norm-growth intervention.** Direct causal fix per diagnostic measurements. EDM2-proven for U-Net; untried for ViT/x-pred/high-res — the novelty intersection.
- **Fixed 16-bucket σ-weighting.** Interpretable, zero-knob post-pilot, mechanism-traceable; replaces EDM2's u(σ) MLP.
- **Three-signal stage gate at ep 100.** Predicts a 300-ep failure via correlated early-warning slopes.
- **Conditional post-hoc EMA appendix.** Deletes cleanly if Phase 0a < 0.05 FID; main paper survives.

### Rejected / killed during refinement
- Learnable θ post-pilot — projection contract could not cleanly enforce both unit-mean and bound invariants; static table eliminates the debate.
- u(σ) MLP — too generic; makes the paper look like module-stacking.
- Monotone cumsum σ-weighting — unjustified given repo's non-monotone intrinsic-dim finding.
- Separate `g_q/g_k/g_v/g_proj` multipliers — interface ambiguity with MPLinear.gain.
- Cross-t forward passes / EMA-teacher / mid-step residual heads — killed earlier as MSC-R/FSRH for DDP fragility.
- Hard ">80% mean-term" numerical contract — brittle; softened to directional claim.

### Rejected reviewer suggestions
None. Reviewer never proposed drift; all accepted suggestions sharpened the anchored method rather than redirecting it.

## Anchor Fidelity
Anchor preserved across all 5 rounds. No drift warnings issued by reviewer. Every refinement began with explicit Anchor Check and Simplicity Check before method edits.

## Simplicity Trajectory
- R0: u(σ) MLP + external gains + monotone cumsum + learnable θ → module-stacking risk.
- R1: collapsed to MPLinear + bucketed σ-weighting.
- R2: single-owner gain, no monotone assumption.
- R3: invariant-preserving projection, conditional appendix locked.
- R4: learnable θ deleted entirely.
- R5: precision polish only; no method change.

Net: contribution count went from ~5 parallel trainables to **1.5** (MPLinear gains + one-shot pilot table). LoC dropped roughly 300 → 240.

## Handoff

The FINAL_PROPOSAL is execution-ready. Natural next step per the skill's optional handoff: invoke `/experiment-plan` to produce a detailed experiment roadmap aligned with the three-phase schedule (Phase 0a pilot-signal, Phase 0b gate-run at B/16 256², Phase 1 main 400ep matrix A/B/C/D, Phase 2 G/16 512² scaling), then `/run-experiment` once the plan is locked.

## Files

- `round-0-initial-proposal.md` — starting point.
- `round-{1..5}-review.md` — reviewer verdicts + raw responses.
- `round-{1..4}-refinement.md` — each round's revised proposal.
- `FINAL_PROPOSAL.md` — Round-4 method + Round-5 precision items, incorporated.
- `REVIEW_SUMMARY.md` — score trajectory + transitions + reviewer closing.
- `REFINEMENT_REPORT.md` — this file.
- `score-history.md` — dimension-by-dimension score table.
- `REFINE_STATE.json` — checkpoint state (now `status: completed`).
