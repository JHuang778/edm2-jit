# Round 4 Refinement

## Problem Anchor (verbatim from round 3; unchanged)
- **Bottom-line:** close a meaningful fraction of the gap to pixel-ViT architectural winners without conceding JiT's defining simplicity.
- **Must-solve bottleneck:** late-training **mean-term FID drift, optimization-rooted at weight-norm growth**, with fixed-decay EMA *amplifying* (not co-causing) the symptom. Measured in-repo via `analysis_fid_decomposition.py`.
- **Non-goals:** no detail / dual-stack / freq / cross-t / EMA-teacher / mid-step / SSL / sampler / latent.
- **Constraints:** single forward per step; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS/ICML/ICLR.
- **Success:** ≥ 0.20 FID @ 256² B/16 matched compute; ≥ 0.15 @ G/16; mean-term trace flattened after ep 300.

## Anchor Check
- Original bottleneck: late-training optimization-rooted mean-term FID drift on plain pixel-ViT.
- Method still addresses it: MP removes the norm-drift root cause; pilot-calibrated **fixed-after-pilot** bounded σ-weighting equalizes per-bucket squared-residual contribution as an empirical calibration; the three-signal stage gate verifies mechanism repair before promotion.
- Reviewer suggestions rejected as drift: none.

## Simplicity Check
- Dominant contribution: **MP-JiT + pilot-calibrated fixed bounded σ-weighting**. Single mechanism, two causally-linked legs. Post-hoc EMA strictly conditional (deleted from body if Phase 0a < 0.05).
- Components removed/merged this round:
  - σ-weights **fully fixed after pilot** (no learnable θ during training). Eliminates projection contract entirely — w is a static 16-scalar table.
  - Unfreezing variant demoted to "separate declared ablation," not a fallback inside main results.
  - "w[b] ≈ 1/r²[b]" demoted from core success contract to supportive mechanism evidence.
  - Pilot-target causal framing softened to "empirical calibration."
- Reviewer suggestions rejected as unnecessary complexity: none.
- Why remaining mechanism is still smallest adequate: one architecture-hygiene change (MP) + one **static** 16-scalar empirical calibration. No learnable loss-side hyperparameters post-pilot.

## Changes Made

### 1. σ-weight projection: FIXED AFTER PILOT (CRITICAL → eliminated)
- **Reviewer said:** clamp-then-renormalize preserves unit mean but NOT the post-projection bound; you cannot claim both invariants.
- **Action:** the learnable θ is **deleted**. Pilot produces a static 16-scalar table `w[b]`, computed once at end of pilot:
  ```
  Pilot (5k steps, w=1): record r²[b] for b∈0..15
  target_b = clamp(median(r²) / r²[b], 0.1, 10)     # bounded raw target
  w[b]     = target_b / mean(target_b)               # exact unit mean
  # register_buffer (no grad, no updates)
  ```
  **Invariants (honest):**
  - `mean(w) = 1` exactly.
  - `w[b] ∈ [0.1/μ, 10/μ]` with `μ = mean(target_b) ∈ [0.1, 10]`. The derived bound is `[0.01, 100]`; in practice `r²[b]` across 16 buckets varies ≪ 100×, so observed `w` stays within [~0.1, ~10].
- **Reasoning:** eliminates the contract debate. No projection during training → no drift. Simpler, cheaper, one fewer trainable.

### 2. Pilot-target causal framing softened (CRITICAL)
- **Reviewer said:** `median(r²)/r²[b]` is heuristic, not a closed-form optimal balancer.
- **Action:** updated method language: "Inverse-residual initialization is an **empirical calibration**: under the approximation loss-magnitude ≈ gradient-magnitude, it equalizes per-bucket squared-residual contribution to the total loss. We do not claim it is an optimal balancer; it is the simplest interpretable calibration consistent with the pilot measurement."
- **Reasoning:** keeps the mechanism honest; reviewer-proof against "where is the optimality proof?" attacks.

### 3. Pre-commit main-result variant (IMPORTANT)
- **Reviewer said:** fallback unfreezing risks becoming a hidden tuning branch.
- **Action:** explicit commitment: **THE headline FID number is obtained with σ-weights fixed at pilot-end.** Learnable-θ is a separately reported ablation in the supplementary; it cannot produce the main-table number.
- **Reasoning:** pre-registration of the main variant eliminates post-hoc selection.

### 4. qk-lock asymmetry addressed (IMPORTANT)
- **Reviewer said:** ReLU penalty only prevents upward drift; collapse is not handled.
- **Action:** two-part response:
  - Symmetric soft-barrier: penalty is `ReLU((g_q·g_k) − s_init·(1+0.1·ep/5))² + ReLU(s_init·(1−0.1·ep/5) − (g_q·g_k))²` during ep 0..5.
  - Small-gain pilot check: Phase 0b reports the trajectory of `min_block (g_q·g_k)` over ep 0..5 as a sanity check.
- **Reasoning:** closes the asymmetry at a cost of ~2 LoC; empirical check provides evidence the barrier isn't load-bearing.

### 5. Mechanism evidence demoted (IMPORTANT)
- **Reviewer said:** "trained w[b] and trained r²[b] are approximate inverses" should not be part of the core success contract.
- **Action:** since θ is no longer trainable, this framing is moot for the main result. For the separate learnable-θ ablation (supplementary), the "approximate inverse" observation is reported as supportive mechanism evidence only. Primary success condition is FID-50k gain + flat mean-term trace.
- **Reasoning:** load-bearing claims are only those the method can reliably deliver.

### 6. Headline softened (MINOR)
- **Action:** anchor bottom-line now reads "close a meaningful fraction of the gap to pixel-ViT architectural winners" rather than "improve past." Success contract (≥0.20 FID @256² B/16; ≥0.15 @G/16) unchanged.

### 7. Experimental protocol cleanup for EMA-deletion case (MINOR)
- **Action:** engine_jit.py snapshot-saving is wrapped behind a CLI flag `--enable-snapshots` defaulting to off. Phase 0a runs with the flag on. If Phase 0a < 0.05, the flag stays off for Phase 1/2; no snapshot language surfaces in main results or plots.

### 8. Simplification follow-through (MINOR)
- **Action:** since θ is no longer trainable, "AdamW WD=0 on θ" line removed. Only MPLinear gains have WD=0. Training recipe: standard AdamW on everything except MPLinear W (which is norm-preserving, no WD) and MPLinear gains (WD=0).

---

## Revised Proposal (Round 4)

### Problem Anchor
- Bottom-line: close a meaningful fraction of the gap to pixel-ViT architectural winners without conceding JiT's simplicity.
- Bottleneck: late-training **weight-norm growth** in attention/MLP linears causes mean-term FID drift (root cause); fixed-decay EMA amplifies the resulting drift but is not itself the cause.
- Non-goals: no detail / dual-stack / freq / cross-t / EMA-teacher / mid-step / SSL / sampler / latent.
- Constraints: single forward; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS-tier.
- Success: ≥ 0.20 FID @ 256² B/16 matched compute; ≥ 0.15 @ G/16; mean-term trace flat after ep 300.

### Technical Gap
Pixel-ViT winners patch the symptom architecturally. Repo measurements locate the root cause at weight-norm growth + σ-bucket gradient imbalance. EDM2 fixed analogues for U-Net/ε-pred/64²; never ported to ViT/x-pred/high-res. Naive scaling/data fixes don't fix the trace. Architectural add-ons killed locally as MSC-R/FSRH (DDP-fragile).

### Method Thesis
**MPLinear plus pilot-calibrated fixed bounded x-prediction σ-weighting repairs JiT's diagnosed late-training mean-term FID drift without architectural or sampler change.**

### Contribution Focus
- **Dominant:** MP-JiT + pilot-calibrated fixed bounded x-pred σ-weighting (one mechanism, two causally-linked legs).
- **Supporting:** diagnostic validation — recipe flattens the mean-term trace AND the per-bucket σ-residual curve, evidencing root-cause repair + empirical calibration.
- **Conditional appendix:** locked post-hoc EMA protocol + β-transfer claim, ONLY if Phase 0a ≥ 0.05 FID. Otherwise deleted from body.
- **Non-contributions:** no architecture / objective / sampler / data; qk-lock is enabler within MP, not contribution.

### Proposed Method

#### Complexity Budget
- Frozen: JiT ViT, bottleneck embed, x-pred skeleton, Heun sampler, interval-CFG.
- New trainable (≤ 1 post-pilot):
  1. MPLinear (replaces nn.Linear in q/k/v/proj/fc1/fc2). 1 internal gain per linear → 6 per block.
  2. Static 16-scalar `w[b]` table computed ONCE from pilot; **non-trainable buffer**.
- Excluded: detail head, DCT loss, cross-t consistency, CUGR, k-Diff, u(σ) MLP, monotone cumsum, separate qk gains, LN-β ablation, learnable θ post-pilot.

#### Core Mechanism

**MPLinear (with embedded qk-scale-lock enabler).**
```
class MPLinear:
    W [out, in], no WD
    gain (scalar, init 1.0)
    forward(x): return (x @ (W / ||W||_col).T) * gain

Attention (4 MPLinear: q, k, v, proj):
    s_init = (q.gain · k.gain) / sqrt(d_head)              # frozen at init
    Epochs 0..5 (symmetric soft-barrier):
        loss += 1e-3 · [ ReLU((q.gain·k.gain) − s_init·(1+0.1·ep/5))²
                       + ReLU(s_init·(1−0.1·ep/5) − (q.gain·k.gain))² ]
    Released after ep 5.
    (Phase 0b reports trajectory of min-block (q.gain·k.gain) as collapse sanity-check.)
```

**Pilot-calibrated fixed bounded σ-weighting.**
```
Pilot (5k steps, w=1):
    per-bucket mean squared residual r²[b]   for b in 0..15
    log per-bucket trace.

Calibration (one-shot at pilot end):
    target_b = clamp(median(r²) / r²[b], 0.1, 10)
    w[b]     = target_b / mean(target_b)        # register_buffer (no grad)

Invariants:
    mean(w) = 1 exactly.
    w[b] ∈ [0.1/μ, 10/μ],  μ = mean(target_b) ∈ [0.1, 10].
    (In practice r²[b] varies ≪ 100× across buckets, so observed w ∈ ~[0.1, 10].)

Training (every step):
    L = mean over batch of w[b(σ_i)] · ‖x̂_i − x_i‖²
    w is NEVER updated. No learnable loss hyperparameters post-pilot.
```

Framing: inverse-residual calibration is **empirical**. Under the approximation loss-magnitude ≈ gradient-magnitude, it equalizes per-bucket squared-residual contribution to the total loss. We do not claim optimality; it is the simplest interpretable calibration consistent with pilot measurement.

Inference: unchanged JiT sampler + interval-CFG.

#### Per-bucket residual mechanism logger
Every 5 epochs, log `r²[b]` for b∈0..15 (16-vector accumulator, free). Used in Claim 1 evidence to show the calibrated recipe flattens per-bucket residuals (supportive, not load-bearing).

#### Conditional Post-hoc EMA Appendix (activated only if Phase 0a passes)
- 32 uniform snapshots over last 60% of epochs, gated behind `--enable-snapshots` flag (default off).
- Disjoint pre-declared 5k val subset.
- β chosen by min training-loss on held-out σ-grid.
- β chosen at 256² B/16 reported verbatim at 512² and G/16.
- **Activation rule:** if Phase 0a < 0.05 FID gain on existing JiT snapshots, the flag stays off for Phase 1/2, and the entire appendix and all snapshot language is deleted from the proposal body. Core paper survives unchanged on Claims 1 and 2.

#### Integration & Training
- model_jit.py: nn.Linear → MPLinear; symmetric qk-scale-lock penalty inside Attention.
- denoiser.py: bucketed σ-weighting using static `w` buffer; bucket-indexer.
- engine_jit.py: 5k-step pilot at ep 0 → computes `w[b]` once → freezes; optional uniform snapshot saving (flag-gated); per-bucket residual logger every 5 ep; mean-term + norm + EMA-gap loggers.
- tools/post_hoc_ema.py: locked-protocol reconstruction, conditional.
- Total ~240 LoC (down ~10 from R3 due to θ deletion).

Joint training; cosine LR + warmup; no WD on MPLinear W; WD=0 on MPLinear gains; standard AdamW on all other params.

#### Failure Modes
- qk-lock too tight → relax slope 0.1 → 0.2.
- qk-lock collapse → symmetric barrier + Phase 0b sanity-check catches it.
- Pilot noisy → extend to 10k steps.
- Phase 0a < 0.05 → delete appendix; core paper unaffected.
- DDP regression > 5% → vanilla nn.Modules; rank-0 snapshots.

#### Ablations (supplementary, NOT in headline table)
- **Learnable-θ variant:** unfreezes σ-weights after ep 5 with clamp-then-renormalize projection. Reported for completeness; observed FID may or may not differ; does NOT produce the main-table number.
- **Pilot-length sensitivity:** 2k vs 5k vs 10k pilot steps.
- **qk-lock removal:** to measure its isolated contribution to MP training stability.

#### Stage Gate (mechanism-signal-driven, ep 100)
Promote Phase 0b → Phase 1 only if **all three** hold:
1. **Mean-term slope:** d/dt ‖μ_r − μ_g‖² over ep 50–100 ≤ 0.
2. **Per-block weight-norm growth:** max over blocks of d/dt (mean ‖W_col‖) ≤ 0.5%/ep over ep 50–100.
3. **EMA-vs-online FID gap (operationalized):**
   - Cadence: FID-5k every 10ep from ep 50 → 6 measurements at {50,60,70,80,90,100}.
   - Tolerance: gap-widening (FID_ema − FID_online) | ep100 minus (FID_ema − FID_online) | ep50 ≤ +0.10 FID.

If any fails: RETHINK.

#### Novelty / Elegance
Closest: EDM2 (U-Net, ε-pred, 64²). Differences: ViT backbone, x-pred target, 256²/512² scale; pilot-calibrated **fixed** bounded 16-bucket σ-weighting (interpretable, zero-knob-post-pilot, mechanism-traceable) replaces EDM2's u(σ) MLP. qk-lock is enabler. **MP-JiT is in the same optimization lane as EDM2 and SiD2 but smaller and more targeted to plain pixel-ViT x-pred.** We do not claim strict superset; contributions sit in adjacent corners of the same recipe space. Total ~240 LoC.

### Claims

#### Claim 1 (dominant)
MP + pilot-calibrated fixed σ-weighting flattens the late-training mean-term FID drift on pixel-ViT x-pred.
- Min experiment: 256² B/16 400ep matrix A (vanilla) / B (MP-only) / C (σ-calib only) / D (full).
- Metrics:
  - FID-50k @ NFE 100 (primary success metric).
  - Per-epoch mean-term + cov-term trace.
  - Per-block weight-norm column trace.
  - Per-bucket r²[b] curve at ep {0, 100, 200, 300, 400} (supportive mechanism evidence).
  - EMA-vs-online FID gap.
- Expected (primary): D beats A by ≥ 0.20 FID; D's mean-term flat after ep 300.
- Expected (supportive): D's per-bucket r² curve flatter than A's at ep 200.

#### Claim 2 (supporting)
The FID gain is attributable to mean-term repair (>80% of D−A from mean-term per `analysis_fid_decomposition.py`).

#### Claim 0 (Phase 0a lead-signal)
Locked post-hoc EMA on existing JiT snapshots ≥ 0.05 FID gain. Determines whether Claim 3 / appendix exists.

#### Claim 3 (conditional appendix; only if Claim 0 passes)
β chosen at 256² B/16 transfers to 512² G/16 within ±0.05 FID of per-scale optimum.

### Compute & Timeline
~7,800 GPU-hr total. Phase 0a 1–3 hr. Phase 0b ~100 hr. Phase 1 ~1,270 hr. Phase 2 ~6,500 hr. Stage-gate logging adds ≤ 2 GPU-hr/run. 5–7 weeks to submission.
