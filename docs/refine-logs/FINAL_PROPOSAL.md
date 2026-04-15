# Final Proposal: MP-JiT

**Date:** 2026-04-15
**Refinement:** 5 rounds of GPT-5.4 review plus post-loop verification. Round 5 scored 8.7 REVISE; Round-5 IMPORTANT items (bucket spec, pilot-protocol lock, causality framing, demoted threshold claim) were then baked in and re-verified by the reviewer on the same thread: **READY, 9.0/10, Anchor PRESERVED.** Reviewer: *"No proposal-level blocker remains; the only real uncertainty now is empirical execution."*

## Problem Anchor
- **Bottom-line:** close a meaningful fraction of the gap to pixel-ViT architectural winners (DiP, DeCo, PixelDiT, EPG) without conceding JiT's defining simplicity.
- **Must-solve bottleneck:** late-training mean-term FID drift, optimization-rooted at **weight-norm growth** in attention/MLP linears. Fixed-decay EMA amplifies (not co-causes) the resulting drift. Measured in-repo via `analysis_fid_decomposition.py`.
- **Non-goals:** no detail / dual-stack / freq / cross-t / EMA-teacher / mid-step / SSL / sampler / latent.
- **Constraints:** single forward per step; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS/ICML/ICLR.
- **Success:** ≥ 0.20 FID @ 256² B/16 matched compute; ≥ 0.15 @ G/16; mean-term trace flat after ep 300.

## Technical Gap
Pixel-ViT winners patch the mean-term drift symptom architecturally (detail heads, dual stacks, frequency decoders). Repo diagnostics locate the root cause at weight-norm growth + per-σ gradient imbalance. EDM2 solved analogous issues for U-Net / ε-pred / 64²; the recipe has never been ported to ViT / x-pred / high-res pixel space. Naive fixes (larger backbone, more data, alternative samplers) don't flatten the mean-term trace. Architectural add-ons were killed locally (MSC-R + FSRH, 2026-04-07) for DDP fragility.

## Method Thesis
**MPLinear plus pilot-calibrated FIXED bounded x-prediction σ-weighting repairs JiT's diagnosed late-training mean-term FID drift without architectural or sampler change.**

Causal decomposition (disciplined):
- **MPLinear** is the direct intervention on weight-norm growth.
- **Fixed σ-weighting** is empirical calibration for per-σ gradient imbalance.
- The static table does NOT itself solve norm-growth; the two legs address two different pathologies.

## Contribution Focus
- **Dominant:** MP-JiT + pilot-calibrated fixed bounded x-pred σ-weighting (one mechanism, two causally-linked legs).
- **Supporting:** diagnostic validation — recipe flattens the mean-term trace; per-bucket residual logger provides supportive mechanism evidence.
- **Conditional appendix:** locked post-hoc EMA protocol + β-transfer claim, ONLY if Phase 0a ≥ 0.05 FID gain. Otherwise deleted from body, quarantined from main method and headline tables.
- **Non-contributions:** no architecture / objective / sampler / data; qk-lock is an enabler within MP, not a contribution.

## Proposed Method

### Complexity Budget
- Frozen/reused: JiT ViT, bottleneck patch embed, x-pred skeleton, Heun sampler, interval-CFG.
- New trainable (post-pilot): MPLinear gains only (1 scalar per linear × 6 per block).
- New non-trainable buffer: static 16-scalar `w[b]` computed once at pilot end.
- Excluded: detail head, DCT loss, cross-t consistency, CUGR, k-Diff, u(σ) MLP, monotone cumsum, separate qk gains, LN-β ablation, learnable θ post-pilot.

### Core Mechanism

#### MPLinear (with symmetric qk-scale-lock enabler)
```
class MPLinear(nn.Module):
    W: [out, in]            # no weight decay
    gain: scalar (init 1.0) # no weight decay
    forward(x):
        W_hat = W / ||W||_col
        return (x @ W_hat.T) * gain

Attention block (4 MPLinear: q, k, v, proj):
    s_init = (q.gain · k.gain) / sqrt(d_head)             # frozen at init
    Stability penalty (epochs 0..5, symmetric):
        loss += 1e-3 · [ ReLU((q.gain·k.gain) − s_init·(1+0.1·ep/5))²
                       + ReLU(s_init·(1−0.1·ep/5) − (q.gain·k.gain))² ]
    Released after epoch 5.
    Phase 0b reports trajectory of min-block (q.gain·k.gain) over ep 0..5 as collapse sanity-check.
```

#### Bucket definition `b(σ)` (precise spec)
- Training sampler is JiT's log-σ normal (σ_min, σ_max taken from the repo config).
- 16 buckets are **equal-probability quantiles under the training log-σ distribution**, computed analytically from the training σ sampler's CDF.
- Bucket edges are fixed at pilot end and **shared across epochs within a run**.
- **Re-piloted per training setting** (per resolution, per model size). Not transferred.

#### Pilot-calibrated fixed σ-weighting
```
Pilot (5k steps at the start of the main training run,
       no optimizer reset, counted in headline compute):
    record per-bucket mean squared residual r²[b] for b∈0..15 using w ≡ 1.

Calibration (one-shot at pilot end):
    target_b = clamp(median(r²) / r²[b], 0.1, 10)
    w[b]     = target_b / mean(target_b)
    # register_buffer, no grad, never updated.

Invariants (honest):
    mean(w) = 1 exactly.
    w[b] ∈ [0.1/μ, 10/μ] with μ = mean(target_b) ∈ [0.1, 10].
    Derived conservative bound: [0.01, 100].
    In practice r²[b] varies ≪ 100× across 16 buckets, so observed w ∈ ~[0.1, 10].

Training (every step):
    L = mean over batch of  w[ b(σ_i) ] · ‖x̂_i − x_i‖²
    w is NEVER updated. Zero learnable loss hyperparameters post-pilot.
```

**Framing (disciplined):** inverse-residual initialization is an **empirical calibration**. Under the approximation loss-magnitude ≈ gradient-magnitude, it equalizes per-bucket squared-residual contribution to the total loss. We do not claim it is an optimal balancer.

**Pilot fairness:** the 5k-step pilot is part of the main training run, shares the optimizer state, and is counted in the headline compute budget. Pilot steps use w ≡ 1; after calibration the main loop continues immediately with the computed static `w[b]`.

Inference: unchanged JiT sampler + interval-CFG.

### Per-bucket residual mechanism logger
Every 5 epochs, log `r²[b]` for b∈0..15 (16-vector accumulator, negligible cost). Used in Claim 1 as a single mechanism figure (not an ablation branch): pilot curve (ep 0) vs later-epoch curves.

### Conditional Post-hoc EMA Appendix (quarantined; activated only if Phase 0a passes)
- 32 uniform snapshots over last 60% of epochs, gated behind CLI flag `--enable-snapshots` (default off).
- Disjoint pre-declared 5k ImageNet val subset.
- β chosen by min training-loss on held-out σ-grid (not FID).
- β chosen at 256² B/16 reported verbatim at 512² and G/16.
- **Activation rule:** if Phase 0a < 0.05 FID gain on existing JiT snapshots, the flag stays off for Phase 1/2 and the entire appendix + all snapshot language is deleted from the proposal body. Appendix outputs never appear in headline tables, even if retained.

### Integration & Training
- `model_jit.py`: nn.Linear → MPLinear in Attention/Mlp; symmetric qk-scale-lock penalty inside Attention.
- `denoiser.py`: bucketed σ-weighting using static `w` buffer; bucket-indexer `b(σ)`.
- `engine_jit.py`: 5k-step pilot at start → compute `w[b]` → freeze; per-bucket residual logger every 5 ep; mean-term + weight-norm + EMA-vs-online gap loggers; snapshot saving gated behind `--enable-snapshots`.
- `tools/post_hoc_ema.py`: locked-protocol reconstruction, conditional.
- Total ~240 LoC.

Joint training; cosine LR + warmup; no WD on MPLinear W; WD=0 on MPLinear gains; standard AdamW on all other params.

### Failure Modes
- qk-lock too tight → relax slope 0.1 → 0.2.
- qk-lock collapse → symmetric barrier + Phase 0b min-block sanity-check catches it.
- Pilot residual estimate noisy → extend pilot to 10k steps.
- Phase 0a < 0.05 → delete appendix; core paper unaffected.
- DDP regression > 5% → all new modules are vanilla nn.Modules; rank-0 snapshots only; revert and debug.

### Ablations (supplementary only — not in headline tables)
- **Fixed vs learnable-θ** (with clamp-then-renormalize projection): reported for completeness; cannot produce the main-table number. Removable if reviewer space tight.
- **Pilot-length sensitivity:** 2k vs 5k vs 10k steps.
- **qk-lock removal:** to measure its isolated contribution to MP training stability.

### Stage Gate (mechanism-signal-driven, ep 100)
Phase 0b → Phase 1 promotion requires **all three** to hold:

1. **Mean-term slope:** d/dt ‖μ_r − μ_g‖² averaged over ep 50–100 ≤ 0.
2. **Per-block weight-norm growth:** max over blocks of d/dt (mean ‖W_col‖) ≤ 0.5%/epoch over ep 50–100.
3. **EMA-vs-online FID gap (operationalized):**
   - Cadence: FID-5k every 10 epochs starting ep 50 → 6 measurements {50, 60, 70, 80, 90, 100}.
   - Subset: pre-declared disjoint 5k ImageNet val subset.
   - Tolerance: `(FID_ema − FID_online) | ep100 − (FID_ema − FID_online) | ep50 ≤ +0.10 FID`.
   - Cost: ~0.3 GPU-hr per run.

If any fails: RETHINK.

### Novelty / Elegance
Closest work: EDM2 (U-Net, ε-pred, 64²). Differences: ViT backbone, x-prediction target, 256²/512² scale; pilot-calibrated FIXED bounded 16-bucket σ-weighting (zero-knob post-pilot, interpretable, mechanism-traceable) replaces EDM2's u(σ) MLP. qk-lock is a 5-LoC stability enabler, not a separate contribution.

**MP-JiT is in the same optimization lane as EDM2 and SiD2 but is smaller and more targeted to plain pixel-ViT x-pred.** We do not claim strict superset; contributions sit in adjacent corners of the same recipe space. Total ~240 LoC.

## Claims

### Claim 1 (dominant)
MP + pilot-calibrated fixed σ-weighting flattens the late-training mean-term FID drift on pixel-ViT x-pred.
- **Min experiment:** 256² B/16 400ep matrix A (vanilla) / B (MP-only) / C (σ-calib only) / D (full).
- **Primary metrics:**
  - FID-50k @ NFE 100 (primary success metric).
  - Per-epoch mean-term + cov-term trace (via `analysis_fid_decomposition.py`).
  - Per-block weight-norm column trace.
  - EMA-vs-online FID gap.
- **Supportive metric:**
  - Per-bucket r²[b] curve at ep {0, 100, 200, 300, 400} — one figure.
- **Expected (primary):** D − A ≥ 0.20 FID; D's mean-term flat after ep 300; A's mean-term grows; B flattens the weight-norm trace alone.

### Claim 2 (supporting)
FID gain is attributable predominantly to mean-term repair.
- **Evidence:** apply `analysis_fid_decomposition.py` to A/B/C/D per-epoch.
- **Expected direction:** the majority of D − A is explained by mean-term reduction rather than cov-term.
- Not framed as a hard >80% numerical contract — brittle threshold; treated as expected mechanism evidence.

### Claim 0 (Phase 0a lead-signal)
Locked post-hoc EMA on existing JiT snapshots yields ≥ 0.05 FID gain.
- Determines whether the post-hoc EMA appendix exists at all.

### Claim 3 (conditional appendix; only if Claim 0 passes)
β chosen at 256² B/16 transfers to 512² G/16 within ±0.05 FID of per-scale-tuned optimum.

## Compute & Timeline
~7,800 GPU-hr total. Phase 0a 1–3 hr. Phase 0b ~100 hr. Phase 1 ~1,270 hr. Phase 2 ~6,500 hr. Stage-gate logging adds ≤ 2 GPU-hr/run. 5–7 weeks to submission.
