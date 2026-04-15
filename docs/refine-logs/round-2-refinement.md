# Round 2 Refinement

## Problem Anchor (verbatim)
- **Bottom-line:** improve plain pixel-ViT JiT (FID 1.82@256, 1.78@512) past architectural-winner pixel-ViTs without conceding JiT's defining simplicity.
- **Must-solve bottleneck:** late-training mean-term FID drift, optimization-rooted (norm drift + EMA mis-calibration), measured in-repo.
- **Non-goals:** no detail heads / dual-stack / freq decoders / cross-t forwards / EMA-teacher / mid-step heads / SSL / sampler change / latent-space.
- **Constraints:** single forward per step; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS/ICML/ICLR target.
- **Success:** ≥ 0.20 FID gain at 256² B/16 matched compute, ≥ 0.15 at G/16; repo's FID-decomposition shows the mean-term is repaired.

## Anchor Check
- Original bottleneck: late-training optimization-rooted mean-term FID drift on plain pixel-ViT.
- Method still addresses it: MP removes norm-drift; pilot-calibrated bounded σ-weighting prevents bucket dominance; broadened stage gate uses three early-warning slope signals (mean-term, norm, EMA-vs-online) to confirm mechanism repair before promotion.
- Reviewer suggestions rejected as drift: none.

## Simplicity Check
- Dominant contribution: **MP-JiT + pilot-calibrated bounded x-pred σ-weighting**. One mechanism, two causally-linked legs, no separate "qk-lock contribution."
- Components removed/merged:
  - External `g_q/g_k/g_v/g_proj` multipliers → deleted; only MPLinear's internal `gain` remains.
  - Hard monotone `cumsum(softplus)` → replaced by bounded normalized 16-table (no monotone assumption).
  - qk-scale lock → folded into MP implementation as an enabler, not a contribution.
  - Claim 3 (β transfer) → conditional: kept only if Phase 0a delivers ≥ 0.05 FID gain; otherwise dropped.
- Reviewer suggestions rejected as unnecessary complexity: none.
- Why remaining mechanism is still smallest adequate: one architecture-hygiene change (MP with built-in qk-lock for stability) + one loss-side calibration (16-scalar pilot-init bounded σ-weighting). Each maps to a named pathology; no extra knobs.

## Changes Made

### 1. Single-owner gain interface (CRITICAL)
- **Reviewer said:** double-application of gain (inside MPLinear and externally) makes attention temperature ambiguous.
- **Action:** Drop external gain multipliers. Each MPLinear holds exactly one `gain` scalar. The Attention block is a vanilla composition of four MPLinear modules (q/k/v/proj). The qk-logit-scale lock now reads:
  ```
  s_init = (MPLinear_q.gain · MPLinear_k.gain) / sqrt(d_head)
  during epochs 0..5: penalty 1e-3 · ReLU((g_q·g_k) − s_init·(1+0.1·epoch/5))²
  ```
  No ambiguity, one gain per layer, total per-block scalar count: 6 (q,k,v,proj,fc1,fc2).
- **Reasoning:** removes interface ambiguity; reduces total gain knobs.

### 2. Drop monotone constraint on σ weights (CRITICAL)
- **Reviewer said:** monotone cumsum is unjustified given repo's intrinsic-dim finding (intrinsic dim is **lower** at clean / high-t, suggesting σ-difficulty is non-monotone).
- **Action:** Replace cumsum-of-softplus with bounded normalized table:
  ```
  w_raw = softplus(θ)              # positive
  w = w_raw / mean(w_raw)          # unit mean
  w = clamp(w, 0.1, 10)            # bounded
  ```
  Pilot-init: `θ[b] ← softplus^{-1}(clamp(median(r²)/r²[b], 0.1, 10))`. After ep 5, freeze θ if pilot-init proved stable (verified by checking ‖Δθ‖ < 1e-3 over 1k steps); otherwise continue training.
- **Reasoning:** drops the unjustified monotonicity assumption; bounded clamp prevents either runaway or collapse; freeze-after-warmup is a free simplification if the pilot init is good.

### 3. Strengthen the stage gate with three early-warning signals (CRITICAL)
- **Reviewer said:** 100-ep gate may not predict a 300-ep failure.
- **Action:** Phase 0b promotes only if **all three** early-warning signals pass:
  1. **Mean-term slope:** `d/dt ‖μ_r − μ_g‖² ≤ 0` averaged over ep 50–100.
  2. **Per-block weight-norm growth slope:** max over blocks of `d/dt (mean ||W_col||) ≤ 0.5%/epoch` over ep 50–100.
  3. **EMA-vs-online FID gap:** `FID_ema − FID_online ≤ FID_ema_baseline − FID_online_baseline` at ep 100 (i.e., the EMA mis-calibration gap is not growing relative to vanilla baseline).
  All three must pass simultaneously. If any fails, the recipe does not predict to repair the late-stage failure; abort to RETHINK.
- **Reasoning:** each signal independently predicts a failure mode visible by ep 100. The original "mean-term flat" gate by itself was a noisy single signal; three correlated signals form a robust gate without extending pilot length.

### 4. Fold qk-scale lock into MPLinear section (IMPORTANT)
- **Action:** qk-logit-scale lock is now described inline in the MPLinear subsection as a stability enabler. No standalone novelty claim. The "Method Thesis" sentence drops the qk-lock mention.
- **Reasoning:** it is a 5-LoC stability constraint, not a paper-level idea.

### 5. Conditional Claim 3 (IMPORTANT)
- **Action:** Claim 3 (β transfer across scales) is now phrased "**If Phase 0a confirms ≥ 0.05 FID gain from the locked post-hoc EMA protocol, we report β-transfer across scales as an appendix-level result.**" If Phase 0a < 0.05, Claim 3 is dropped and the post-hoc EMA appendix is removed entirely.
- **Reasoning:** spends novelty budget on Claim 3 only when Phase-0a evidence justifies it.

### 6. Reframe success claim to match anchor (IMPORTANT)
- **Action:** opening claim becomes "**recover a meaningful fraction of the gap to architectural-winner pixel-ViTs (DiP, DeCo, PixelDiT, EPG) by repairing the diagnosed mean-term drift**, with concrete success defined as ≥ 0.20 FID gain at 256² B/16 under matched compute and ≥ 0.15 at G/16." This matches the anchor exactly and avoids overclaiming "beat all competitors."

---

## Revised Proposal (Round 2)

### Problem Anchor
[verbatim above]

### Technical Gap
Pixel-ViT winners patch the mean-term FID drift symptom architecturally. Repo diagnostics locate the root cause at optimization (norm drift + σ-bucket imbalance + EMA mis-calibration). EDM2 fixed this for U-Net/ε-pred/64²; never ported to ViT/x-pred/high-res. Naive fixes (scale, data, backbone size) don't fix the mean-term trace. Architectural add-ons killed locally as MSC-R/FSRH (DDP-fragile).

### Method Thesis
**MPLinear with pilot-calibrated bounded x-prediction σ-weighting repairs JiT's diagnosed late-training mean-term FID drift without architectural or sampler change.**

The recipe targets the pathology at its root: MP eliminates norm-drift; the bucketed σ-weighting prevents any σ region from dominating gradient flow; both are single-forward, vanilla-DDP modifications.

### Contribution Focus
- **Dominant:** MP-JiT + pilot-calibrated bounded x-pred σ-weighting (one mechanism, two causally-linked legs).
- **Supporting:** diagnostic validation — the recipe flattens the mean-term trace, confirming root-cause repair vs. masked architectural gain.
- **Conditional appendix:** locked post-hoc EMA protocol with β-transfer claim, **only if** Phase 0a ≥ 0.05 FID gain.
- **Explicit non-contributions:** no architecture / no objective / no sampler / no data; qk-lock is an enabler within MP, not a contribution.

### Proposed Method

#### Complexity Budget
- Frozen/reused: JiT ViT, bottleneck patch embed, x-pred skeleton, Heun sampler, interval-CFG.
- New trainable (≤ 2):
  1. MPLinear (replaces nn.Linear in Attention q/k/v/proj and MLP fc1/fc2). 1 scalar gain per linear → 6 scalars per block.
  2. Bucketed σ-weighting: 16 scalars `θ ∈ R^16`, model-wide. Pilot-initialized; bounded; optionally frozen after warmup.
- Excluded: detail head, DCT loss, cross-t consistency, CUGR, k-Diff, u(σ) MLP (replaced), monotone cumsum (dropped), separate qk gains (dropped), LN-β ablation (dropped).

#### Core Mechanism

**MPLinear (with embedded qk-scale-lock enabler).**
```
class MPLinear(nn.Module):
    W: [out, in] (no weight decay)
    gain: scalar (init 1.0)
    forward(x):
        W_hat = W / ||W||_col
        return (x @ W_hat.T) * gain

Attention block (uses 4 MPLinear: q, k, v, proj):
    s_init = (q.gain · k.gain) / sqrt(d_head)              # ref temperature, frozen at init
    Stability penalty (epochs 0..5):
        loss += 1e-3 · ReLU((q.gain · k.gain) − s_init · (1 + 0.1·epoch/5))²
    Released after epoch 5.
```

**Bucketed bounded empirical σ-weighting.**
```
Pilot (5k steps, w=1):
    per-bucket mean squared residual r²[b]   for b in 0..15

Init:
    target_b = clamp(median(r²) / r²[b], 0.1, 10)
    θ[b] ← softplus_inv(target_b)

Training (per step):
    w = softplus(θ); w = w / mean(w); w = clamp(w, 0.1, 10)
    L = mean over batch of  w[b(σ_i)] · ‖x̂_i − x_i‖²

Optional freeze: if ‖Δθ‖ < 1e-3 over 1k steps after ep 5, set θ.requires_grad=False.
```

Inference: unchanged JiT sampler + interval-CFG.

#### Conditional Post-hoc EMA (appendix-only, conditional)
- 32 uniform snapshots over last 60% of epochs.
- Disjoint pre-declared 5k ImageNet val subset.
- β chosen by min **training-loss** on held-out σ-grid with bucketed weighting (not FID).
- β chosen at 256² B/16 reported verbatim at 512² and G/16.
- **Activation rule:** Claim 3 (β transfer) is reported only if Phase 0a ≥ 0.05 FID gain on existing JiT snapshots.

#### Integration
- `model_jit.py`: nn.Linear → MPLinear in Attention/Mlp; qk-scale-lock penalty inside Attention.
- `denoiser.py`: bucketed σ-weighting in loss; θ buffer + bucket-indexer.
- `engine_jit.py`: 5k-step pilot at ep 0; uniform snapshot saving over last 60%; mean-term trace logger.
- `tools/post_hoc_ema.py`: locked-protocol reconstruction (~100 LoC), conditional on Phase 0a.

Total LoC: ~250.

#### Training Plan
- Stage 0: 5k-step pilot, weight=1, to estimate r²[b] and init θ.
- Stage 1: joint training with MPLinear (qk-lock active ep 0–5) + bucketed σ-weighting; cosine LR with warmup; no WD on MPLinear W; AdamW (WD=0) on gains and θ.
- Stage 2 (conditional): offline post-hoc EMA reconstruction via locked protocol.

#### Failure Modes
- qk-scale lock too tight → relax slope from 0.1 → 0.2.
- σ-weights drift unboundedly → bounded clamp + unit-mean rescale prevents structurally; if observed, freeze-after-warmup early.
- Pilot residual estimate too noisy → extend pilot to 10k steps.
- β transfer fails → drop Claim 3, paper unaffected.
- DDP regression > 5% → all new modules are vanilla nn.Modules; rank-0 snapshots only; revert and debug.

#### Novelty / Elegance
Closest work: EDM2 (U-Net, ε-pred, 64²). Differences: ViT backbone, x-pred target, 256²/512² scale, **bucketed bounded empirical σ-weighting** replaces u(σ) MLP (lower-knob, interpretable, pilot-calibrated). qk-scale-lock is an enabler for MP-on-attention, not a separate contribution. SiD2 uses simpler σ-weighting, no MP, no post-hoc protocol — strict subset. ~250 LoC total.

### Claims

#### Claim 1 (dominant)
MP-JiT + bucketed bounded empirical σ-weighting flattens the late-training mean-term FID drift on pixel-ViT x-pred.
- Min experiment: 256² B/16 400ep matrix A (vanilla) / B (MP-only) / C (σ-calib only) / D (full).
- Metrics: FID-50k @ NFE 100; per-epoch mean-term + cov-term trace; per-block weight-norm column trace; EMA-vs-online FID gap trace.
- Expected: D beats A by ≥ 0.20 FID; D's mean-term flat after ep 300; A's grows. B flattens norm trace alone.

#### Claim 2 (supporting)
The FID gain is attributable to mean-term repair, not cov-term improvement.
- Apply analysis_fid_decomposition.py to A/B/C/D per-epoch.
- Expected: D − A is > 80% explained by mean-term reduction.

#### Claim 0 (Phase 0a lead-signal)
Locked post-hoc EMA on existing JiT snapshots yields ≥ 0.05 FID gain.
- Determines whether the post-hoc EMA appendix exists at all.

#### Claim 3 (conditional appendix)
**Only if Claim 0 passes:** β chosen at 256² B/16 transfers to 512² G/16 within ±0.05 FID of per-scale-tuned optimum.

### Compute & Timeline
~7,800 GPU-hr total. Phase 0a 1–3 hr. Phase 0b ~100 hr. Phase 1 ~1,270 hr. Phase 2 ~6,500 hr. 5–7 weeks to submission.

### Stage Gate (mechanism-signal-driven)
Phase 0b → Phase 1 promotion requires **all three** to hold at ep 100:
1. Mean-term slope over ep 50–100: ≤ 0.
2. Max per-block weight-norm column growth slope: ≤ 0.5%/ep.
3. EMA-vs-online FID gap not growing relative to vanilla baseline.

If any fails, the recipe does not predict mechanism repair; RETHINK.
