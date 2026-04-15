# Round 3 Refinement

## Problem Anchor (verbatim from round 0; EMA causal-role updated per R3 fix)
- **Bottom-line:** improve plain pixel-ViT JiT (1.82@256, 1.78@512) past architectural-winner pixel-ViTs without conceding JiT's defining simplicity.
- **Must-solve bottleneck:** late-training **mean-term FID drift, optimization-rooted at weight-norm growth**, with fixed-decay EMA *amplifying* (not co-causing) the symptom. Measured in-repo via `analysis_fid_decomposition.py`. Architectural patches (DiP/DeCo/PixelDiT) hide the symptom; we fix the cause.
- **Non-goals:** no detail head / dual-stack / freq decoders / cross-t forwards / EMA-teacher / mid-step heads / SSL / sampler change / latent-space.
- **Constraints:** single forward per step; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS/ICML/ICLR.
- **Success:** ≥ 0.20 FID @ 256² B/16 matched compute; ≥ 0.15 @ G/16; mean-term trace flattened.

## Anchor Check
- Original bottleneck: late-training optimization-rooted mean-term FID drift on plain pixel-ViT.
- Method still addresses it: MP removes the norm-drift root cause; pilot-calibrated bounded σ-weighting equalizes per-bucket gradient pressure; the broadened stage gate verifies mechanism repair via three correlated early-warning slope signals before promotion.
- Reviewer suggestions rejected as drift: none. EMA reframed as amplifier (R3 fix) preserves anchor intent.

## Simplicity Check
- Dominant contribution: **MP-JiT + pilot-calibrated bounded x-pred σ-weighting**. Single mechanism, two causally-linked legs, post-hoc EMA *strictly conditional* (deleted from main body if Phase 0a < 0.05).
- Components removed/merged this round:
  - σ-weight clamp/normalize order swapped to preserve exact unit-mean invariant.
  - "freeze θ after warmup" promoted to default behavior; continued learning is fallback.
  - "strict subset" / "beat all" phrasing softened.
- Reviewer suggestions rejected as unnecessary complexity: none.
- Why remaining mechanism is still smallest adequate: one architecture-hygiene change (MP) + one loss-side calibration (16-scalar pilot-init bounded σ-weighting). Mechanism evidence is now traceable per-bucket and per-block.

## Changes Made

### 1. σ-weight projection invariant fix (IMPORTANT)
- **Reviewer said:** clamp-after-renormalize doesn't preserve exact unit mean.
- **Action:** swap order to clamp-then-renormalize:
  ```
  w_raw = clamp(softplus(θ), 0.1, 10)        # bounded
  w     = w_raw / mean(w_raw)                # exact unit mean
  ```
  **Invariant:** `mean(w) = 1` exactly; per-bucket `w ∈ [0.1/μ_raw, 10/μ_raw]` ⊂ a bounded interval since `μ_raw ∈ [0.1, 10]`.
- **Reasoning:** keeps the global loss scale invariant by construction; eliminates effective-LR drift across training.

### 2. EMA reframed as amplifier in anchor (IMPORTANT)
- **Reviewer said:** anchor names EMA mis-calibration as co-equal root cause but EMA is conditional in the method.
- **Action:** updated anchor language: "weight-norm growth is the root cause; fixed-decay EMA amplifies the resulting mean-term drift but is not itself the cause." Post-hoc EMA appendix is then a hygiene check, not a co-equal claim.
- **Reasoning:** keeps causal story coherent across anchor and method; respects the conditional-appendix design.

### 3. Stage-gate signal #3 operationalization (IMPORTANT)
- **Reviewer said:** EMA-vs-online FID gap at ep 100 too noisy without pre-declared cadence + tolerance.
- **Action:** pre-declared protocol:
  - **Cadence:** measure FID-5k every 10 ep starting ep 50 (6 measurements: 50, 60, 70, 80, 90, 100).
  - **FID set:** disjoint pre-declared 5k val subset.
  - **Variance tolerance:** signal #3 fires if `(FID_ema − FID_online) | ep 100  −  (FID_ema − FID_online) | ep 50  > 0.10 FID` (i.e., gap widened by > 0.10 over the 50-epoch window — comfortably above FID-5k noise floor of ~0.05).
- **Reasoning:** robust to FID noise; cheap (~6×5k FID = ~ 0.3 GPU-hr per run).

### 4. Per-bucket residual mechanism trace added (IMPORTANT)
- **Reviewer said:** without direct calibration evidence, 16-table looks like generic reweighting.
- **Action:** log per-bucket squared residual `r²[b]` every 5 epochs (free, just a 16-vector accumulator). Add to Claim 1 evidence:
  - Pilot σ-difficulty curve (ep 0): `r²[b]` has characteristic high-/low-σ shape.
  - Trained σ-residual curve (ep 200): `r²[b]` is flatter, indicating the σ-weighting equalized per-bucket loss as designed.
  - Direct mechanism: the trained `r²[b]` curve and the trained `w[b]` curve should be approximate inverses (pre-clamp).
- **Reasoning:** turns the 16-table from a knob into a measurable calibration mechanism.

### 5. Soften "strict subset" novelty phrasing (MINOR)
- **Action:** in the Novelty section: "MP-JiT is in the same optimization lane as SiD2 and EDM2 but is smaller and more targeted to plain pixel-ViT x-pred. We do not claim strict superset; the contributions sit in adjacent corners of the same recipe space."

### 6. Headline discipline (MINOR)
- **Action:** explicitly remove any "beat all" or "best-in-class" language. The paper's contract is exactly: ≥ 0.20 FID @ 256² B/16 matched compute, ≥ 0.15 @ G/16, with mean-term trace flattened. No more, no less.

### 7. Default-freeze σ-weights after warmup (simplification)
- **Action:** make `θ` `requires_grad=False` after epoch 5 by default, since pilot init is expected to be near-optimal. Continued learning is a fallback only if the stage-gate signals come back borderline (one of the three signals between 0 and the threshold).

### 8. Conditional appendix tightening (simplification)
- **Action:** Phase 0a outcome rule made explicit: "If Phase 0a yields < 0.05 FID gain, delete the post-hoc EMA appendix and all snapshot-saving language from the main proposal. The core paper survives unchanged on Claims 1 and 2."

---

## Revised Proposal (Round 3)

### Problem Anchor (updated EMA framing)
- Bottom-line: improve plain JiT past pixel-ViT architectural winners without conceding JiT's simplicity.
- Bottleneck: late-training **weight-norm growth** in attention/MLP linears causes mean-term FID drift (root cause); fixed-decay EMA amplifies the resulting drift but is not itself the cause.
- Non-goals: no detail / dual-stack / freq / cross-t / EMA-teacher / mid-step / SSL / sampler / latent.
- Constraints: single forward; vanilla DDP; ≤ 300 LoC; ImageNet-1k; ~8k GPU-hr; NeurIPS-tier.
- Success: ≥ 0.20 FID @ 256² B/16; ≥ 0.15 @ G/16; mean-term trace flat after ep 300.

### Technical Gap
Pixel-ViT winners patch the symptom architecturally. Repo measurements locate the root cause at weight-norm growth + σ-bucket gradient imbalance. EDM2 fixed analogues for U-Net/ε-pred/64²; never ported. Naive scaling/data fixes don't fix the trace. Architectural add-ons killed locally as MSC-R/FSRH (DDP-fragile).

### Method Thesis
**MPLinear plus pilot-calibrated bounded x-prediction σ-weighting repairs JiT's diagnosed late-training mean-term FID drift without architectural or sampler change.**

### Contribution Focus
- **Dominant:** MP-JiT + pilot-calibrated bounded x-pred σ-weighting (one mechanism, two causally-linked legs).
- **Supporting:** diagnostic validation — recipe flattens the mean-term trace AND the per-bucket σ-residual curve, evidencing root-cause repair AND calibration mechanism (not generic reweighting).
- **Conditional appendix:** locked post-hoc EMA protocol + β-transfer claim, ONLY if Phase 0a ≥ 0.05 FID. Otherwise deleted from body.
- **Non-contributions:** no architecture / objective / sampler / data; qk-lock is enabler within MP, not contribution.

### Proposed Method

#### Complexity Budget
- Frozen: JiT ViT, bottleneck embed, x-pred skeleton, Heun sampler, interval-CFG.
- New trainable (≤ 2):
  1. MPLinear (replaces nn.Linear in q/k/v/proj/fc1/fc2). 1 internal gain per linear → 6 per block.
  2. Bucketed σ-weighting: 16 scalars `θ ∈ R^16`. Default frozen after ep 5.
- Excluded: detail head, DCT loss, cross-t consistency, CUGR, k-Diff, u(σ) MLP, monotone cumsum, separate qk gains, LN-β ablation.

#### Core Mechanism

**MPLinear (with embedded qk-scale-lock enabler).**
```
class MPLinear:
    W [out, in], no WD
    gain (scalar, init 1.0)
    forward(x): return (x @ (W / ||W||_col).T) * gain

Attention (4 MPLinear: q, k, v, proj):
    s_init = (q.gain · k.gain) / sqrt(d_head)               # frozen at init
    Epochs 0..5: loss += 1e-3 · ReLU((q.gain · k.gain) − s_init·(1+0.1·ep/5))²
    Released after ep 5.
```

**Pilot-calibrated bounded σ-weighting.**
```
Pilot (5k steps, w=1):
    per-bucket mean squared residual r²[b]   for b in 0..15
    log per-bucket trace.

Init:
    target_b = clamp(median(r²) / r²[b], 0.1, 10)
    θ[b] ← softplus_inv(target_b)

Training (every step):
    w_raw = clamp(softplus(θ), 0.1, 10)        # bounded
    w     = w_raw / mean(w_raw)                # EXACT unit mean (invariant)
    L     = mean over batch of w[b(σ_i)] · ‖x̂_i − x_i‖²

Default: θ.requires_grad = False after epoch 5.
Fallback (only if stage-gate borderline): keep θ trainable.
```

Inference: unchanged JiT sampler + interval-CFG.

#### Per-bucket residual mechanism logger
Every 5 epochs, log `r²[b]` for b∈0..15 (16-vector accumulator, free). Used in Claim 1 evidence.

#### Conditional Post-hoc EMA Appendix (activated only if Phase 0a passes)
- 32 uniform snapshots over last 60% of epochs.
- Disjoint pre-declared 5k val subset.
- β chosen by min training-loss on held-out σ-grid.
- β chosen at 256² B/16 reported verbatim at 512² and G/16.
- **Activation rule:** if Phase 0a < 0.05 FID gain on existing JiT snapshots, the entire appendix and all snapshot language is deleted from the proposal body.

#### Integration & Training
- model_jit.py: nn.Linear → MPLinear; qk-scale-lock penalty inside Attention.
- denoiser.py: bucketed σ-weighting + invariant-preserving projection; θ buffer + bucket-indexer.
- engine_jit.py: 5k-step pilot at ep 0; uniform snapshot saving over last 60% (conditional on Phase 0a); per-bucket residual logger every 5 ep; mean-term + norm + EMA-gap loggers.
- tools/post_hoc_ema.py: locked-protocol reconstruction, conditional.
- Total ~250 LoC.

Joint training; cosine LR + warmup; no WD on MPLinear W; AdamW WD=0 on gains and θ.

#### Failure Modes
- qk-lock too tight → relax slope 0.1 → 0.2.
- σ-weights drift (during fallback unfreezing) → clamp+renormalize prevents structurally; freeze early if stable.
- Pilot noisy → extend to 10k steps.
- Phase 0a < 0.05 → delete appendix; core paper unaffected.
- DDP regression > 5% → vanilla nn.Modules; rank-0 snapshots.

#### Stage Gate (mechanism-signal-driven, ep 100)
Promote Phase 0b → Phase 1 only if **all three** hold:

1. **Mean-term slope:** d/dt ‖μ_r − μ_g‖² over ep 50–100 ≤ 0.
2. **Per-block weight-norm growth:** max over blocks of d/dt (mean ‖W_col‖) ≤ 0.5%/ep over ep 50–100.
3. **EMA-vs-online FID gap (operationalized):**
   - Cadence: FID-5k every 10ep from ep 50 → 6 measurements at {50,60,70,80,90,100}.
   - Tolerance: gap-widening (FID_ema − FID_online) | ep100 minus (FID_ema − FID_online) | ep50 ≤ +0.10 FID.

If any fails: RETHINK.

#### Novelty / Elegance
Closest: EDM2 (U-Net, ε-pred, 64²). Differences: ViT backbone, x-pred target, 256²/512² scale; pilot-calibrated bounded 16-bucket σ-weighting (interpretable, low-knob, mechanism-traceable) replaces u(σ) MLP. qk-lock is enabler. **MP-JiT is in the same optimization lane as EDM2 and SiD2 but is smaller and more targeted to plain pixel-ViT x-pred.** Total ~250 LoC.

### Claims

#### Claim 1 (dominant)
MP + pilot-calibrated σ-weighting flattens both the mean-term FID drift AND the per-bucket σ-residual curve.
- Min experiment: 256² B/16 400ep matrix A (vanilla) / B (MP-only) / C (σ-calib only) / D (full).
- Metrics:
  - FID-50k @ NFE 100.
  - Per-epoch mean-term + cov-term trace.
  - Per-block weight-norm column trace.
  - Per-bucket r²[b] curve at ep {0, 100, 200, 300, 400}.
  - EMA-vs-online FID gap.
- Expected: D beats A by ≥ 0.20 FID; D's mean-term flat after ep 300; D's per-bucket curve flatter than A's at ep 200; D's `w[b]` and trained `r²[b]` are approximate inverses.

#### Claim 2 (supporting)
The FID gain is attributable to mean-term repair (>80% of D−A from mean-term per `analysis_fid_decomposition.py`).

#### Claim 0 (Phase 0a lead-signal)
Locked post-hoc EMA on existing JiT snapshots ≥ 0.05 FID gain. Determines whether Claim 3 / appendix exists.

#### Claim 3 (conditional appendix; only if Claim 0 passes)
β chosen at 256² B/16 transfers to 512² G/16 within ±0.05 FID of per-scale optimum.

### Compute & Timeline
~7,800 GPU-hr total. Phase 0a 1–3 hr. Phase 0b ~100 hr. Phase 1 ~1,270 hr. Phase 2 ~6,500 hr. Stage-gate logging adds ≤ 2 GPU-hr/run. 5–7 weeks to submission.
