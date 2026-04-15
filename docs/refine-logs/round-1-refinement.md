# Round 1 Refinement

## Problem Anchor (verbatim from round 0)
- **Bottom-line problem:** JiT posts FID 1.82 @ 256², 1.78 @ 512². 7+ pixel-ViT competitors (DiP 1.79, DeCo 1.62, PixelDiT 1.61, EPG 1.58, SiD2 1.50 @ 512) already beat it. Improve JiT's FID without conceding its defining architectural simplicity.
- **Must-solve bottleneck:** Late-training weight-norm drift + fixed-decay EMA → FID **mean term grows** after ~300 ep while cov term improves. Every contemporary pixel-ViT winner patches this symptom architecturally; repo diagnostics place the root cause at optimization.
- **Non-goals:** No detail heads / dual-stack / frequency decoders; no cross-t forwards; no EMA-teacher functional_call; no mid-step residual heads; no SSL pretraining; no sampler change; not latent-space.
- **Constraints:** Single forward pass per step; vanilla DDP; ≤ 300 LoC; ImageNet-1k only; ~8k GPU-hr total; NeurIPS/ICML/ICLR target.
- **Success condition:** ≥ 0.20 FID gain at 256² B/16 under matched compute, ≥ 0.15 at G/16; repo's FID-decomposition shows the mean-term is repaired, not masked.

## Anchor Check
- **Original bottleneck:** late-training mean-term FID drift, optimization-rooted, on plain pixel-ViT.
- **Why revised method still addresses it:** MP layers remove the norm-drift source; x-pred σ-calibration prevents any σ bucket from dominating gradient flow; Phase-0 uses measured mean-term trace as the gate. The revised plan makes this one-mechanism-one-diagnostic story explicit.
- **Reviewer suggestions rejected as drift:** none. Reviewer called no drift and all suggestions reinforce anchor.

## Simplicity Check
- **Dominant contribution after revision:** **MP-JiT with empirical x-pred σ-calibration** (one method). Post-hoc EMA is now *calibration*, with a locked protocol, demoted to a supporting appendix result unless Phase-0 shows a transferable gain.
- **Components removed or merged:**
  - LayerNorm-β ablation: dropped (not tied to mean-term pathology).
  - 2-layer u(σ) MLP → 16-bucket logσ weighting table (reducing knobs from a free MLP to a clamped, pilot-initialized vector).
  - Post-hoc EMA: demoted from core to "locked calibration protocol" (not an argmin over FID).
- **Reviewer suggestions rejected as unnecessary complexity:** none — all simplifications accepted.
- **Why remaining mechanism is still the smallest adequate route:** one architectural hygiene change (MP) + one loss-side calibration (16-dim logσ weight vector) + one offline hygiene protocol (post-hoc EMA with fixed β-rule). The three legs are causally tight and each targets a named pathology.

## Changes Made

### 1. σ-uncertainty head → bucketed empirical σ-calibration (CRITICAL)
- **Reviewer said:** free MLP is generic heteroscedastic regression; not specific to x-pred.
- **Action:** replace 2-layer u(σ) MLP with a **16-bucket monotone-clamped logσ weighting table** `w[b(σ)]`, where `b: logσ → {0,…,15}` is a fixed uniform binning over the JiT log-normal σ range. Initialize from a pilot: run 5k training steps with uniform weighting, measure per-bucket squared residual `r²[b]`, and set `log w[b] ← -log(r²[b])` (clamped to `[-3, +3]`). During training, `w[b]` is trainable but monotone-clamped: enforce `w[b] ≤ w[b+1]` via a sorted-positive parameterization (`w = cumsum(softplus(θ))`). This forgoes the theoretical `u(σ) + exp(-u(σ))…` form in favor of an honest empirical weighting.
- **Reasoning:** x-pred's σ-difficulty curve is empirical (cleaner than the ε-pred derivation) — a bucketed vector captures it with far fewer knobs than an MLP, is easier to interpret, and avoids the "metric-tuned weight learning" trap.
- **Impact on core method:** u(σ) head deleted. New loss: `L = Σ_i w[b(σ_i)] · ‖x̂_i − x_i‖²`. ~0.5k trainable params (down from 20k).

### 2. MPLinear attention handling (CRITICAL)
- **Reviewer said:** how MP on q/k/v preserves attention temperature is unspecified.
- **Action:** separate learnable scalar gains `g_q, g_k, g_v, g_proj` per block. **Initialization locks the qk logit scale:** at init, compute `s_init = (g_q · g_k) / sqrt(d_head)` and treat it as the reference temperature. During the first 5 epochs, constrain `(g_q · g_k) ≤ s_init · (1 + 0.1 · epoch / 5)` via a soft penalty (λ = 1e-3) so that the logit scale is not free to explode early. After warmup the constraint is released.
- **Reasoning:** in attention, the q·k dot-product variance is the sensitive quantity — letting both `g_q` and `g_k` grow unconstrained re-introduces the norm-drift pathology MP is supposed to remove. The logit-scale lock is a 1-line constraint.
- **Impact:** MPLinear signature stays ~30 LoC, plus a ~10 LoC qk-logit-scale regularizer in the Attention module.

### 3. Stage-gate feasibility plan (CRITICAL)
- **Reviewer said:** three changes coupled; mechanism must de-risk before 400ep/512² headline.
- **Action:** hard stage gate inserted between Phase 0 and Phase 1:

  ```
  Phase 0a: post-hoc EMA on EXISTING JiT snapshots (1–3 GPU-hr)
      ├─ ≥ 0.05 FID gain → proceed
      └─ < 0.05 FID gain → EMA calibration demoted; proceed with MP + σ-calibration only
  Phase 0b: short pilot 256² B/16 100ep — {baseline, MP-only, MP + σ-calibration}
      ├─ mean-term trace flat after ep 80 on MP runs → PROMOTE
      └─ mean-term still rises → RETHINK (likely means non-MP optimizer bug)
  Phase 1: 400ep ablation matrix ONLY after Phase 0b gates pass.
  Phase 2: 512² headline ONLY after Phase 1 shows the recipe-level win at 256².
  ```
  The "mean-term trace flat" gate is the mechanism-repair evidence, not FID alone.
- **Reasoning:** attribution muddiness is the reviewer's real concern. The stage gate uses the mean-term trace (a mechanism signal, not an outcome signal) as the gate, so we promote only when the mechanism is repaired.
- **Impact:** no compute lost — Phase 0a is 1–3 GPU-hr, Phase 0b is ~100 GPU-hr (3 runs × 100ep), both already in the plan; the gate rule is what's new.

### 4. Post-hoc EMA → locked calibration protocol (IMPORTANT)
- **Reviewer said:** `argmin_β FID(val_5k)` looks like metric-tuned checkpoint search.
- **Action:** pre-declare the EMA calibration protocol before any experiment runs:
  - **Snapshot cadence:** 32 uniform snapshots over the last 60% of training (ignore early snapshots, covers the ep ≥ 0.4 · total regime where drift starts).
  - **Calibration subset:** a fixed held-out 5k-image ImageNet val split (disjoint from 50k FID evaluation set), declared at the start, never touched during training.
  - **β-selection rule:** β is chosen by **minimum training-loss** on a held-out σ-grid (not FID), using the same σ-bucket weighting. FID is used only as a verification metric.
  - **β transfer rule:** the β chosen at 256² B/16 is the β reported at 512² and at G/16 — no per-scale re-tuning. Transferability is a testable claim of the paper.
- **Reasoning:** locking the protocol converts post-hoc EMA from "FID-hacked search" into a calibration step with a pre-declared rule, like LR warmup.
- **Impact:** paper gains a clean protocol spec; risk of "metric tuning" reviewer critique eliminated.

### 5. Drop LayerNorm-β ablation (simplification)
- **Reviewer said:** delete it unless tied to the mean-term pathology.
- **Action:** removed. Keep LayerNorm γ and β both on (standard JiT config). This is now a non-change.

### 6. Reframe as empirical σ-calibration, not theoretical x-pred derivation (IMPORTANT)
- **Reviewer said:** if it's empirical, don't call it a derivation.
- **Action:** the paper framing becomes: **"Empirical σ-calibration + magnitude-preserving attention for pixel-ViT x-prediction."** The Bayes/Kendall-Gal derivation is in an appendix as a motivation, not a claim. The main-text framing is: (a) diagnostic → (b) bucketed σ-weighting learned from pilot → (c) MP layers fix norm-drift → (d) post-hoc EMA protocol as calibration hygiene.

---

## Revised Proposal

### Problem Anchor
[see top]

### Technical Gap
Every pixel-ViT winner adds architecture to patch the mean-term drift symptom. Repo diagnostics locate the root cause at optimization (weight-norm drift, σ-bucket imbalance, EMA mis-calibration). EDM2 fixed these for U-Net/ε-pred/64²; nobody ported the recipe to ViT/x-pred/high-res. Naive fixes — scaling, more data, bigger backbones — don't fix the mean-term trace.

### Method Thesis
**One-sentence thesis:** Two minimal optimization-side changes — MPLinear with per-projection gains under a qk-logit-scale lock, and a 16-bucket empirical σ-weighting calibrated from a 5k-step pilot — repair JiT's diagnosed late-training mean-term FID drift without any architectural or sampling change.

Post-hoc EMA is used as a fixed calibration protocol, not as an independent contribution.

**Why smallest adequate:** MP targets the norm-drift source; bucketed σ-weighting targets σ-bucket imbalance; no new module on the denoising path.

**Why timely:** pixel-ViT has converged; recipe-level optimization fixes are the leverage point. Architectural patching is saturated.

### Contribution Focus
- **Dominant contribution:** MP-JiT with bucketed empirical σ-calibration (single mechanism, two causally-linked legs: architecture hygiene + loss hygiene).
- **Supporting contribution:** diagnostic validation — we show the recipe flattens the mean-term trace, confirming the repair is at the root cause, not a masked architectural gain.
- **Appendix-level:** post-hoc EMA with locked protocol (a hygiene method, transferable β across scales is the testable claim).
- **Non-contributions:** no new architecture; no new objective beyond bucketed weighting; no new sampler/solver; no new data.

### Proposed Method

#### Complexity Budget
- Frozen/reused: JiT ViT backbone, bottleneck patch embed, x-pred loss skeleton, Heun sampler, interval-CFG.
- New trainable (≤ 2):
  1. `MPLinear` in Attention (q/k/v/proj) and MLP (fc1/fc2). Four scalar gains (g_q, g_k, g_v, g_proj) per block + 2 for MLP. Same weight matrix shape.
  2. Bucketed logσ weight vector: 16 scalars per model (not per-block). Monotone-clamped positive.
- Intentionally excluded: detail head, DCT loss, cross-t consistency, CUGR, k-Diff, u(σ) MLP (replaced), LN-β ablation (dropped).

#### Core Mechanism

**MPLinear with qk-scale lock.**
```
MPLinear.forward(x):
    W_hat = W / ||W||_col
    return x @ W_hat * gain          # per-layer scalar gain

Attention block:
    q = MPLinear_q(x) * g_q
    k = MPLinear_k(x) * g_k
    v = MPLinear_v(x) * g_v
    logits = (q @ k.T) / sqrt(d_head)
    out = softmax(logits) @ v
    out = MPLinear_proj(out) * g_proj

At init: compute s_init = (g_q · g_k) / sqrt(d_head)       # treat as reference temperature
During epochs 0..5: add penalty λ · ReLU((g_q · g_k) − s_init · (1 + 0.1·epoch/5))²
After epoch 5: no penalty.
```

**Bucketed empirical σ-calibration.**
```
Pilot (5k steps, weight=1):
    for each step: compute r² = ‖x̂ − x‖², log σ → bucket b(σ)
    per-bucket mean r²[b]
    init w[b] ← clamp(-log(r²[b]/median(r²)), [-3, +3])

Sorted-positive parameterization:
    θ[0..15] ∈ R^16 (free)
    w = softplus(θ)
    w_cum = cumsum(w)                # enforces monotonicity: w_cum[0] ≤ … ≤ w_cum[15]
    w_cum /= mean(w_cum)             # keeps overall loss scale constant

Main training loss:
    L = Σ_i w_cum[b(σ_i)] · ‖x̂_i − x_i‖²
```

Rationale: monotone-clamped, pilot-initialized, cannot collapse to uniform (sorted-positive); cannot explode (rescaled to unit mean); 16 scalars; interpretable.

**Inference:** unchanged — JiT sampler + interval-CFG, loads post-hoc-reconstructed weights.

#### Locked Post-hoc EMA Calibration Protocol (appendix)
- Snapshot cadence: 32 snapshots uniformly over the last 60% of epochs.
- Calibration subset: pre-declared disjoint 5k ImageNet val split.
- β selection: minimum **training-loss** on a held-out σ-grid with the same bucketed weighting (not FID).
- β transfer: the β chosen at 256² B/16 is reported verbatim at 512² and G/16. Transferability is a claim.

#### Integration
- `model_jit.py`: `nn.Linear` → `MPLinear` in Attention/Mlp; add 4 scalar gains per Attention block + qk-scale-lock penalty.
- `denoiser.py`: apply bucketed σ-weight to the loss; hold the 16-dim `θ` and bucket-indexer.
- `engine_jit.py`: 5k-step pilot at epoch 0 to initialize `w`; uniform snapshot saving over last 60%.
- `tools/post_hoc_ema.py`: locked-protocol reconstruction (~100 LoC).

#### Training Plan
- Stage 0: 5k-step pilot with weight=1 to initialize σ-bucket weights.
- Stage 1: joint training with MPLinear + bucketed σ-weighting, cosine LR with warmup (JiT default).
- Stage 2: offline post-hoc EMA reconstruction with locked protocol.

No weight decay on MPLinear weights; AdamW (WD=0) for gains and `θ` to suppress drift.

#### Failure Modes
- **qk-scale lock too tight** → training stalls for first 5 ep. Detection: loss plateau. Mitigation: raise relaxation slope from 0.1 to 0.2.
- **σ-bucket weights collapse to uniform** → monotone-positive + rescaling prevents it; if observed, the sorted-positive parameterization has a bug.
- **Pilot residual estimate too noisy** → re-run pilot for 10k steps; `w` clamp handles outlier buckets.
- **Post-hoc EMA β transfer fails** (different β optimal at 512²) → report honestly; the appendix claim weakens; core method unaffected.
- **DDP regression >5%** → all new modules rank-0 snapshot only, no custom hooks. Strict vanilla DDP.

#### Novelty / Elegance
Closest work: EDM2 (U-Net, ε-pred, 64²). Different backbone, target, scale, and — critically — the bucketed empirical σ-weighting with monotone-clamp + pilot init is a simpler, more interpretable replacement for EDM2's u(σ) MLP. The qk-logit-scale lock is a new design element not in EDM2 (U-Nets have no attention). SiD2 uses a simpler σ-weighting but no MP, no post-hoc EMA, no pilot calibration; ours is a strict superset.

Elegance: ~250 LoC total (MPLinear + gain bookkeeping + 16-scalar σ calibration + qk lock + offline script).

### Claim-Driven Validation Sketch

#### Claim 1 (dominant): MPLinear + bucketed empirical σ-calibration repairs the diagnosed late-training mean-term FID drift on pixel-ViT x-prediction.
- **Minimal experiment:** JiT-B/16 256² 400ep with (MP + σ-calibration) vs vanilla compute-matched. Measure FID per-epoch **and** `analysis_fid_decomposition.py` per-epoch.
- **Ablations:** A vanilla / B MP-only / C σ-calibration-only / D MP + σ-calibration (headline).
- **Metric:** (i) FID-50k at NFE 100. (ii) FID mean-term trace per epoch. (iii) weight-norm trace per epoch.
- **Expected evidence:** D beats A by ≥ 0.20 FID; D's mean-term trace is flat after ep 300; A's grows. B flattens norm trace but may not move FID. C helps but less than D.

#### Claim 2 (supporting): The FID gain is attributable to mean-term repair, not covariance-term improvements.
- **Minimal experiment:** apply `analysis_fid_decomposition.py` to every ablation cell.
- **Expected evidence:** D − A is > 80% explained by mean-term reduction.

#### Claim 0 (lead-signal, Phase 0a): Post-hoc EMA with locked protocol, on existing JiT snapshots, yields ≥ 0.05 FID gain at zero training cost.
- **Minimal experiment:** existing `output_ylab/` and `output_imagenet_128_fm/` snapshots → β-selection → FID on disjoint 5k.
- **Expected:** ≥ 0.05 FID gain. Transferability check: does β selected at 256² transfer to 128²?

#### Claim 3 (appendix): The β from the locked calibration protocol transfers across model scale and resolution.
- **Minimal experiment:** pick β at 256² B/16; apply verbatim to 512² G/16 run.
- **Expected evidence:** the same β stays within ±0.05 FID of the per-scale-tuned optimum.

### Experiment Handoff
- **Must-prove:** Claim 1 (with mean-term trace), Claim 2.
- **Must-run ablations:** A/B/C/D matrix at 256² B/16 400ep matched compute.
- **Critical metrics:** FID-50k @ NFE 100 (primary), mean-term + cov-term per epoch (mechanism), weight-norm column trace (optimization diagnostic), qk-logit variance per epoch (MP hygiene diagnostic).
- **Highest-risk:** (1) MP + qk-scale lock stable from scratch; (2) bucketed σ-weighting doesn't collapse; (3) Phase 0a snapshot coverage is sufficient.

### Compute & Timeline
~7,800 GPU-hr to full paper. Phase 0a: 1–3 GPU-hr (same-day). Phase 0b stage gate: ~100 GPU-hr (3 days). Phase 1: ~1,270 GPU-hr (5 days). Phase 2: ~6,500 GPU-hr (10 days). Timeline: 5–7 weeks to submission.
