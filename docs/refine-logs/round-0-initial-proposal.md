# Research Proposal: EDM2-JiT — Optimization-Recipe Port to Pixel-Space Vision Transformer with x-Prediction

## Problem Anchor
- **Bottom-line problem:** JiT ([arXiv:2511.13720](https://arxiv.org/abs/2511.13720)), a plain pixel-space ViT with x-prediction, posts FID-50k 1.82 @ ImageNet 256² and 1.78 @ 512². Within 5 months of its release, 7+ pixel-space transformer competitors have published better FID (DiP 1.79, DeCo 1.62, PixelDiT 1.61, EPG 1.58, SiD2 1.50 @ 512). We need to improve JiT's FID without conceding its defining architectural simplicity.
- **Must-solve bottleneck:** JiT's FID degrades in late training because of a weight-norm drift / EMA mis-calibration pathology — empirically measured in this repo: the FID **mean term ‖μ_r − μ_g‖²** grows after ~300 epochs while the covariance term improves. Standard EMA does not recover from this; naive re-training does not fix it. Every contemporary pixel-ViT winner patches this symptom **architecturally** (detail head, dual-stack, pixel decoder) instead of at its optimization root.
- **Non-goals:**
  - No detail heads, no frequency-split decoders, no dual-stack architectures (those are DiP / PixelDiT / DeCo territory and were already attempted and killed in this repo as MSC-R + FSRH).
  - No cross-timestep forward passes, no EMA-teacher `functional_call`, no mid-forward residual heads (MSC-R/FSRH post-mortem: broke DDP scaling).
  - No representation-alignment / SSL pretraining (PixelREPA has cornered that).
  - No new denoising-step internals; no change to JiT's sampler.
  - Not a latent-space method.
- **Constraints:**
  - Single forward pass per training step. Vanilla DDP only; any change must preserve JiT's current multi-node scaling.
  - Compute budget: user cluster (8×H200 per node, ~8k GPU-hours available to the full paper).
  - Code change must be small (target <300 LoC) and live in `model_jit.py`, `denoiser.py`, `engine_jit.py`, `main_jit.py`.
  - Training data: ImageNet-1k (no extra data).
  - Venue target: NeurIPS/ICML/ICLR-scale submission.
- **Success condition:** An executable JiT variant that, under matched compute, improves FID at 256² B/16 by ≥ 0.20 (from 2.08 → ≤ 1.80) and at 256²/512² G/16 by ≥ 0.15 (target ≤ 1.60 / ≤ 1.50), without any new network on the denoising path and without breaking DDP. The user's existing diagnostic scripts (`analysis_fid_decomposition.py`, `measure_cond_uncond_gap.py`, DCT error analysis) must show the **mean-term is repaired**, not just masked.

## Technical Gap
Every published pixel-ViT method that beats JiT — DiP (detail head), DeCo (pixel decoder + DCT-band loss), PixelDiT (dual-stack pixel DiT), EPG (SSL-pretrained encoder) — patches the FID gap by adding a model capable of correcting high-frequency detail after the main ViT. This is a symptomatic fix: the core ViT is the same, it just gets repaired downstream.

User's repo diagnostics establish a different diagnosis: in the late phase of JiT training, column norms of `nn.Linear` weights in attention/MLP grow unboundedly; the standard EMA with a single fixed decay cannot track these changing weight statistics; the resulting FID mean term ‖μ_r − μ_g‖² degrades while the covariance term keeps improving. The symptom looks like "high-frequency detail wash-out," but the source is **optimization drift**, not architectural capacity shortage.

Karras's EDM2 ([2312.02696](https://arxiv.org/abs/2312.02696)) identified exactly this phenomenon on U-Nets and shipped a three-part recipe: (1) **magnitude-preserving (MP) layers** that normalize weight columns and remove weight decay, eliminating the norm-growth source; (2) **per-σ uncertainty-weighted loss** that prevents any σ bucket from dominating gradient flow; (3) **post-hoc EMA reconstruction** that solves offline for the EMA decay that minimizes FID, rather than using a guessed fixed decay during training. The recipe is SOTA at pixel-ImageNet-64. Nobody has ported it to (a) a ViT backbone, (b) x-prediction target, (c) high-resolution pixel diffusion. The pixel-ViT field went architectural instead.

Naive extensions fail because:
- **Scaling the model** does not change the weight-norm-drift dynamics (the user has tested up to G/16).
- **More training data** changes covariance-term improvements but not mean-term drift (per `analysis_fid_decomposition.py`).
- **Larger architectural add-ons** (FSRH, DiP-style head) incur the exact MSC-R/FSRH failure mode in this repo and have been banned.

Missing mechanism: a drop-in optimization recipe that removes the root cause of mean-term drift on the existing JiT backbone.

## Method Thesis
- **One-sentence thesis:** Porting the three-part EDM2 optimization recipe — magnitude-preserving layers + per-σ uncertainty-weighted x-prediction loss + post-hoc EMA reconstruction — to JiT's plain pixel-ViT backbone repairs the diagnosed late-training FID mean-term drift and closes most of the pixel-ViT gap without adding any new network or changing the denoising step.
- **Why this is the smallest adequate intervention:** The three EDM2 ingredients target three causally-linked failure modes (norm drift, σ-bucket dominance, post-training EMA mis-calibration). Each ingredient attacks a single named pathology. There is no fourth knob added; no architectural change; no sampler change; no auxiliary model. The dominant contribution is the *port itself* (with the necessary x-prediction re-derivation), not an invention.
- **Why this route is timely in the foundation-model era:** The pixel-ViT diffusion field has converged on a single backbone (plain ViT, x-prediction, bottleneck patch-embed). Large models trained end-to-end at pixel scale are now routine and are optimizer-limited in the same way EDM2 diagnosed for U-Nets in 2024. Solving the optimization layer centrally — rather than patching every backbone with a detail head — is the natural foundation-model-era move. This becomes the pixel-ViT analogue of "what LR warmup is to transformers": a recipe the field adopts uniformly.

## Contribution Focus
- **Dominant contribution:** The EDM2 → pixel-ViT x-prediction port, including the non-trivial re-derivation of the per-σ uncertainty-weighted loss under x-prediction targets (EDM2's derivation is σ-parameterized for ε-prediction and does not transfer directly).
- **Optional supporting contribution:** Diagnostic-driven framing — we show, using FID mean/cov decomposition and cond/uncond gap measurements already present in the repo, that the EDM2 recipe repairs the specific pathology it is designed to repair, not merely the aggregate FID. This turns a "recipe port" into a "root-cause-first" paper.
- **Explicit non-contributions:**
  - No new network architecture.
  - No new training objective beyond the EDM2 port.
  - No new sampler or solver.
  - No new data.
  - No separate contribution for register tokens, SiD2 σ-schedules, or k-Diff (these are cheap side ablations, acknowledged as orthogonal).

## Proposed Method
### Complexity Budget
- **Frozen / reused backbone:** JiT model (ViT, bottleneck patch embed, x-prediction loss, Heun sampler, interval-CFG). No change to `_denoise_step`. No change to the sampler.
- **New trainable components (≤ 2):**
  1. `MPLinear` replacement for `nn.Linear` in attention (q/k/v/proj) and MLP (fc1/fc2). Same parameter count as `nn.Linear` modulo a learned scalar gain. Single file; ~30 LoC.
  2. σ-uncertainty head: a 2-layer MLP taking JiT's existing σ-embedding and outputting a scalar log-variance u(σ). <20k params.
- **Tempting additions intentionally not used:** detail head (archived banned), frequency-aware DCT loss (DeCo territory), cross-t consistency loss (MSC-R killed), cond/uncond gap regularizer (deferred), k-Diff learned parameterization (deferred).

### System Overview
```
image x, label c
    │
    ▼
[noise z_t = α_t x + β_t ε]  (JiT unchanged)
    │
    ▼
[JiT ViT with MPLinear everywhere] ── x̂(z_t, t, c)
    │
    ├──► ‖x̂ − x‖²
    │
    ▼
[σ-uncertainty head on σ(t)] ── u(σ)
    │
    ▼
Loss = exp(−u(σ)) · ‖x̂ − x‖² + u(σ)

(Training-only.  Inference is unchanged: JiT sampler + interval-CFG,
but the weights at inference come from post-hoc EMA reconstruction
rather than the live EMA.)

During training: save M=32 checkpoints uniformly over epochs.
After training: solve offline for β* via
    β* = argmin_β FID(PowerEMA(snapshots, β), val_5k)
```

### Core Mechanism
- **Input / output:** same as vanilla JiT. Model input: (z_t, t, c). Output: x̂ (clean-image prediction). Nothing changes at the interface.
- **Architecture:**
  - Replace every `nn.Linear` in `Attention` and `Mlp` with `MPLinear`. `MPLinear.forward(x) = x @ (W / ||W||_col) · gain`, with `gain` a trainable scalar per layer (init 1.0), no weight decay on `W`.
  - Keep LayerNorm γ, drop β (EDM2 hygiene — or keep β if stability demands it; ablate).
  - Add 2-layer σ-uncertainty MLP on the existing σ-embedding.
- **Training signal / loss:**
  - Weighted x-prediction MSE: `L = exp(−u(σ)) · ‖x̂ − x‖² + u(σ)`.
  - The `u(σ)` term is an **uncertainty regularizer** — enforces that u(σ) is consistent with the realized per-σ residual magnitude, per Kendall & Gal 2017 / EDM2 Eq. 12.
  - **Re-derivation under x-prediction:** in EDM2 the per-σ residual magnitude is σ-dependent because the target is ε (noise) and `σ·∇log p_t` scales with σ. Under x-prediction the target is the clean image, so the ideal-model residual is σ-independent in expectation. But the *realizable* residual is σ-dependent because low-t regions have low intrinsic dimension (user's `analysis_intrinsic_dimension.py`) and are easier to fit. Thus `u(σ)` learns the **empirical** σ-difficulty curve, not the theoretical one. This is the paper's non-trivial derivation contribution.
  - No weight decay on MPLinear weights. AdamW → Adam for MP layers; AdamW retained for gain / u(σ) head.
- **Why this is the main novelty:** The three-way intersection {EDM2 recipe × ViT × x-prediction} has no prior entry in the literature. The x-pred re-derivation of the uncertainty loss is the missing technical piece.

### Optional Supporting Component
- **Post-hoc EMA reconstruction** (Karras 2024). Save M=32 snapshots uniformly during training; offline, sweep β ∈ {0.9, 0.99, 0.999, 0.9999} and solve for the power-function-EMA that minimizes FID on a val 5k subset.
- Input / output: list of saved model snapshots → single EMA-reconstructed model checkpoint.
- Training signal / loss: offline FID minimization over β (not trained end-to-end; this is a one-time offline optimization).
- **Why it doesn't create contribution sprawl:** it is the third leg of the EDM2 recipe and is offline/free-lunch. The paper frames all three legs (MP + uncertainty + post-hoc EMA) as one recipe, not three contributions.

### Modern Primitive Usage
This paper deliberately does *not* attach an LLM / VLM / RL / Diffusion-distillation primitive. The correct foundation-model-era move here is: **solve the optimizer, not the architecture.** Modern pixel-ViT diffusion is large-model, end-to-end, pixel-space — exactly the regime where an optimizer-level recipe has leverage. The "frontier-native" part is recognizing that pixel-ViT has converged enough that a recipe-level contribution is the right unit.

If the reviewer pushes toward adding a modern primitive, the push-back is: MSC-R/FSRH tried that (EMA-teacher, ~distillation-flavored), broke DDP, and was killed by the user. This paper's contribution is recipe-level precisely because the architectural-addon axis is saturated.

### Integration into Base Generator / Downstream Pipeline
- Replace `nn.Linear` → `MPLinear` in `model_jit.py` (localized to `Attention` and `Mlp`).
- Add σ-uncertainty head to `denoiser.py`; apply to loss in the loss block.
- Snapshot saving every N epochs: add to `engine_jit.py`.
- Offline script `tools/post_hoc_ema.py` (~80 LoC) for the EMA reconstruction sweep.
- Inference: fully unchanged; JiT sampler + interval-CFG, loads the reconstructed checkpoint.

### Training Plan
- **Joint training** (no stages): from-scratch retrain with MPLinear + uncertainty loss on. Snapshot every 1/32 of epochs.
- Data: ImageNet-1k at 128²/256²/512² (following JiT's protocol).
- Optimizer: mixed — Adam (no weight decay) for MPLinear weights, AdamW for gain + u(σ) head.
- LR schedule: JiT's default (cosine with warmup).
- Batch size and σ-sampling: unchanged from JiT (log-normal σ).
- **Offline post-hoc EMA step:** run after training finishes; 1–3 GPU-hours.

### Failure Modes and Diagnostics
- **MP layers destabilize x-prediction at high-t.** Detection: loss spike or NaN within first 20 epochs. Mitigation: warmup MP gain; initialize gain at `||W||_col` so MPLinear ≡ nn.Linear at t=0; decay toward unit norm over first 5 epochs.
- **Uncertainty head collapses u(σ) to a constant** (ignoring σ-dependence). Detection: plot u(σ) vs σ at the end of training; if flat, the regularizer is inert. Mitigation: add a prior over u(σ) derived from JiT's published σ-loss curve, or a small KL regularizer.
- **Post-hoc EMA gives no gain over live EMA.** Detection: β* converges to the live-EMA decay. Implication: the mean-term drift diagnostic was misdiagnosed; fall back to SiD2 σ-schedule + MP only.
- **DDP regression.** Detection: per-step time increases >5% over vanilla JiT. Mitigation: all new modules are standard `nn.Module`s with no custom DDP hooks; the snapshot save happens on rank 0 only. If per-step time regresses, revert uncertainty head (it's the only module that runs on every step beyond MP).

### Novelty and Elegance Argument
Closest work: **EDM2** ([2312.02696](https://arxiv.org/abs/2312.02696)) — same recipe on U-Nets, ε-prediction, ImageNet-64. Exact difference: backbone (ViT vs U-Net), target (x-pred vs ε-pred), scale (256²/512² vs 64²). The x-pred re-derivation of the uncertainty loss is genuinely new.
Next-closest: **SiD2** (2410.19324) — simpler σ-weighting on U-ViT; uses v-prediction; no MP, no post-hoc EMA. Our recipe is a strict superset.
PixelREPA, DiP, DeCo, PixelDiT, EPG: all architectural; not in the same lane.

Elegance argument: the paper's contribution is a three-line change to `model_jit.py`, a one-block change to `denoiser.py`, and a 80-LoC offline script. The diagnostic story (mean-term drift → MP fixes it; σ-bucket dominance → uncertainty fixes it; late-training EMA mis-calibration → post-hoc EMA fixes it) is tight enough to be a single figure.

## Claim-Driven Validation Sketch

### Claim 1 (dominant): The EDM2 recipe port improves FID on pixel-ViT x-prediction at 256² and 512² without any architectural change.
- **Minimal experiment:** train JiT-B/16 256² 400ep with all three recipe ingredients; compare to vanilla JiT-B/16 at compute-matched FLOPs.
- **Baselines / ablations:** vanilla (A); MP-only (B); MP + uncertainty (D); MP + uncertainty + post-hoc EMA (E — full).
- **Metric:** FID-50k at NFE 100.
- **Expected evidence:** E beats A by ≥ 0.20 FID. B matches or slightly beats A. D beats B by ≥ 0.05 FID. E beats D by ≥ 0.05 FID.

### Claim 2 (supporting): The FID gain comes from repairing the diagnosed late-training mean-term drift, not from a masked architectural gain.
- **Minimal experiment:** apply `analysis_fid_decomposition.py` to A, B, D, E; apply `weight_norm_rescale.py` to A and B.
- **Baselines / ablations:** same runs as Claim 1.
- **Metric:** FID mean term ‖μ_r − μ_g‖² and cov term Tr(...) separately, per-epoch trace.
- **Expected evidence:** A's mean term grows after ep 300 and is the dominant remaining gap; B/D/E's mean term is approximately flat after ep 300. The weight-norm rescale hack that partially repairs A is a no-op on B (since norms are already bounded).

### (Optional) Claim 0 (lead-signal): Post-hoc EMA alone, applied to existing JiT checkpoints, already recovers a measurable fraction of the FID gap at zero training cost.
- **Minimal experiment:** scan `output/` and `output_ylab/` for saved JiT snapshots; apply `tools/post_hoc_ema.py` sweep.
- **Baselines / ablations:** live-EMA `checkpoint-last.pth` vs best-β reconstruction.
- **Metric:** FID-50k on a 5k val subset (to keep cheap).
- **Expected evidence:** ≥ 0.05 FID gain at zero training cost. If true, this is a Day-1 figure in the paper.

## Experiment Handoff Inputs
- **Must-prove claims:** Claim 1 and Claim 2 above.
- **Must-run ablations:** the 5-cell matrix (A baseline / B MP-only / C uncertainty-only / D MP+uncertainty / E full recipe), all at 256² B/16 400ep matched compute.
- **Critical datasets / metrics:** ImageNet-1k 256² FID-50k (primary), 512² FID-50k (scale check), FID mean/cov decomposition (mechanism check), cond/uncond gap trace (secondary mechanism check).
- **Highest-risk assumptions:**
  1. MP layers do not destabilize x-prediction from scratch.
  2. u(σ) learns a non-trivial σ-dependent difficulty signal under x-prediction.
  3. Existing JiT snapshots exist and span enough of late training for post-hoc EMA to give Day-1 signal.

## Compute & Timeline Estimate
- **Estimated GPU-hours to full paper:** ~7,800
  - Phase 0a (post-hoc EMA on existing snapshots): 1–3 GPU-hr.
  - Phase 0b (kill-or-validate MP at 128²): ~38 GPU-hr.
  - Phase 1 (256² B/16 ablation matrix): ~1,270 GPU-hr.
  - Phase 2 (headline G/16 at 256² + 512²): ~6,500 GPU-hr.
- **Data / annotation cost:** none beyond ImageNet-1k.
- **Timeline:** ~4–6 weeks from go-ahead to submission-ready results at 256²; +2 weeks for 512² headline.
