# Experiment Plan: MP-JiT

**Problem:** Close a meaningful fraction of the gap to pixel-ViT architectural winners on ImageNet-1k by repairing JiT's late-training mean-term FID drift at its optimization root, without architectural or sampler change.

**Method Thesis:** MPLinear plus pilot-calibrated fixed bounded x-prediction σ-weighting repairs JiT's diagnosed late-training mean-term FID drift without architectural or sampler change.

**Date:** 2026-04-15

**Repo layout:**
- **Vanilla JiT (preserved, read-only reference):** `/home/hzy980512/JiT/` @ commit `869190a`.
- **MP-JiT implementation (new, private):** `/home/hzy980512/edm2-jit/`. To be pushed to `JHuang778/edm2-jit` by the user.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1** — MP + pilot-calibrated fixed σ-weighting flattens late-training mean-term FID drift on pixel-ViT x-pred | Anchored problem; defines success | D − A ≥ 0.20 FID @ 256² B/16 400ep; mean-term trace of D is flat after ep 300 while A grows; three-signal stage gate passes at ep 100 | B1, B2 |
| **C2** — FID gain is attributable predominantly to mean-term repair (not cov-term / not architecture) | Kills the "it just learns a better ensemble" and "it just spent more compute" attacks | Majority of D−A explained by mean-term via `analysis_fid_decomposition.py`; B (MP-only) flattens weight-norm trace but does not recover full FID; matched-compute controls | B1, B2, B3 |
| **A1 (anti-claim)** — gains do NOT come from generic reweighting | Reviewer's #1 attack vector | Per-bucket r²[b] curve flattens over training; static `w[b]` = 1/r²[b]_pilot by construction; learnable-θ ablation (supp) does not materially improve over static table | B2, B4 |
| **A2 (anti-claim)** — gains do NOT come from parameter count or extra compute | Matched-compute fairness | Pilot counted in headline compute; matched-steps A/B/C/D; gain-only parameter delta is trivial (6 scalars/block + 16 buffer = negligible) | B1 |
| **C3 (conditional)** — locked post-hoc EMA appendix yields ≥ 0.05 FID on existing JiT snapshots and β transfers across scales | Adds optional headline line | Phase 0a pilot on existing snapshots; β-transfer measurement at G/16 512² | B5 |

## Paper Storyline

- **Main paper must prove:**
  - The MP-JiT recipe beats vanilla JiT by ≥ 0.20 FID @ 256² B/16 matched compute (headline number).
  - The FID gain repairs the mean-term drift (mechanism).
  - Neither leg alone is enough (one-leg-lesion ablation B and C each underperform D).
  - ≥ 0.15 FID @ G/16 512² scaling.
- **Appendix can support:**
  - Per-bucket r²[b] mechanism figure.
  - Weight-norm column trace figure.
  - Pilot-length sensitivity (2k / 5k / 10k).
  - Learnable-θ variant vs fixed.
  - qk-lock removal.
  - Post-hoc EMA β-transfer (conditional on Phase 0a ≥ 0.05).
- **Experiments intentionally cut:**
  - Cross-resolution w[b] transfer study (per reviewer: re-pilot per setting; no transfer claim).
  - Alternate bucket counts (8 / 32) — fixed at 16.
  - Alternate bucket schemes (log-σ uniform, sigmoid-quantile) — fixed at equal-probability log-σ quantiles.
  - Separate `g_q/g_k/g_v` multipliers — out of method.
  - Detail heads / dual stacks / frequency decoders — out of non-goals.

## Experiment Blocks

### Block B1 — Main anchor result (4-cell matrix)
- **Claim tested:** C1, C2, A2.
- **Why this block exists:** This is THE headline FID number and the one-leg-lesion that isolates each contribution.
- **Dataset / split / task:** ImageNet-1k @ 256², train split; FID-50k vs `jit_in256_stats.npz` (included in repo).
- **Compared systems (all on JiT-B/16, 400 epochs, single forward, vanilla DDP):**
  - **A** — vanilla JiT (no MP, no σ-weighting) — baseline.
  - **B** — JiT + MPLinear only (σ-weighting off, pilot not run).
  - **C** — JiT + fixed σ-weighting only (MP off, 5k-step pilot inside the run).
  - **D** — JiT + MPLinear + fixed σ-weighting (full MP-JiT).
- **Metrics (primary):** FID-50k @ NFE 100 with Heun sampler and interval-CFG (matching vanilla JiT protocol); per-epoch FID mean-term + cov-term (via `analysis_fid_decomposition.py`); per-block weight-norm column mean trace; EMA-vs-online FID-5k gap (cadence every 10 epochs from ep 50).
- **Metrics (supportive):** per-bucket r²[b] curve at epochs {0, 100, 200, 300, 400} (single figure, 16 points × 5 epochs, not an ablation branch).
- **Setup details:** JiT-B/16 @ 256²; same cosine LR + warmup schedule; batch-size matched to vanilla; AdamW (WD=0 on MPLinear W and gains; standard AdamW elsewhere); MPLinear gains init 1.0; symmetric qk-lock slope 0.1 for ep 0..5; pilot = first 5k steps with `w≡1`, counted in headline compute, no optimizer reset; 3 seeds (C1 headline only — A and D; B and C on 1 seed each for budget).
- **Success criterion:**
  - **C1 passes if** D − A ≥ 0.20 FID AND D's mean-term trace flat after ep 300 AND D's stage gate passes at ep 100.
  - **C2 passes if** ≥ 50% of D − A is explained by mean-term reduction (direction-only; not a hard 80% threshold).
  - **C1 isolation passes if** D > max(B, C) by ≥ 0.05 FID (both legs contribute).
- **Failure interpretation:**
  - D − A < 0.20 → recipe underdelivers at B/16; report actual number, check stage-gate signals, diagnose which leg failed, do NOT scale to G/16.
  - B ≈ D or C ≈ D → one leg is redundant; contribution story collapses to one leg.
  - Stage gate fails at ep 100 → RETHINK per proposal.
- **Table / figure target:**
  - **Table 1** (main): FID-50k A/B/C/D + confidence intervals.
  - **Figure 1** (main): per-epoch mean-term + cov-term trace for A and D.
  - **Figure 2** (main): weight-norm column mean trace per-block for A, B, D.
  - **Figure 3** (appendix): per-bucket r²[b] curve at 5 epochs.
- **Priority:** MUST-RUN.

### Block B2 — Scaling to G/16 512² (headline secondary)
- **Claim tested:** C1 at scale.
- **Why this block exists:** The anchor success condition requires ≥ 0.15 FID @ G/16 matched compute. Without this, the paper does not meet its stated contract.
- **Dataset / split / task:** ImageNet-1k @ 512²; FID-50k vs `jit_in512_stats.npz`.
- **Compared systems:** **A'** vanilla JiT-G/16 @ 512², **D'** MP-JiT-G/16 @ 512². No B/C at G/16 (budget).
- **Metrics:** FID-50k @ NFE 100; mean-term trace; weight-norm trace. Stage gate evaluated at ep 100.
- **Setup details:** JiT-G/16; re-piloted σ-weighting (16 equal-probability log-σ quantiles under the G/16 training sampler); 400 epochs matched with vanilla. 1 seed (budget).
- **Success criterion:** D' − A' ≥ 0.15 FID AND D's mean-term trace flat after ep 300 AND G/16 stage gate passes.
- **Failure interpretation:** If B1 passed but B2 fails, recipe may be B/16-specific; analyze whether symptom (mean-term drift) was present at G/16 to begin with; paper falls back to 256²-only claim.
- **Table / figure target:** Table 2 (main): G/16 FID-50k A' vs D'.
- **Priority:** MUST-RUN (conditional on B1 passing stage gate at ep 100).

### Block B3 — Mechanism attribution & simplicity check
- **Claim tested:** C2, A1, simplicity-of-method.
- **Why this block exists:** Prevents "you just spent epoch budget differently" and "you just reweighted the loss" attacks.
- **Runs (all derived from B1 checkpoints; no new training):**
  - **Matched-compute pilot cost accounting:** measure wall-clock of D's 5k pilot vs A's vanilla first-5k steps. Verify the headline number uses compute-matched epochs.
  - **Mechanism decomposition:** apply `analysis_fid_decomposition.py` to A/B/C/D at {100, 200, 300, 400} ep.
  - **Per-bucket residual trace:** show pilot r²[b] curve (ep 0) vs trained r²[b] curve (ep 200) — the calibrated recipe flattens the curve by design.
- **Metrics:** Percentage of D−A from mean-term; bucket-curve flatness indicator (ratio max/min of r²[b] at ep 200 vs ep 0).
- **Success criterion (directional):** majority of D−A is mean-term; trained r²[b] curve at ep 200 is strictly flatter than pilot curve.
- **Failure interpretation:** If gain is mostly cov-term, C2 is false → main paper framing moves to "FID improvement" without mean-term attribution; mechanism claims demoted.
- **Table / figure target:** Table 3 (main, compact): decomposition percentages. Figure 4 (appendix): r²[b] flattening.
- **Priority:** MUST-RUN.

### Block B4 — Design-choice ablations (supplementary)
- **Claim tested:** A1 (not generic reweighting); design-choice stability.
- **Why this block exists:** Prevents reviewer asking "why 16 buckets?" "why fixed not learnable?" "why qk-lock?" in rebuttal.
- **Runs (all on JiT-B/16 @ 256², 200 epochs — half-budget to fit):**
  - **B4a** — MP-JiT-full with **learnable θ** (clamp-then-renormalize projection, ep 5 unfreeze). Compare to MP-JiT-fixed at 200ep.
  - **B4b** — **Pilot length sensitivity:** 2k / 5k / 10k pilot steps, holding everything else fixed. Compare calibrated `w[b]` stability.
  - **B4c** — **qk-lock removal:** MP-JiT-full without the symmetric barrier. Check training stability (gain trajectory, loss curves) and resulting FID.
- **Metrics:** FID-50k @ 200 ep; w[b] table distance (L2) across pilot lengths; gain trajectory max / min across training.
- **Success criterion (directional):**
  - B4a: learnable-θ does not meaningfully outperform fixed (supports Simplification #1 from reviewer — fixed stands alone).
  - B4b: w[b] stable between 5k and 10k (justifies 5k as default).
  - B4c: removing qk-lock destabilizes training at least at one layer (justifies its inclusion).
- **Failure interpretation:**
  - If learnable-θ is substantially better, restructure paper to include it as the main variant.
  - If qk-lock is removable, drop it from the method entirely.
- **Table / figure target:** Table 4 (appendix): design-choice ablation table.
- **Priority:** NICE-TO-HAVE (run only if B1 passes).

### Block B5 — Conditional: post-hoc EMA β-transfer (appendix-only)
- **Claim tested:** C3 (conditional); activates only if Phase 0a passes.
- **Why this block exists:** Low-risk, quick, appendix-level; adds one more headline line if it works.
- **Phase 0a (prerequisite):**
  - Existing JiT-B/16 256² snapshots from the vanilla training run (the `output/` or `output_imagenet_*` directories in the vanilla repo).
  - Reconstruct 32-snapshot uniform set; apply locked post-hoc EMA protocol (β chosen by min training-loss on held-out σ-grid, not FID); report best FID.
  - **Activation rule:** if Phase 0a ≥ 0.05 FID gain, proceed to B5; else delete appendix + all snapshot-saving language from the paper body.
- **Phase B5 (conditional):**
  - At B/16 256², enable `--enable-snapshots`; save 32 uniform snapshots over last 60% of epochs; apply post-hoc EMA with β chosen by same locked rule.
  - At G/16 512², enable `--enable-snapshots`; apply post-hoc EMA with β chosen at B/16 256² verbatim (no re-tuning at G/16).
- **Metrics:** FID-50k gain from β chosen at B/16 vs per-scale-tuned optimum at G/16.
- **Success criterion:** gap ≤ ±0.05 FID between B/16-chosen β and G/16-optimal β.
- **Failure interpretation:** β does not transfer → drop Claim 3; core paper unchanged.
- **Table / figure target:** Table 5 (appendix): β-transfer numbers.
- **Priority:** CONDITIONAL (Phase 0a gate).

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-hr) | Risk |
|-----------|------|------|---------------|---------------|------|
| **M0** Sanity | Verify code pipeline on tiny scale before burning compute | (i) MPLinear forward-backward unit test on toy tensor; (ii) σ-bucket loss on random σ sanity-check; (iii) 1k-step JiT-B/16 ImageNet training run; (iv) FID-5k pipeline correctness on vanilla snapshot | Training loss decreases; FID-5k on vanilla ≈ published JiT@step=1k | ~5 | Low (pure plumbing) |
| **M1** Phase 0a (cheap, optional) | Evaluate locked post-hoc EMA protocol on existing JiT-B/16 snapshots | Run `tools/post_hoc_ema.py` on existing vanilla JiT snapshots with 32-uniform, β-search on training-loss grid | If ≥ 0.05 FID gain → B5 activated; else delete EMA appendix | 1–3 | Very low |
| **M2** Phase 0b (stage-gate pilot) | Verify three-signal stage gate passes at ep 100 on MP-JiT-D @ B/16 256² before committing full 400ep | D-cell for 100 epochs; three signals logged; FID-5k cadence every 10ep from ep 50 | All three gate signals pass → proceed to M3; any fail → RETHINK | ~100 | Medium |
| **M3** Main method (B1) | Run full 400ep A/B/C/D matrix | A (3 seeds), D (3 seeds), B (1 seed), C (1 seed) | D−A ≥ 0.20 FID AND gate passes | ~1,270 | High — headline |
| **M4** Mechanism attribution (B3) | Decisive ablations for novelty, simplicity | Apply `analysis_fid_decomposition.py` to B1 checkpoints | ≥ 50% of D−A from mean-term (directional) | ~20 (analysis only) | Low |
| **M5** Scaling (B2) | Run G/16 512² A' and D' | A' (1 seed), D' (1 seed), full 400ep matched | D'−A' ≥ 0.15 FID AND gate passes | ~6,500 | High — scale transfer |
| **M6** Design-choice ablations (B4) | Supplementary tables | B4a / B4b / B4c each 200ep @ B/16 | Directional — informs paper structure, not gate | ~600 | Low-medium |
| **M7** Conditional appendix (B5) | β-transfer at scale | If Phase 0a passed, B5 at B/16 and G/16 | Gap ≤ ±0.05 FID | ~0 (amortized with B1/B2) | Low |
| **M8** Polish | Figures, tables, qualitative samples | 64-image panel at NFE 100 from A and D @ 256² and @ 512² | Human-inspectable quality improvement | ~5 | Very low |

**Total estimated compute:** ~7,500 GPU-hr core path (M0–M5), ~600 extra for M6. Fits ~8k budget.

## Compute and Data Budget

- **Total estimated GPU-hours:** ~7,800 (within 8k constraint).
- **Data preparation needs:** ImageNet-1k already present at `/gs/bs/hp190122/jiang/dataset` per previous commit; FID reference stats in `fid_stats/` already committed.
- **Human evaluation needs:** None required; qualitative panel is optional for camera-ready only.
- **Biggest bottleneck:** **M5 (G/16 512² scaling run)** at ~6,500 GPU-hr. Any regression or retry here blows the budget. Mitigation: do not launch M5 until M3 stage gate clears and M4 mechanism attribution is confirmed directional.

## Risks and Mitigations

- **Risk:** σ-bucket edges computed from the sampler CDF are off-by-one or inconsistent between pilot and main training.
  **Mitigation:** compute bucket edges ONCE at construction time in `Denoiser.__init__` from `(P_mean, P_std)`; store as non-persistent buffer; unit-test that bucket occupancy is approximately uniform (χ² within tolerance) over a 10k-step sample.

- **Risk:** MPLinear + `@torch.compile` (used on `JiTBlock.forward` and `FinalLayer.forward`) interact badly.
  **Mitigation:** validate forward/backward parity against nn.Linear in M0; if `@torch.compile` chokes on column-norm normalization, disable compile on affected blocks or implement MPLinear as a regular nn.Linear with a forward-hook wrapper.

- **Risk:** 5k-step pilot is noisy → static `w[b]` captures noise, not signal.
  **Mitigation:** M0 includes a pilot-noise sanity run (std of r²[b] across two independent 5k pilots); if inter-pilot std > 15% of r²[b] mean, extend pilot to 10k.

- **Risk:** qk-lock collapse (lower-barrier hit, gains shrink too much).
  **Mitigation:** symmetric barrier already in method; Phase 0b reports min-block gain trajectory; if any block touches the lower barrier, relax slope 0.1 → 0.05.

- **Risk:** DDP regression — vanilla DDP breaks under the new MPLinear column-normalization op.
  **Mitigation:** MPLinear implemented as a vanilla `nn.Module` with standard params; no `find_unused_parameters` needed; test 2-rank single-node DDP in M0 before multi-node; rank-0-only snapshot saving.

- **Risk:** Stage gate at ep 100 yields a noisy false-negative (RETHINK) when the full run would have succeeded.
  **Mitigation:** three correlated signals, each with pre-declared tolerances and noise floors; RETHINK only if ALL three fail; borderline (2 of 3 pass marginally) → extend Phase 0b by 50 epochs before deciding.

- **Risk:** `@torch.compile` instability with changing gain parameters.
  **Mitigation:** store gains as `nn.Parameter` tensors so autograd treats them uniformly; no dynamic shape or control flow changes.

## Execution on TSUBAME 4.0

All runs are launched via AGE job scripts under `edm2-jit/scripts/tsubame/`.
TSUBAME scheduler docs: <https://www.t4.cii.isct.ac.jp/docs/faq.en/scheduler/>.

### One-time setup

```bash
cd /path/to/edm2-jit
export TSUBAME_GROUP=tga-xxxxx   # your T4 group code from `show_group`
```

Adjust `scripts/tsubame/env.sh` if the conda env name is not `jit`, the CUDA
module path differs, or `DATA_PATH` / `FID_STATS_DIR` need overriding. The
default `DATA_PATH` is `/gs/bs/hp190122/jiang/dataset` (matching the vanilla
repo's last-known TSUBAME dataset commit).

### Submission map

| Milestone | Tracker IDs | Command | Resource | Wall per job | Chain length |
|-----------|-------------|---------|----------|--------------|--------------|
| **M0** unit + e2e sanity | R001, R002, R003 | `scripts/tsubame/submit_m0.sh` | `node_q` then `node_f` | 30 min + 24 h | 1 |
| **M2** gate pilot | R020 | `scripts/tsubame/submit_gate.sh` | `node_f=1` | 24 h | `N_CHAIN=4` |
| **M3** 4-cell × 3-seed matrix | R030–R037 | `scripts/tsubame/submit_matrix.sh` | `node_f=1` | 24 h | `N_CHAIN=14` |
| **M5** G/16 512² scaling pair | R050, R051 | `scripts/tsubame/submit_scaling.sh` | `node_f=1` | 24 h | `N_CHAIN=20` |

`node_f=1` reserves one H100 node (4 × H100 80 GB) and the submitter launches
`torchrun --nproc_per_node=4`.

### Chained resubmission for long trainings

Each long-training submitter enqueues `N_CHAIN` copies of the same AGE job
back-to-back via `qsub -hold_jid`. `main_jit.py` auto-resumes from
`$OUTPUT_DIR/checkpoint-last.pth` when present, so:

- A job killed by the 24 h walltime continues exactly where it stopped.
- A chain link whose training is already past `--epochs` exits in <1 min —
  surplus links are cheap no-ops. Pick `N_CHAIN` generously.

### Partial re-runs and recovery

```bash
# resume only the two Cell-D failures:
scripts/tsubame/submit_matrix.sh R035_D_full_seed1 R036_D_full_seed2

# kill a chain (must also qdel the dependent links):
qdel <first_job_id>
```

### Per-run configs

`scripts/tsubame/configs/R<ID>_*.sh` define `RUN_ID`, `MODEL`, `EPOCHS`,
`SEED`, `BATCH_SIZE`, `IMG_SIZE`, `EXTRA_ARGS`. Each file maps 1-to-1 to a
row in `EXPERIMENT_TRACKER.md`. To add a new experiment (e.g. pilot-length
ablation R061), drop another config file in that directory and submit via
`submit_matrix.sh R061_<name>`.

### Logs and monitoring

Stdout/stderr merge to `logs/<job_name>.o<job_id>`. TensorBoard output goes
to `$OUTPUT_DIR/` (default `output/<RUN_ID>/`). MP-JiT diagnostic scalars
logged by `engine_jit._log_mp_jit_diagnostics` cover per-block column norms,
gains, qk-gain product, per-bucket r², and bucket weights.

## Final Checklist
- [x] Main paper tables are covered: Table 1 (4-cell), Table 2 (G/16), Table 3 (decomposition).
- [x] Novelty is isolated: B vs C vs D one-leg-lesion in B1.
- [x] Simplicity is defended: static vs learnable θ ablation in B4a; qk-lock removal in B4c; reviewer Simplification #1 (drop learnable-θ entirely) explicitly covered.
- [x] Frontier contribution is justified or explicitly not claimed: proposal already says "same optimization lane as EDM2/SiD2, smaller and more targeted"; no frontier primitive added.
- [x] Nice-to-have runs (B4, B5) are separated from must-run runs (B1, B2, B3).
- [x] Anchor: all experiments trace to Problem Anchor claims C1/C2 or anti-claims A1/A2.
- [x] TSUBAME submission scripts committed under `edm2-jit/scripts/tsubame/` with chained-resubmission support for 400-epoch runs.
