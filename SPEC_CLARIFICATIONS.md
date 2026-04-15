# MP-JiT spec / implementation clarifications

Tracks intentional deviations from `refine-logs/FINAL_PROPOSAL.md` where the
frozen spec's terminology does not one-to-one translate to the JiT codebase.
Each entry names a spec line, the implementation choice, and the justification.

## 1. "x-pred σ-weighting" → v-space residual calibration

**Spec text (FINAL_PROPOSAL.md §Pilot-calibrated fixed σ-weighting):**
```
L = mean over batch of  w[ b(σ_i) ] · ‖x̂_i − x_i‖²
```

**Implementation (denoiser.py `forward`):**
```
m = ((v − v_pred) ** 2).mean(dim=(1,2,3))          # per-sample v-MSE
loss = (w[b(σ)] · m).mean()                         # post-pilot
```

**Why:** JiT is a flow-matching model that trains a v-MSE loss with an x-pred
network head. The identity `v − v_pred = (x − x_pred) / (1 − t)` means
v-space MSE equals x-space MSE inflated by `(1 − t)^{-2}` — a strongly σ-dependent
factor already baked into JiT's baseline gradient.

If the pilot calibrated on x-MSE while training optimized v-MSE, the weights
`w[b]` would equalize the wrong signal. Worse, the A/B/C/D ablation becomes
**confounded**: cell A (vanilla) uses v-MSE, cell D (weighted) would use a
different loss function entirely, not just a reweighting.

**Resolution:** the MP-JiT contribution is "pilot-calibrated per-σ-bucket loss
weighting". Both the pilot r² statistic and the post-pilot weighted loss
operate on the **same** residual estimator (v-space), so `w[b]` is a pure
reweighting of the baseline loss — cell D = cell A + `w[b]` factor. The EDM2
paper's "x-pred" terminology was inherited into the refine-logs spec; in
JiT's flow-matching operationalization the equivalent is v-space.

## 2. `qk_s_init` omits `1/sqrt(d_head)` (equivalent to spec)

**Spec text (FINAL_PROPOSAL.md §MPLinear):**
```
s_init = (q.gain · k.gain) / sqrt(d_head)           # frozen at init
```

**Implementation (model_jit.py `JiT.__init__`):**
```
s_init = stack([(blk.attn.q.gain * blk.attn.k.gain).reshape(()) for blk in ...])
```

**Why:** the `1/sqrt(d_head)` factor appears on both sides of every
ReLU-barrier comparison (`s` and `s_init`) and cancels. Omitting it does not
change the penalty value and keeps the buffer dimensionally identical to the
runtime product. Equivalent formula.

## 3. qk-lock active epoch range is [0, qk_lock_epochs] inclusive

**Spec text (FINAL_PROPOSAL.md §MPLinear):**
```
Stability penalty (epochs 0..5, symmetric): ... Released after epoch 5.
```

**Implementation (model_jit.py `qk_lock_penalty`):** gated by
`epoch > self.qk_lock_epochs`, so epochs 0..5 inclusive (6 active epochs)
with `band = 0.1 · (ep/5)` reaching its full value 0.1 at `ep=5`. Released at
epoch 6 onwards.

**Why:** with `epoch >= qk_lock_epochs` (the earlier code) the band never
reaches 0.1 because `ep=5` is skipped. The spec's band schedule implies the
full-width barrier is applied on the final lock epoch, so inclusive gating
matches the arithmetic.

## 4. qk-lock penalty computed in fp32 (outside autocast)

Not a spec deviation — the spec is silent on numerics. But the `1e-3 · ReLU(ε)²`
term is too small to survive bf16 reductions; wrapping the penalty in
`autocast(enabled=False)` keeps it faithful without changing the formula.

## 5. Unseen-bucket safeguard in `_finalize_pilot`

With a 5k-step pilot over 16 equal-probability buckets and >1M samples seen in
standard multi-GPU runs, every bucket is overwhelmingly likely to be non-empty.
The safeguard (unseen → `w[b] = 1`, median/mean over seen buckets only) exists
for debug runs with `pilot_steps` shrunk to tens of steps. Spec-neutral.
