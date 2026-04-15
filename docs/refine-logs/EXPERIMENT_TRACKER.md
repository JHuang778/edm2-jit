# Experiment Tracker: MP-JiT

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | MPLinear forward/backward parity vs nn.Linear on toy tensor | unit test | — | max abs err, grad-check | MUST | TODO | run pytest on `edm2-jit/tests/test_mplinear.py` |
| R002 | M0 | σ-bucket indexer unit test (CDF-based quantiles, χ² uniformity) | unit test | — | χ² p-value | MUST | TODO | run `edm2-jit/tests/test_sigma_buckets.py` |
| R003 | M0 | End-to-end 1k-step JiT-B/16 training sanity | MP-JiT-D | ImageNet-1k 256² | loss ↓, no NaN | MUST | TODO | tiny run on 1-node 4-GPU |
| R004 | M0 | FID-5k pipeline sanity on a vanilla JiT snapshot | vanilla JiT | IN-1k 256² | FID-5k ≈ published | MUST | TODO | reuse existing vanilla snapshot |
| R005 | M0 | 2-rank DDP sanity of MPLinear + bucketed loss | MP-JiT-D | IN-1k 256² | no hang, loss ↓ | MUST | TODO | `torchrun --nproc=2` |
| R006 | M0 | Pilot-noise sanity: 2× independent 5k pilots | MP-JiT pilot only | IN-1k 256² | std r²[b] / mean < 15% | MUST | TODO | if fails → extend pilot to 10k |
| R010 | M1 | Phase 0a post-hoc EMA on existing vanilla JiT-B/16 snapshots | vanilla JiT | IN-1k 256² | ΔFID vs best existing EMA | NICE | TODO | if ≥ 0.05 → activate B5; else delete EMA appendix |
| R020 | M2 | Phase 0b stage-gate pilot: MP-JiT-D for 100 ep at B/16 256² | MP-JiT-D | IN-1k 256² | mean-term slope; ‖W‖ slope; FID-5k gap cadence | MUST | TODO | all three signals logged |
| R030 | M3 / B1 | A vanilla 400ep seed 1 | JiT-B/16 vanilla | IN-1k 256² | FID-50k, mean/cov traces, ‖W‖ trace | MUST | TODO | headline baseline |
| R031 | M3 / B1 | A vanilla 400ep seed 2 | JiT-B/16 vanilla | IN-1k 256² | FID-50k | MUST | TODO | — |
| R032 | M3 / B1 | A vanilla 400ep seed 3 | JiT-B/16 vanilla | IN-1k 256² | FID-50k | MUST | TODO | — |
| R033 | M3 / B1 | B MP-only 400ep seed 1 | JiT-B/16 + MPLinear | IN-1k 256² | FID-50k, ‖W‖ trace | MUST | TODO | isolates MP leg |
| R034 | M3 / B1 | C σ-calib-only 400ep seed 1 | JiT-B/16 + fixed σ-weighting | IN-1k 256² | FID-50k, per-bucket r² | MUST | TODO | isolates σ leg |
| R035 | M3 / B1 | D full MP-JiT 400ep seed 1 | JiT-B/16 full | IN-1k 256² | FID-50k, all traces | MUST | TODO | headline |
| R036 | M3 / B1 | D full MP-JiT 400ep seed 2 | JiT-B/16 full | IN-1k 256² | FID-50k | MUST | TODO | — |
| R037 | M3 / B1 | D full MP-JiT 400ep seed 3 | JiT-B/16 full | IN-1k 256² | FID-50k | MUST | TODO | — |
| R040 | M4 / B3 | FID-decomposition (mean / cov) on R030–R037 checkpoints | analysis only | IN-1k 256² | % D−A from mean-term at {100, 200, 300, 400} ep | MUST | TODO | ≥ 50% → C2 passes |
| R041 | M4 / B3 | Per-bucket r²[b] trace figure | analysis only | IN-1k 256² | flatness ratio at {0, 100, 200, 300, 400} | MUST | TODO | supports A1 |
| R050 | M5 / B2 | A' vanilla JiT-G/16 400ep seed 1 @ 512² | JiT-G/16 vanilla | IN-1k 512² | FID-50k, mean/cov traces | MUST | BLOCKED on M3 gate | heavy |
| R051 | M5 / B2 | D' MP-JiT-G/16 400ep seed 1 @ 512² | JiT-G/16 full | IN-1k 512² | FID-50k, all traces | MUST | BLOCKED on M3 gate | heavy |
| R060 | M6 / B4a | Learnable-θ variant MP-JiT-full 200ep B/16 256² | MP-JiT + learnable θ | IN-1k 256² | FID-50k, θ trajectory | NICE | TODO | clamp-then-renormalize projection |
| R061 | M6 / B4b | Pilot-length ablation: 2k pilot | MP-JiT-full | IN-1k 256² | FID-50k, w[b] L2 to 5k | NICE | TODO | — |
| R062 | M6 / B4b | Pilot-length ablation: 10k pilot | MP-JiT-full | IN-1k 256² | FID-50k, w[b] L2 to 5k | NICE | TODO | — |
| R063 | M6 / B4c | qk-lock removal: MP-JiT-full no barrier | MP-JiT-full no-qk-lock | IN-1k 256² | FID-50k, gain trajectory | NICE | TODO | — |
| R070 | M7 / B5 | β-transfer: β chosen at B/16 256² applied at G/16 512² | MP-JiT + post-hoc EMA | IN-1k 512² | ΔFID-50k vs per-scale-tuned β | CONDITIONAL | BLOCKED on R010 | — |
| R080 | M8 | Qualitative panel: 64-sample grid from A and D @ 256² | sampling | — | visual | NICE | TODO | camera-ready only |
| R081 | M8 | Qualitative panel: 64-sample grid from A' and D' @ 512² | sampling | — | visual | NICE | TODO | camera-ready only |
