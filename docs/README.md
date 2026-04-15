# MP-JiT documentation

Self-contained copy of the research artifacts that gated this repo, pulled
from the vanilla JiT workspace on 2026-04-15. Source of truth is here;
`/home/hzy980512/JiT/refine-logs/` in the author's workspace is the original
but that directory is not published.

## Entry points

- **[refine-logs/FINAL_PROPOSAL.md](refine-logs/FINAL_PROPOSAL.md)** — the
  locked-in MP-JiT method specification (READY 9.0/10 after 5 rounds of
  review). Read this first to understand what the code implements.
- **[refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)** —
  claim-driven experiment roadmap with TSUBAME submission instructions.
- **[refine-logs/EXPERIMENT_TRACKER.md](refine-logs/EXPERIMENT_TRACKER.md)** —
  flat table of all runs (R001–R081) with priority and status.

## Context

- **[refine-logs/REFINEMENT_REPORT.md](refine-logs/REFINEMENT_REPORT.md)** —
  why the method looks the way it does: what survived refinement, what was
  killed, and the anchor that was preserved across all 5 rounds.
- **[refine-logs/REVIEW_SUMMARY.md](refine-logs/REVIEW_SUMMARY.md)** — score
  trajectory and major transitions between rounds.
- **[refine-logs/score-history.md](refine-logs/score-history.md)** —
  per-dimension score table.

## Raw refinement history

- `refine-logs/round-0-initial-proposal.md` — starting point.
- `refine-logs/round-{1..5}-review.md` — GPT-5.4 reviewer verdicts.
- `refine-logs/round-{1..4}-refinement.md` — per-round rewritten proposal.
- `refine-logs/REFINE_STATE.json` — checkpoint state (status: completed).

## Where the code lives

- Method: `../model_jit.py` (MPLinear, qk-lock) and `../denoiser.py` (σ-bucket
  pilot + static w buffer).
- Training hooks: `../engine_jit.py` (qk-lock loss term, diagnostic logger).
- CLI surface: `../main_jit.py` (`--use_mp`, `--use_sigma_weight`,
  `--pilot_steps`, `--qk_lock_epochs`, `--qk_lock_slope`).
- TSUBAME launch: `../scripts/tsubame/`.
- Unit tests: `../tests/`.
