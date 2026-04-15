# TSUBAME 4.0 submission scripts

Job scripts for the Altair Grid Engine scheduler on TSUBAME 4.0.
Reference: <https://www.t4.cii.isct.ac.jp/docs/faq.en/scheduler/>.

## One-time setup

```bash
export TSUBAME_GROUP=tga-xxxxx     # your T4 group code; check: `show_group`
# optional overrides (see scripts/tsubame/env.sh):
#   DATA_PATH=/gs/bs/hp190122/jiang/dataset
#   FID_STATS_DIR=$PWD/fid_stats
```

Edit `env.sh` if your conda environment name is not `jit` or CUDA module is
named differently on your allocation.

## Submission order (mirrors EXPERIMENT_PLAN.md milestones)

| Stage | Command | Resource | Wall (per job) |
|-------|---------|----------|----------------|
| **M0** sanity (R001–R003) | `scripts/tsubame/submit_m0.sh` | `node_q` + `node_f` | 30 min + 24 h |
| **M2** gate pilot (R020) | `scripts/tsubame/submit_gate.sh` | `node_f` × `N_CHAIN=4` | 24 h each |
| **M3** main matrix (R030–R037) | `scripts/tsubame/submit_matrix.sh` | `node_f` × `N_CHAIN=14` | 24 h each |
| **M5** G/16 512² (R050–R051) | `scripts/tsubame/submit_scaling.sh` | `node_f` × `N_CHAIN=20` | 24 h each |

## How chaining works

Each submitter uses `qsub -hold_jid <prev>` to enqueue `N_CHAIN` copies of the
same job back-to-back. `main_jit.py` auto-resumes from
`$OUTPUT_DIR/checkpoint-last.pth` when present, so:

- a job interrupted by walltime continues exactly where it left off;
- a chain link whose training already completed exits in under a minute.

`N_CHAIN` is just an upper bound on how many 24 h slots may be needed. Pick
generously — surplus links are cheap no-ops.

## Submitting a subset

```bash
# only Cell D seeds 1 and 2:
scripts/tsubame/submit_matrix.sh R035_D_full_seed1 R036_D_full_seed2
```

## Monitoring

```bash
qstat                        # your jobs
qstat -f -j <job_id>         # job detail
tail -f logs/<job_name>.o<job_id>   # live stdout
```

## Cancel a chain

```bash
qdel <first_job_id>          # then later links' -hold_jid becomes orphan and
                             # must also be qdel'd; use qstat | awk to batch.
```

## Config matrix

All config files live in `configs/`. Each defines `RUN_ID`, `MODEL`, `EPOCHS`,
`SEED`, and `EXTRA_ARGS` — the schema consumed by `run_jit.sh`. Runs map 1-to-1
to the IDs in `refine-logs/EXPERIMENT_TRACKER.md`.
