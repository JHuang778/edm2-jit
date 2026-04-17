"""
Phase 0b stage-gate decision for MP-JiT (M2 → M3).

Reads three signals recorded during the ep-100 pilot and emits GO / NO-GO:

  1. mean-term slope   — linreg slope of mean_term over ep 50..100 from
                         ssd/tmp/<run>/mean_cov_decomposition.csv.
                         GO if slope < 0 (mean-matching improving, not stuck).
  2. weight-norm slope — linreg slope of mean MP col_norm across MP blocks
                         over ep 50..100 from the TensorBoard event file.
                         GO if |slope| < WN_TOL (norms not drifting).
  3. FID gap           — final value of stage_gate/ema_vs_online_fid_gap at ep 100.
                         GO if |gap| < FID_GAP_TOL (EMA and online agree).

Usage:
    python experiment_gate.py --run_dir output_dir/R020_D_gate_pilot \
        [--csv ssd/tmp/R020_D_gate_pilot/mean_cov_decomposition.csv]
"""

import argparse
import csv
import os
import sys
from glob import glob

import numpy as np


WN_TOL = 0.05        # |Δ col_norm_mean / ep| tolerated (on normalized scale)
FID_GAP_TOL = 1.0    # FID units
# Training loop is zero-based (epoch ∈ [0, EPOCHS)), so the final
# human-"epoch 100" checkpoint is saved as ep99. Window stays inclusive.
GATE_WINDOW = (50, 99)


def _linreg_slope(xs, ys):
    if len(xs) < 2:
        return float('nan')
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    m, _ = np.polyfit(xs, ys, 1)
    return float(m)


def signal_1_mean_term_slope(csv_path):
    if not os.path.exists(csv_path):
        return None, f"CSV not found: {csv_path}"
    xs, ys = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ep = int(row['epoch'])
            if GATE_WINDOW[0] <= ep <= GATE_WINDOW[1]:
                xs.append(ep)
                ys.append(float(row['mean_term']))
    if len(xs) < 3:
        return None, f"Only {len(xs)} rows in gate window; need ≥3."
    slope = _linreg_slope(xs, ys)
    return slope, f"mean_term slope over ep {xs[0]}..{xs[-1]} = {slope:+.4f} / ep"


def signal_2_weight_norm_slope(run_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None, "tensorboard not installed; skipping signal #2"
    event_files = glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None, f"no tfevents in {run_dir}"
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = [t for t in ea.Tags()['scalars']
            if t.startswith('mp/block') and t.endswith('/q_col_norm_mean')]
    if not tags:
        return None, "no mp/block*/q_col_norm_mean scalars found"
    eps_to_vals = {}
    for tag in tags:
        for ev in ea.Scalars(tag):
            ep = ev.step // 1000
            if GATE_WINDOW[0] <= ep <= GATE_WINDOW[1]:
                eps_to_vals.setdefault(ep, []).append(ev.value)
    if len(eps_to_vals) < 3:
        return None, f"only {len(eps_to_vals)} epochs of weight-norm data in window"
    xs = sorted(eps_to_vals.keys())
    ys = [float(np.mean(eps_to_vals[x])) for x in xs]
    slope = _linreg_slope(xs, ys)
    return slope, f"mean q-col_norm slope over ep {xs[0]}..{xs[-1]} = {slope:+.4f} / ep"


def signal_3_ema_online_gap(run_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None, "tensorboard not installed; skipping signal #3"
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tag = 'stage_gate/ema_vs_online_fid_gap'
    if tag not in ea.Tags()['scalars']:
        return None, f"{tag} not logged (was --log_ema_online_gap passed?)"
    last = ea.Scalars(tag)[-1]
    return last.value, f"{tag}@ep{last.step} = {last.value:+.4f}"


def decide(signal_1, signal_2, signal_3):
    """Three-signal AND gate. Any None/unknown → abstain (NO-GO)."""
    reasons = []

    if signal_1 is None:
        s1_ok = False; reasons.append("signal#1 unavailable")
    else:
        s1_ok = signal_1 < 0
        reasons.append(f"signal#1 {'GO' if s1_ok else 'NO-GO'} (slope={signal_1:+.4f})")

    if signal_2 is None:
        s2_ok = False; reasons.append("signal#2 unavailable")
    else:
        s2_ok = abs(signal_2) < WN_TOL
        reasons.append(f"signal#2 {'GO' if s2_ok else 'NO-GO'} "
                       f"(|slope|={abs(signal_2):.4f} vs tol {WN_TOL})")

    if signal_3 is None:
        s3_ok = False; reasons.append("signal#3 unavailable")
    else:
        s3_ok = abs(signal_3) < FID_GAP_TOL
        reasons.append(f"signal#3 {'GO' if s3_ok else 'NO-GO'} "
                       f"(|gap|={abs(signal_3):.4f} vs tol {FID_GAP_TOL})")

    return (s1_ok and s2_ok and s3_ok), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True,
                    help='output_dir of the pilot run (tfevents live here)')
    ap.add_argument('--csv', default=None,
                    help='mean_cov_decomposition.csv; defaults to '
                         'ssd/tmp/<run_dir>/mean_cov_decomposition.csv')
    args = ap.parse_args()

    # run_dir is normally `output/<RUN_ID>`; sample dumps and CSV live under
    # `ssd/tmp/<args.output_dir>/...`, mirroring evaluate()'s save path.
    csv_path = args.csv or os.path.join(
        'ssd/tmp', args.run_dir.rstrip('/'),
        'mean_cov_decomposition.csv')

    s1, m1 = signal_1_mean_term_slope(csv_path)
    s2, m2 = signal_2_weight_norm_slope(args.run_dir)
    s3, m3 = signal_3_ema_online_gap(args.run_dir)
    print(m1); print(m2); print(m3)

    go, reasons = decide(s1, s2, s3)
    print("---")
    for r in reasons:
        print(" ", r)
    print("---")
    print("DECISION:", "GO — authorize full 400ep" if go else "NO-GO — abort / replan")
    sys.exit(0 if go else 1)


if __name__ == '__main__':
    main()
