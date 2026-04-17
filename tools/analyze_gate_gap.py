"""
Diagnostic for signal #3 (EMA-vs-online FID gap) — reads tfevents and prints
per-epoch fid_ema, fid_online, gap, and the gap's slope over the gate window.

Used to disambiguate a signal-#3 NO-GO: if the gap is closing as training
progresses, it's healthy convergence (EMA smooths volatile online params),
not a real instability.

Usage (CPU-only, login node OK):
    python tools/analyze_gate_gap.py --run_dir output/R020_D_gate_pilot
"""

import argparse
import os
from glob import glob

import numpy as np


def _load_scalars(run_dir):
    from tensorboard.backend.event_processing import event_accumulator
    if not glob(os.path.join(run_dir, 'events.out.tfevents.*')):
        raise SystemExit(f"No tfevents in {run_dir}")
    ea = event_accumulator.EventAccumulator(
        run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea


def _series(ea, tag):
    if tag not in ea.Tags()['scalars']:
        return {}
    return {ev.step: ev.value for ev in ea.Scalars(tag)}


def _find_tag(ea, prefix):
    cands = [t for t in ea.Tags()['scalars'] if t.startswith(prefix)]
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ep_min', type=int, default=50)
    ap.add_argument('--ep_max', type=int, default=99)
    args = ap.parse_args()

    ea = _load_scalars(args.run_dir)

    ema_tag = _find_tag(ea, 'fid_ema_')
    online_tag = _find_tag(ea, 'fid_online_')
    is_ema_tag = _find_tag(ea, 'is_ema_')
    gap_tag = 'stage_gate/ema_vs_online_fid_gap'

    print(f"run_dir: {args.run_dir}")
    print(f"  fid_ema    tag: {ema_tag}")
    print(f"  fid_online tag: {online_tag}")
    print(f"  is_ema     tag: {is_ema_tag}")
    print(f"  gap        tag: {gap_tag}")

    fid_ema = _series(ea, ema_tag) if ema_tag else {}
    fid_online = _series(ea, online_tag) if online_tag else {}
    is_ema = _series(ea, is_ema_tag) if is_ema_tag else {}
    gap = _series(ea, gap_tag)

    rows = []
    for ep in sorted(set(fid_ema) | set(fid_online) | set(gap)):
        if args.ep_min <= ep <= args.ep_max:
            rows.append((ep, fid_ema.get(ep), fid_online.get(ep),
                         gap.get(ep), is_ema.get(ep)))
    if not rows:
        raise SystemExit("No epochs in gate window; was --log_ema_online_gap set?")

    print(f"\nGate window ep {args.ep_min}..{args.ep_max}:")
    print(f"  {'ep':>4}  {'fid_ema':>8}  {'fid_online':>10}  {'gap':>9}  "
          f"{'ratio':>6}  {'is_ema':>7}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*6}  {'-'*7}")
    for ep, fe, fo, g, ise in rows:
        fe_s = f"{fe:.3f}" if fe is not None else "   —   "
        fo_s = f"{fo:.3f}" if fo is not None else "   —   "
        g_s  = f"{g:+.3f}" if g is not None else "   —   "
        r_s  = f"{(fo/fe):.2f}" if (fe and fo and fe > 0) else "  —  "
        i_s  = f"{ise:.2f}" if ise is not None else "  —  "
        print(f"  {ep:>4}  {fe_s:>8}  {fo_s:>10}  {g_s:>9}  {r_s:>6}  {i_s:>7}")

    xs = np.array([ep for ep, _, _, g, _ in rows if g is not None], dtype=np.float64)
    ys = np.array([g  for _, _, _, g, _ in rows if g is not None], dtype=np.float64)
    if xs.size >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        print(f"\nGap linear fit: gap(ep) = {slope:+.4f} * ep + {intercept:+.2f}")
        print(f"  slope > 0 → gap closing (healthy, online catching up to EMA)")
        print(f"  slope ≈ 0 → gap flat (borderline)")
        print(f"  slope < 0 → gap widening (online diverging — real instability)")
        print(f"  extrapolated gap@ep200: {slope*200 + intercept:+.2f}")
        print(f"  extrapolated gap@ep400: {slope*400 + intercept:+.2f}")

    # Quick verdict summary
    if xs.size >= 2:
        first_gap, last_gap = ys[0], ys[-1]
        closing = abs(last_gap) < abs(first_gap) * 0.9  # 10%+ closure
        max_ratio = max((fo / fe) for _, fe, fo, _, _ in rows
                        if fe and fo and fe > 0) if any(
                            fe and fo and fe > 0 for _, fe, fo, _, _ in rows) else None
        print()
        print(f"  |gap| ep{int(xs[0])}={abs(first_gap):.2f} → ep{int(xs[-1])}={abs(last_gap):.2f}"
              f"  ({'closing' if closing else 'NOT closing'})")
        if max_ratio is not None:
            sev = 'OK' if max_ratio < 2.0 else ('WARN' if max_ratio < 4.0 else 'BAD')
            print(f"  max fid_online/fid_ema ratio in window: {max_ratio:.2f}x  [{sev}]")


if __name__ == '__main__':
    main()
