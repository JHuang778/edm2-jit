"""
M1 / R010 — Phase 0a post-hoc EMA α-sweep on vanilla JiT-B/16 snapshots.

For each α in a uniform grid, synthesizes a new EMA state as
    θ(α) = (1 − α) · θ_ema1 + α · θ_ema2
(covering the β-range between the two stored decays 0.9999 and 0.9996),
installs it as the sampling EMA, and calls engine_jit.evaluate for FID-5k.

Gate criterion (from EXPERIMENT_PLAN.md):
    ΔFID = min_α FID(α) − FID(α=0, i.e. stored ema1) ≤ −0.05 → activate B5.
    Otherwise, delete the EMA appendix.

Usage (launched by scripts/slurm/run_m1_post_hoc_ema.sh via torchrun):
    torchrun --nproc_per_node=1 tools/post_hoc_ema.py \
        --ckpt /mnt/nfs/.../jit_archived/output_imagenet_256/checkpoint-last.pth \
        --output_dir output/R010_post_hoc_ema \
        --n_alpha 32 --num_images 5000 --gen_bsz 64
"""

import argparse
import copy
import os
import sys

import torch

# allow `from denoiser import Denoiser` when launched from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util.misc as misc
from denoiser import Denoiser
from engine_jit import evaluate


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_alpha', type=int, default=32)
    ap.add_argument('--num_images', type=int, default=5000)
    ap.add_argument('--gen_bsz', type=int, default=64)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--class_num', type=int, default=1000)
    ap.add_argument('--cfg', type=float, default=None,
                    help='CFG scale; default: inherited from ckpt args')
    ap.add_argument('--num_sampling_steps', type=int, default=None)
    ap.add_argument('--dist_url', default='env://')
    ap.add_argument('--dist_on_itp', action='store_true')
    return ap.parse_args()


def main():
    cli = parse_cli()

    ckpt = torch.load(cli.ckpt, map_location='cpu', weights_only=False)
    args = ckpt['args']

    args.output_dir = cli.output_dir
    args.num_images = cli.num_images
    args.gen_bsz = cli.gen_bsz
    args.img_size = cli.img_size
    args.class_num = cli.class_num
    if cli.cfg is not None:
        args.cfg = cli.cfg
    if cli.num_sampling_steps is not None:
        args.num_sampling_steps = cli.num_sampling_steps
    args.dist_url = cli.dist_url
    args.dist_on_itp = cli.dist_on_itp
    args.evaluate_gen = True
    args.online_eval = False
    args.resume = ''
    # MP-specific args present in the new codebase, absent from the vanilla ckpt
    for k, v in (('use_mp', False),
                 ('qk_lock_epochs', 5),
                 ('qk_lock_slope', 0.1),
                 ('use_sigma_weight', False),
                 ('pilot_steps', 5000),
                 ('log_ema_online_gap', False),
                 ('keep_gate_samples', False),
                 ('log_freq', 100)):
        if not hasattr(args, k):
            setattr(args, k, v)

    misc.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = None
    if misc.is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(log_dir=args.output_dir)

    # Build model with ckpt's exact config so state_dict matches bit-identically.
    model = Denoiser(args).to('cuda')
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu],
        broadcast_buffers=False,
    )
    m = model_ddp.module

    # Vanilla JiT ckpts were saved under torch.compile, so 'model' keys have
    # '_orig_mod.' prefix if args.compile was True. Strip if needed.
    def _strip_orig_mod(sd):
        return {k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k: v
                for k, v in sd.items()}

    m.load_state_dict(_strip_orig_mod(ckpt['model']), strict=True)

    ema1_sd = _strip_orig_mod(ckpt['model_ema1'])
    ema2_sd = _strip_orig_mod(ckpt['model_ema2'])
    named = list(m.named_parameters())
    ema1 = [ema1_sd[name].to('cuda', non_blocking=True) for name, _ in named]
    ema2 = [ema2_sd[name].to('cuda', non_blocking=True) for name, _ in named]
    if misc.is_main_process():
        print(f"Loaded ckpt epoch={ckpt.get('epoch', '?')}, "
              f"{len(ema1)} EMA params.")

    alphas = torch.linspace(0.0, 1.0, cli.n_alpha).tolist()
    csv_path = os.path.join(args.output_dir, 'post_hoc_ema_sweep.csv')
    if misc.is_main_process():
        with open(csv_path, 'w') as f:
            f.write('step,alpha,fid,inception_score\n')

    # Record which decay a given α corresponds to (assuming both EMA were at
    # same step, a 2-point blend synthesizes a β on the exponential curve).
    # ema_decay1 = 0.9999, ema_decay2 = 0.9996; β(α) ≈ β1^(1-α) · β2^α.
    beta1, beta2 = args.ema_decay1, args.ema_decay2
    results = []
    for i, alpha in enumerate(alphas):
        beta_eff = (beta1 ** (1.0 - alpha)) * (beta2 ** alpha)
        blend = [((1.0 - alpha) * p1 + alpha * p2).contiguous()
                 for p1, p2 in zip(ema1, ema2)]
        m.ema_params1 = blend

        with torch.no_grad():
            fid, isc = evaluate(
                m, args, epoch=i,
                batch_size=cli.gen_bsz, log_writer=log_writer, use_ema=True,
            )

        if misc.is_main_process():
            print(f"[{i:02d}/{cli.n_alpha}] α={alpha:.4f} "
                  f"β_eff≈{beta_eff:.6f} → FID={fid:.4f}  IS={isc:.4f}",
                  flush=True)
            with open(csv_path, 'a') as f:
                f.write(f'{i},{alpha:.6f},{fid:.4f},{isc:.4f}\n')
            log_writer.add_scalar('post_hoc_ema/fid', fid, i)
            log_writer.add_scalar('post_hoc_ema/is', isc, i)
            log_writer.add_scalar('post_hoc_ema/beta_eff', beta_eff, i)
            results.append((alpha, beta_eff, fid, isc))

        del blend
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    if misc.is_main_process() and results:
        results_sorted = sorted(results, key=lambda r: r[2])
        best = results_sorted[0]
        baseline = results[0]  # α=0 == stored ema1
        delta = best[2] - baseline[2]
        print("=" * 60)
        print(f"Baseline (α=0, stored ema1): FID={baseline[2]:.4f}")
        print(f"Best blend:  α={best[0]:.4f}  β_eff={best[1]:.6f}  "
              f"FID={best[2]:.4f}  IS={best[3]:.4f}")
        print(f"ΔFID = {delta:+.4f}  "
              f"(gate threshold: ≤ -0.05 → activate B5)")
        print(f"DECISION: {'ACTIVATE B5' if delta <= -0.05 else 'DELETE EMA APPENDIX'}")
        with open(os.path.join(args.output_dir, 'verdict.txt'), 'w') as f:
            f.write(f"baseline_fid={baseline[2]:.4f}\n"
                    f"best_alpha={best[0]:.4f}\n"
                    f"best_beta_eff={best[1]:.6f}\n"
                    f"best_fid={best[2]:.4f}\n"
                    f"delta_fid={delta:+.4f}\n"
                    f"decision={'ACTIVATE_B5' if delta <= -0.05 else 'DELETE_APPENDIX'}\n")

    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    main()
