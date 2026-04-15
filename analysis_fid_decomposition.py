"""
FID decomposition: mean_term + cov_term on an EMA sample folder.

  mean_term = ||mu_gen - mu_ref||^2
  cov_term  = Tr(Sigma_gen + Sigma_ref - 2 * sqrtm(Sigma_gen @ Sigma_ref))
  fid_total = mean_term + cov_term

Used as stage-gate signal #1: the *slope of mean_term over ep 50..100*
determines whether the MP-JiT pilot's gains are concentrated in mean-matching
(unhealthy — would indicate mode collapse) or in covariance-matching (healthy).

CLI is intentionally minimal so that scripts/analysis/run_gate_decomposition.sh
can call this on every saved EMA sample dump.
"""

import argparse
import os

import numpy as np
import torch
from scipy import linalg
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.datasets import ImagesPathDataset
from torch.utils.data import DataLoader
from PIL import Image


@torch.no_grad()
def extract_pool3_features(sample_dir, batch_size=50, num_workers=4, device='cuda'):
    files = sorted(
        os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    if not files:
        raise RuntimeError(f"No images found in {sample_dir}")

    extractor = FeatureExtractorInceptionV3(
        name='inception-v3-compat',
        features_list=['2048'],
    ).to(device).eval()

    ds = ImagesPathDataset(files)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)

    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = extractor(batch)
        feats.append(out[0].cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


def compute_stats(feats):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def fid_decomposition(mu_g, sig_g, mu_r, sig_r, eps=1e-6):
    diff = mu_g - mu_r
    mean_term = float(diff @ diff)

    covmean, _ = linalg.sqrtm(sig_g @ sig_r, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig_g.shape[0]) * eps
        covmean = linalg.sqrtm((sig_g + offset) @ (sig_r + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    cov_term = float(np.trace(sig_g) + np.trace(sig_r) - 2.0 * np.trace(covmean))
    return mean_term, cov_term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='folder of generated images')
    ap.add_argument('--fid_stats', required=True, help='reference-stats npz with mu, sigma')
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--append-csv', required=True,
                    help='CSV to append: epoch,mean_term,cov_term,fid_total')
    ap.add_argument('--batch_size', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    ref = np.load(args.fid_stats)
    mu_r, sig_r = ref['mu'], ref['sigma']

    feats = extract_pool3_features(
        args.samples, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    mu_g, sig_g = compute_stats(feats)
    mean_term, cov_term = fid_decomposition(mu_g, sig_g, mu_r, sig_r)
    fid_total = mean_term + cov_term

    print(f"epoch={args.epoch} mean_term={mean_term:.4f} "
          f"cov_term={cov_term:.4f} fid_total={fid_total:.4f}")

    with open(args.append_csv, 'a') as f:
        f.write(f"{args.epoch},{mean_term:.6f},{cov_term:.6f},{fid_total:.6f}\n")


if __name__ == '__main__':
    main()
