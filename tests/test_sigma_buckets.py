"""M0 sanity tests for σ-bucket indexer (MP-JiT leg 2).

Under the training sampler z ~ N(P_mean, P_std), the 16 equal-probability
quantile edges should produce a (near-)uniform bucket distribution. A χ²
goodness-of-fit test confirms this.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from denoiser import _norm_ppf_equal_prob_edges, N_BUCKETS


def _chi2_p_value(counts, expected):
    """One-sided p-value for upper tail of χ²(df) using a regularized gamma
    approximation via torch.special. Returns scalar float."""
    df = counts.numel() - 1
    chi2 = ((counts.float() - expected) ** 2 / expected).sum().item()
    # P(X > chi2) = 1 - regularized_gamma(df/2, chi2/2)
    # torch.special.gammainc is regularized lower incomplete gamma.
    p = 1.0 - torch.special.gammainc(
        torch.tensor(df / 2.0), torch.tensor(chi2 / 2.0)
    ).item()
    return p, chi2


def test_equal_probability_bucket_uniformity():
    torch.manual_seed(0)
    p_mean, p_std = -0.8, 0.8
    edges = _norm_ppf_equal_prob_edges(N_BUCKETS, p_mean, p_std)
    assert edges.numel() == N_BUCKETS - 1
    # Edges must be strictly increasing.
    assert (edges[1:] > edges[:-1]).all()

    n = 200_000
    z = torch.randn(n) * p_std + p_mean
    b = torch.bucketize(z, edges)
    counts = torch.bincount(b, minlength=N_BUCKETS)
    assert counts.numel() == N_BUCKETS

    expected = n / N_BUCKETS
    p, chi2 = _chi2_p_value(counts, expected)
    # With 200k samples and df=15, a sane implementation should easily pass p > 0.01.
    assert p > 0.01, (
        f'χ² uniformity rejected: p={p:.4f}, χ²={chi2:.2f}, '
        f'counts={counts.tolist()}'
    )


def test_edges_cover_tail_mass_evenly():
    p_mean, p_std = 0.0, 1.0
    edges = _norm_ppf_equal_prob_edges(4, p_mean, p_std)
    # For N(0,1), quartile edges should be ≈ (-0.6745, 0, 0.6745).
    expected = torch.tensor([-0.6745, 0.0, 0.6745])
    assert torch.allclose(edges.float(), expected, atol=1e-3)


def test_bucket_indices_in_range():
    torch.manual_seed(1)
    edges = _norm_ppf_equal_prob_edges(N_BUCKETS, -0.8, 0.8)
    z = torch.randn(10_000) * 0.8 - 0.8
    b = torch.bucketize(z, edges)
    assert int(b.min()) >= 0
    assert int(b.max()) < N_BUCKETS


if __name__ == '__main__':
    test_equal_probability_bucket_uniformity()
    test_edges_cover_tail_mass_evenly()
    test_bucket_indices_in_range()
    print('All σ-bucket tests passed.')
