"""M0 sanity tests for MPLinear (MP-JiT leg 1)."""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model_jit import MPLinear


def test_forward_equivalence_under_gain_scaling():
    torch.manual_seed(0)
    lin = MPLinear(32, 64)
    x = torch.randn(8, 32)
    y0 = lin(x)
    with torch.no_grad():
        lin.weight.mul_(7.3)          # rescale raw W
    y1 = lin(x)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-4), \
        'MPLinear forward must be invariant to weight row scaling.'


def test_gain_scales_output_linearly():
    torch.manual_seed(0)
    lin = MPLinear(16, 16)
    x = torch.randn(4, 16)
    y0 = lin(x)
    with torch.no_grad():
        lin.gain.fill_(3.0)
    y1 = lin(x)
    assert torch.allclose(y1, 3.0 * y0, atol=1e-5, rtol=1e-4)


def test_backward_grad_nonzero():
    torch.manual_seed(0)
    lin = MPLinear(8, 8)
    x = torch.randn(4, 8, requires_grad=True)
    y = lin(x).sum()
    y.backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0
    assert lin.gain.grad is not None and lin.gain.grad.abs().item() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_column_norm_unit_after_forward_rule():
    """Verify that W_norm = W / ||W row|| has unit row norm (so effective column
    norm of W_norm^T is 1)."""
    torch.manual_seed(0)
    lin = MPLinear(64, 128)
    w = lin.weight
    wn = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    norms = wn.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_matches_manual_computation():
    torch.manual_seed(0)
    lin = MPLinear(5, 7)
    x = torch.randn(3, 5)
    W = lin.weight
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    expected = F.linear(x, Wn) * lin.gain
    got = lin(x)
    assert torch.allclose(got, expected, atol=1e-6)


if __name__ == '__main__':
    test_forward_equivalence_under_gain_scaling()
    test_gain_scales_output_linearly()
    test_backward_grad_nonzero()
    test_column_norm_unit_after_forward_rule()
    test_matches_manual_computation()
    print('All MPLinear tests passed.')
