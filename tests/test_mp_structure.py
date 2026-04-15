"""Structural tests for MP-JiT: verifies the live model matches the spec
(MP final head, split SwiGLU w1/w2/w3, qk-lock ep5/ep6 boundary, pilot buffer
state_dict round-trip)."""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model_jit import MPLinear, JiT_models, FinalLayer, SwiGLUFFN


class _Args:
    model = 'JiT-B/16'
    img_size = 256
    class_num = 1000
    attn_dropout = 0.0
    proj_dropout = 0.0
    label_drop_prob = 0.1
    P_mean = -0.4
    P_std = 1.0
    t_eps = 5e-2
    noise_scale = 1.0
    ema_decay1 = 0.9999
    ema_decay2 = 0.999
    sampling_method = 'heun'
    num_sampling_steps = 50
    cfg = 1.0
    interval_min = 0.0
    interval_max = 1.0
    use_mp = True
    use_sigma_weight = True
    pilot_steps = 10
    qk_lock_epochs = 5
    qk_lock_slope = 0.1


def test_final_head_is_mp_when_use_mp():
    net = JiT_models['JiT-B/16'](input_size=256, num_classes=1000, use_mp=True)
    assert isinstance(net.final_layer.linear, MPLinear), \
        'FinalLayer.linear must be MPLinear when use_mp=True.'
    # Gain is zero-initialized per spec (preserves zero-output-at-init).
    assert float(net.final_layer.linear.gain.item()) == 0.0, \
        'MP final head gain must init to 0.'


def test_final_head_stays_vanilla_when_not_use_mp():
    net = JiT_models['JiT-B/16'](input_size=256, num_classes=1000, use_mp=False)
    assert isinstance(net.final_layer.linear, nn.Linear) \
        and not isinstance(net.final_layer.linear, MPLinear)


def test_swiglu_split_w1_w2_under_mp():
    net = JiT_models['JiT-B/16'](input_size=256, num_classes=1000, use_mp=True)
    for blk in net.blocks:
        mlp = blk.mlp
        assert hasattr(mlp, 'w1') and isinstance(mlp.w1, MPLinear)
        assert hasattr(mlp, 'w2') and isinstance(mlp.w2, MPLinear)
        assert hasattr(mlp, 'w3') and isinstance(mlp.w3, MPLinear)
        # Independent gains (not the same Parameter object).
        assert mlp.w1.gain is not mlp.w2.gain


def test_qk_lock_inclusive_schedule():
    net = JiT_models['JiT-B/16'](input_size=256, num_classes=1000, use_mp=True,
                                 qk_lock_epochs=5, qk_lock_slope=0.1).cuda()
    # Perturb one q.gain to force a penalty
    with torch.no_grad():
        net.blocks[0].attn.q.gain.mul_(1.5)
    # Epochs 0..5 inclusive should produce a penalty; ep 6 must be zero.
    pen_ep0 = net.qk_lock_penalty(0).item()
    pen_ep4 = net.qk_lock_penalty(4).item()
    pen_ep5 = net.qk_lock_penalty(5).item()
    pen_ep6 = net.qk_lock_penalty(6).item()
    assert pen_ep0 > 0, 'Ep 0 should activate (band=0 → any deviation triggers).'
    assert pen_ep5 >= 0, 'Ep 5 should still be active.'
    # Ep 0 has zero band so largest penalty; ep 5 has band=0.1 so smaller.
    assert pen_ep0 > pen_ep5, f'Barrier should shrink as band widens: ep0={pen_ep0}, ep5={pen_ep5}.'
    assert pen_ep6 == 0.0, f'Ep 6 must release the barrier (got {pen_ep6}).'


def test_pilot_buffers_round_trip_state_dict():
    """Half-populate the pilot accumulators, save state_dict, load into a fresh
    Denoiser, and verify every pilot buffer matches bit-identically."""
    from denoiser import Denoiser
    torch.manual_seed(0)
    args = _Args()
    d1 = Denoiser(args).cuda()
    # Simulate partial pilot: bump a few buckets.
    with torch.no_grad():
        d1.r2_sum[3] += 1.234
        d1.r2_sum[10] += 5.678
        d1.r2_count[3] += 42.0
        d1.r2_count[10] += 17.0
        d1.step_counter += 1234
    sd = d1.state_dict()

    d2 = Denoiser(args).cuda()
    d2.load_state_dict(sd)
    for name in ('r2_sum', 'r2_count', 'step_counter', 'pilot_done',
                 'bucket_w', 'bucket_edges'):
        a = getattr(d1, name)
        b = getattr(d2, name)
        assert torch.equal(a, b), f'{name} not bit-identical after state_dict round-trip.'


if __name__ == '__main__':
    test_final_head_is_mp_when_use_mp()
    test_final_head_stays_vanilla_when_not_use_mp()
    test_swiglu_split_w1_w2_under_mp()
    test_qk_lock_inclusive_schedule()
    test_pilot_buffers_round_trip_state_dict()
    print('All MP structure tests passed.')
