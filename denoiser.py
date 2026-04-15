import math

import torch
import torch.nn as nn
import torch.distributed as dist
from model_jit import JiT_models


N_BUCKETS = 16


def _norm_ppf_equal_prob_edges(n_buckets, p_mean, p_std, device=None, dtype=torch.float32):
    """Interior edges of `n_buckets` equal-probability quantiles of N(p_mean, p_std)
    in pre-sigmoid (log-σ analog) space. Returns shape (n_buckets - 1,)."""
    q = torch.arange(1, n_buckets, device=device, dtype=dtype) / n_buckets
    # N(μ, σ) ppf(q) = μ + σ √2 · erfinv(2q - 1)
    return p_mean + p_std * math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        use_mp = getattr(args, 'use_mp', False)
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            use_mp=use_mp,
            qk_lock_epochs=getattr(args, 'qk_lock_epochs', 5),
            qk_lock_slope=getattr(args, 'qk_lock_slope', 0.1),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # σ-bucket weighting (MP-JiT leg 2).
        self.use_sigma_weight = getattr(args, 'use_sigma_weight', False)
        self.pilot_steps = getattr(args, 'pilot_steps', 5000)
        edges = _norm_ppf_equal_prob_edges(N_BUCKETS, self.P_mean, self.P_std)
        self.register_buffer('bucket_edges', edges)         # (N_BUCKETS-1,) interior
        self.register_buffer('bucket_w', torch.ones(N_BUCKETS))
        self.register_buffer('r2_sum', torch.zeros(N_BUCKETS, dtype=torch.float64))
        self.register_buffer('r2_count', torch.zeros(N_BUCKETS, dtype=torch.float64))
        self.register_buffer('step_counter', torch.zeros((), dtype=torch.long))
        self.register_buffer('pilot_done', torch.zeros((), dtype=torch.bool))

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        """Returns (t, z_latent) where z_latent is the pre-sigmoid Gaussian draw
        (aka log-σ analog) used for σ-bucket indexing."""
        z_latent = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z_latent), z_latent

    def _bucketize(self, z_latent):
        """Assigns each pre-sigmoid latent to one of N_BUCKETS equal-probability
        buckets. Returns long tensor in [0, N_BUCKETS)."""
        return torch.bucketize(z_latent, self.bucket_edges)

    @torch.no_grad()
    def _finalize_pilot(self):
        """Compute static bucket weights from pilot r² statistics. Called once
        when step_counter crosses pilot_steps. DDP-aware all-reduce.
        Safeguards: unseen buckets (count==0) default to w=1; median/mean are
        taken over seen buckets only. Post-cast renormalization makes
        mean(bucket_w)==1 bit-exact in float32."""
        r2_sum = self.r2_sum.clone()
        r2_count = self.r2_count.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(r2_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(r2_count, op=dist.ReduceOp.SUM)
        # Write global values back so every rank stores the same statistics
        # used for calibration (post-pilot logging, checkpoint, future resumes).
        self.r2_sum.copy_(r2_sum)
        self.r2_count.copy_(r2_count)

        seen = r2_count > 0
        n_seen = int(seen.sum().item())
        w = torch.ones_like(self.bucket_w)
        if n_seen >= 2:
            r2_seen = (r2_sum[seen] / r2_count[seen]).clamp_min(1e-12)
            med = r2_seen.median()
            target_seen = (med / r2_seen).clamp(0.1, 10.0)
            # Normalize over seen buckets so mean==1 over the active support.
            target_seen = target_seen / target_seen.mean().clamp_min(1e-12)
            w_seen = target_seen.to(self.bucket_w.dtype)
            # Renormalize in the storage dtype to get bit-exact mean==1.
            w_seen = w_seen / w_seen.mean().clamp_min(1e-12)
            w[seen] = w_seen
        self.bucket_w.copy_(w)
        if not bool(self.pilot_done.item()):
            # Log a sentinel so downstream diagnostics know if unseen buckets
            # existed at calibration time.
            self._pilot_n_seen = n_seen
        self.pilot_done.fill_(True)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t_flat, z_latent = self.sample_t(x.size(0), device=x.device)
        t = t_flat.view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # Per-sample MSE residual. Computed in fp32 for pilot-calibration
        # numerical fidelity (bf16 autocast here would lose ~3 decimal digits).
        # NOTE: operationalized in v-space to match JiT's flow-matching training
        # loss. Spec text says "‖x̂ − x‖²" (EDM2 x-pred terminology); in JiT
        # v-space this equals ‖x̂ − x‖² / (1 − t)². Calibrating on v-space
        # residuals preserves loss-function consistency across A/B/C/D cells —
        # the ablation isolates the w[b] weighting, not the loss space.
        with torch.amp.autocast('cuda', enabled=False):
            diff = (v.float() - v_pred.float())
            m = (diff * diff).mean(dim=(1, 2, 3))

        if self.use_sigma_weight:
            b = self._bucketize(z_latent)
            in_pilot = self.training and (not bool(self.pilot_done.item()))
            if in_pilot:
                # accumulate per-bucket r² during pilot; weights stay at 1.
                with torch.no_grad():
                    self.r2_sum.index_add_(0, b, m.detach().to(self.r2_sum.dtype))
                    self.r2_count.index_add_(
                        0, b, torch.ones_like(b, dtype=self.r2_count.dtype)
                    )
                    self.step_counter += 1
                    if self.step_counter.item() >= self.pilot_steps:
                        self._finalize_pilot()
                loss = m.mean()
            else:
                w = self.bucket_w[b]
                loss = (w * m).mean()
        else:
            loss = m.mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
