import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)
        # MP-JiT: symmetric qk-lock barrier (active epochs 0..qk_lock_epochs inclusive).
        # Kept outside autocast so the 1e-3 * ReLU(gap)^2 term is accumulated in fp32.
        if getattr(model_without_ddp.net, 'use_mp', False):
            qk_pen = model_without_ddp.net.qk_lock_penalty(epoch)
            loss = loss + qk_pen.to(loss.dtype)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                _log_mp_jit_diagnostics(model_without_ddp, log_writer, epoch_1000x)


@torch.no_grad()
def _log_mp_jit_diagnostics(model_without_ddp, log_writer, step):
    """Logs MP-JiT mechanism traces: per-block column-norm means, qk gain product,
    per-bucket r² and bucket weights w[b]. Cheap; writes only scalars/histograms."""
    net = model_without_ddp.net
    if step == 0 and getattr(model_without_ddp, 'use_sigma_weight', False):
        # One-shot log of the analytical bucket edges so reviewers can verify
        # the equal-probability quantile partition under the log-σ prior.
        for i, edge in enumerate(model_without_ddp.bucket_edges.tolist()):
            log_writer.add_scalar(f'mp/bucket_edge/{i:02d}', edge, step)
    if getattr(net, 'use_mp', False):
        # Also log the MP final prediction head.
        head = net.final_layer.linear
        head_norms = head.weight.norm(dim=1)
        log_writer.add_scalar('mp/final/col_norm_mean', head_norms.mean().item(), step)
        log_writer.add_scalar('mp/final/col_norm_max',  head_norms.max().item(),  step)
        log_writer.add_scalar('mp/final/gain',          head.gain.item(),          step)
        for i, blk in enumerate(net.blocks):
            for name, lin in (('q', blk.attn.q), ('k', blk.attn.k),
                              ('v', blk.attn.v), ('proj', blk.attn.proj),
                              ('w1', blk.mlp.w1), ('w2', blk.mlp.w2), ('w3', blk.mlp.w3)):
                col_norms = lin.weight.norm(dim=1)
                # Raw row-norms: normalized to 1 inside forward but the raw
                # value drifts during training; the trace is the key signal
                # for whether MP's norm-preserving property is active.
                log_writer.add_scalar(f'mp/block{i}/{name}_col_norm_mean', col_norms.mean().item(), step)
                log_writer.add_scalar(f'mp/block{i}/{name}_col_norm_max',  col_norms.max().item(),  step)
                log_writer.add_scalar(f'mp/block{i}/{name}_col_norm_min',  col_norms.min().item(),  step)
                log_writer.add_scalar(f'mp/block{i}/{name}_gain', lin.gain.item(), step)
            log_writer.add_scalar(
                f'mp/block{i}/qk_gain_product',
                (blk.attn.q.gain * blk.attn.k.gain).item(), step,
            )
    if getattr(model_without_ddp, 'use_sigma_weight', False):
        r2 = (model_without_ddp.r2_sum / model_without_ddp.r2_count.clamp_min(1.0)).float()
        for b in range(r2.numel()):
            log_writer.add_scalar(f'mp/r2/bucket{b:02d}', r2[b].item(), step)
            log_writer.add_scalar(f'mp/w/bucket{b:02d}', model_without_ddp.bucket_w[b].item(), step)
            # r2_count exposes bucket sparsity — required to interpret r² validity
            # (an unseen bucket has r²=0 and would produce a garbage w[b]).
            log_writer.add_scalar(f'mp/r2_count/bucket{b:02d}',
                                  model_without_ddp.r2_count[b].item(), step)
        log_writer.add_scalar('mp/pilot_done', float(model_without_ddp.pilot_done.item()), step)
        log_writer.add_scalar('mp/pilot_step', model_without_ddp.step_counter.item(), step)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None, use_ema=True):
    """FID-N evaluation. When `use_ema=True` we swap in ema_params1 (default,
    matches vanilla JiT). When `use_ema=False` we evaluate the live/online
    parameters. Stage-gate signal #3 (EMA-vs-online FID gap) is computed by
    calling this twice per eval epoch and taking the difference."""

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images. Tag with epoch so
    # stage-gate signal #1 post-hoc decomposition can read per-epoch dumps.
    tag = 'ema' if use_ema else 'online'
    keep_samples = (use_ema and getattr(args, 'keep_gate_samples', False)
                    and 50 <= epoch <= 100)
    save_folder = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "ep{:03d}".format(epoch),
        "{}-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            tag, model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    if use_ema:
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)
    else:
        print("Evaluating online (non-EMA) params")

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    fid = None
    inception_score = None
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_{}_cfg{}_res{}".format(tag, model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not keep_samples:
            shutil.rmtree(save_folder)
        else:
            print(f"Stage-gate: preserved EMA sample dump at {save_folder}")

    torch.distributed.barrier()
    return fid, inception_score
