# R020 — Phase 0b stage-gate pilot: 100 epoch MP-JiT-D at B/16 256². The
# three-signal stage gate (mean-term slope, weight-norm slope, FID-5k gap) is
# read off the logged traces at epoch 100 before authorizing the full 400ep.
RUN_ID=R020_D_gate_pilot
MODEL="JiT-B/16"
EPOCHS=200
SEED=1
BATCH_SIZE=256
IMG_SIZE=256
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1 \
--use_mp --qk_lock_epochs 5 --qk_lock_slope 0.1 \
--use_sigma_weight --pilot_steps 5000 \
--online_eval --eval_freq 10 --num_images 5000 --log_ema_online_gap --keep_gate_samples"
