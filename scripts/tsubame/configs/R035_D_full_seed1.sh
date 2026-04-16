# R035 — Cell D (full MP-JiT = MPLinear + σ-bucket weighting) seed 1. Headline.
RUN_ID=R035_D_full_seed1
MODEL="JiT-B/16"
EPOCHS=400
SEED=1
BATCH_SIZE=256
IMG_SIZE=256
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1 \
--use_mp --qk_lock_epochs 5 --qk_lock_slope 0.1 \
--use_sigma_weight --pilot_steps 5000"
