# R034 — Cell C (σ-bucket weighting only, no MPLinear) seed 1.
# Pilot still runs at the start of training to calibrate the 16-bucket w table.
RUN_ID=R034_C_sigma_only_seed1
MODEL="JiT-B/16"
EPOCHS=400
SEED=1
BATCH_SIZE=256
IMG_SIZE=256
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1 \
--use_sigma_weight --pilot_steps 5000"
