# R051 — Cell D' (full MP-JiT-G/16) 400ep @ 512². Heavy; depends on M3 gate.
RUN_ID=R051_Dp_full_G16_512
MODEL="JiT-G/16"
EPOCHS=400
SEED=1
BATCH_SIZE=32
IMG_SIZE=512
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1 \
--use_mp --qk_lock_epochs 5 --qk_lock_slope 0.1 \
--use_sigma_weight --pilot_steps 5000"
