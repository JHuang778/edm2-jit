# R031 — Cell A (vanilla) seed 2.
RUN_ID=R031_A_vanilla_seed2
MODEL="JiT-B/16"
EPOCHS=400
SEED=2
BATCH_SIZE=64
IMG_SIZE=256
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1"
