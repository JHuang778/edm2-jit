# R050 — Cell A' (vanilla JiT-G/16) 400ep @ 512². Heavy; depends on M3 gate.
# JiT-G/16 is not in the upstream vanilla model table — assumes a G/16 entry
# added to model_jit.py matching the paper's G spec (depth 40, hidden 1408,
# heads 16, patch 16) before launching. See EXPERIMENT_PLAN §R050 notes.
RUN_ID=R050_Ap_vanilla_G16_512
MODEL="JiT-G/16"
EPOCHS=400
SEED=1
BATCH_SIZE=32
IMG_SIZE=512
EXTRA_ARGS="--blr 5e-5 --ema_decay1 0.9999 --ema_decay2 0.9996 \
--P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1"
