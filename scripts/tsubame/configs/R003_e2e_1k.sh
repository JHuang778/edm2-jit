# R003 — M0 sanity: end-to-end 1k-step JiT-B/16 training, 4-GPU DDP, bf16.
# A full-node short job (~2h on 4× H100) that exercises the complete pipeline.
RUN_ID=R003_sanity_1k
MODEL="JiT-B/16"
EPOCHS=1          # 1 epoch is ~20k steps at BS=64 on 4 GPUs; we cap walltime.
SEED=0
BATCH_SIZE=64
IMG_SIZE=256
EXTRA_ARGS="--use_mp --use_sigma_weight --pilot_steps 200 \
--blr 5e-5 --P_mean -0.8 --P_std 0.8 --label_drop_prob 0.1"
