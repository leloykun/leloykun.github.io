#!/usr/bin/env bash
set -euo pipefail

OPTIMIZER="${OPTIMIZER:-adamw}" # set OPTIMIZER=muon for Muon experiments
RUN="${RUN:-all}"
STEPS="${STEPS:-1024}"
LR="${LR:-1e-2}"
NS_STEPS="${NS_STEPS:-10}"
NS_EPS="${NS_EPS:-1e-20}"
NS_SCALE="${NS_SCALE:-1.0}"
SCALE_EMA_DECAY="${SCALE_EMA_DECAY:-0.99}"

echo "${OPTIMIZER}" "${RUN}" "${STEPS}" "${LR}"

~/anaconda3/bin/conda run -n nanogpt python content/ponder/eco/train_eco.py \
  --run "${RUN}" \
  --optimizer "${OPTIMIZER}" \
  --eval_interval 32 \
  --eval_batches 8 \
  --batch_size 64 \
  --block_size 128 \
  --d_model 256 \
  --n_layers 2 \
  --mlp_hidden 1024 \
  --json_out "content/ponder/eco/train_data_${OPTIMIZER}_${STEPS}steps.json" \
  --plot_out "content/ponder/eco/loss_plot_${OPTIMIZER}_${STEPS}steps.png" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --ns_steps "${NS_STEPS}" \
  --ns_eps "${NS_EPS}" \
  --ns_scale "${NS_SCALE}" \
  --scale_ema_decay "${SCALE_EMA_DECAY}"
