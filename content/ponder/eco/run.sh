#!/usr/bin/env bash
set -euo pipefail

OPTIMIZER="${OPTIMIZER:-adamw}" # set OPTIMIZER=muon for Muon experiments
RUN="${RUN:-all}"
STEPS="${STEPS:-512}"

echo "${OPTIMIZER}" "${RUN}" "${STEPS}"

~/anaconda3/bin/conda run -n nanogpt python content/ponder/eco/train_residual_mlp_shakespeare_eco.py \
  --run "${RUN}" \
  --optimizer "${OPTIMIZER}" \
  --eval_interval 16 \
  --eval_batches 8 \
  --batch_size 64 \
  --block_size 128 \
  --d_model 256 \
  --n_layers 4 \
  --mlp_hidden 1024 \
  --json_out "content/ponder/eco/train_data_${OPTIMIZER}.json" \
  --plot_out "content/ponder/eco/loss_plot_${OPTIMIZER}.png" \
  --steps "${STEPS}" \
  --lr 1e-2
