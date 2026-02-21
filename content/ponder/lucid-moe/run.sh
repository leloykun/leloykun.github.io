#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash content/ponder/lucid-moe/run.sh
#   LUCID_ROUTER=on STEPS=50000 bash content/ponder/lucid-moe/run.sh
#   LUCID_ROUTER=off BATCH_SIZE=3 bash content/ponder/lucid-moe/run.sh
#   RUN_BOTH_LUCID_ROUTER=on PLOT_LOSS=on bash content/ponder/lucid-moe/run.sh
#
# Toggles:
#   LUCID_ROUTER=off|on
#   RUN_BOTH_LUCID_ROUTER=off|on
#   PLOT_LOSS=off|on
#   LIVE_PLOT=off|on
#   DETERMINISTIC=on|off
#   QE_NORM=off|on
#   SIGMOID_GATING=off|on
#   AUXFREE_BIAS=off|on
#
# RTX 2060 (6GB) friendly defaults are intentionally conservative.

CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-nanogpt}"
PY_SCRIPT="${PY_SCRIPT:-content/ponder/lucid-moe/mh_lmoe_lucid.py}"
OUT_DIR="${OUT_DIR:-content/ponder/lucid-moe/runs}"

LUCID_ROUTER="${LUCID_ROUTER:-off}"             # off | on
RUN_BOTH_LUCID_ROUTER="${RUN_BOTH_LUCID_ROUTER:-on}"  # off | on
PLOT_LOSS="${PLOT_LOSS:-on}"      # on | off
LIVE_PLOT="${LIVE_PLOT:-on}"      # on | off (only used when PLOT_LOSS=on)
DETERMINISTIC="${DETERMINISTIC:-on}"  # on | off
QE_NORM="${QE_NORM:-off}"         # off | on
SIGMOID_GATING="${SIGMOID_GATING:-off}" # off | on
AUXFREE_BIAS="${AUXFREE_BIAS:-on}" # off | on
DEVICE="${DEVICE:-cuda}"          # auto | cuda | cpu

# Longer run defaults.
STEPS="${STEPS:-128}"
EVAL_INTERVAL="${EVAL_INTERVAL:-32}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"

# 6GB VRAM-safe model/data defaults.
BATCH_SIZE="${BATCH_SIZE:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
N_LAYERS="${N_LAYERS:-2}"
MOE_HEADS="${MOE_HEADS:-1}"
D_MOE_LATENT="${D_MOE_LATENT:-$((D_MODEL / MOE_HEADS))}"
ATTN_HEADS="${ATTN_HEADS:-$MOE_HEADS}"
NUM_EXPERTS="${NUM_EXPERTS:-32}"
TOP_K="${TOP_K:-4}"
EXPERT_HIDDEN="${EXPERT_HIDDEN:-128}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-64}"

LR="${LR:-5e-3}"
LR_EMBEDDING="${LR_EMBEDDING:-1e-1}"
LR_ROUTER_EMBEDDING="${LR_ROUTER_EMBEDDING:-5e-2}"
LR_LINEAR="${LR_LINEAR:-5e-2}"
LR_LM_HEAD="${LR_LM_HEAD:-1e-2}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.9}"
EPS="${EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
AUXFREE_BIAS_LR="${AUXFREE_BIAS_LR:-1e-2}"
AUXFREE_BIAS_CLIP="${AUXFREE_BIAS_CLIP:-10.0}"
SEED="${SEED:-1337}"
DATA_SEED="${DATA_SEED:-2026}"

# Determinism aid for cuBLAS kernels. Keep configurable for compatibility.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

mkdir -p "${OUT_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local lucid_router_mode="$1"   # on|off
  local lucid_router_arg
  local lucid_router_suffix

  if [[ "${lucid_router_mode}" == "on" ]]; then
    lucid_router_arg="--enable_lucid_router"
    lucid_router_suffix="lucid_on"
  else
    lucid_router_arg="--no-enable_lucid_router"
    lucid_router_suffix="lucid_off"
  fi

  local run_name="mh_lmoe_${lucid_router_suffix}_flex_on_${timestamp}"
  local json_out="${OUT_DIR}/${run_name}.json"
  local plot_out="${OUT_DIR}/${run_name}.png"
  local deterministic_arg
  local qe_norm_arg
  local sigmoid_gating_arg
  local auxfree_bias_arg
  local live_plot_arg

  local d_moe_latent=$D_MOE_LATENT
  local d_head_attn=$((D_MODEL / ATTN_HEADS))
  if (( D_MODEL % ATTN_HEADS != 0 )); then
    echo "Invalid config: D_MODEL (${D_MODEL}) must be divisible by ATTN_HEADS (${ATTN_HEADS})." >&2
    exit 1
  fi
  if (( d_moe_latent <= 0 )); then
    echo "Invalid config: D_MOE_LATENT (${d_moe_latent}) must be > 0." >&2
    exit 1
  fi
  if (( (d_moe_latent & (d_moe_latent - 1)) != 0 )); then
    echo "Invalid config for MoE FlexAttention: d_moe_latent=${d_moe_latent} is not a power of 2." >&2
    echo "Set D_MOE_LATENT so per-head dim is 32/64/128..." >&2
    exit 1
  fi
  if (( d_moe_latent < 16 )); then
    echo "Invalid config for compiled MoE FlexAttention: d_moe_latent=${d_moe_latent} must be >= 16." >&2
    exit 1
  fi
  if (( (d_head_attn & (d_head_attn - 1)) != 0 )); then
    echo "Invalid config for self-attention FlexAttention: d_head_attn=${d_head_attn} is not a power of 2." >&2
    echo "Set D_MODEL/ATTN_HEADS so per-head dim is 32/64/128..." >&2
    exit 1
  fi
  if (( d_head_attn < 16 )); then
    echo "Invalid config for compiled self-attention FlexAttention: d_head_attn=${d_head_attn} must be >= 16." >&2
    exit 1
  fi

  if [[ "${DETERMINISTIC}" == "on" ]]; then
    deterministic_arg="--deterministic"
  elif [[ "${DETERMINISTIC}" == "off" ]]; then
    deterministic_arg="--no-deterministic"
  else
    echo "Invalid DETERMINISTIC=${DETERMINISTIC}. Use on|off." >&2
    exit 1
  fi
  if [[ "${QE_NORM}" == "on" ]]; then
    qe_norm_arg="--enable_qe_norm"
  elif [[ "${QE_NORM}" == "off" ]]; then
    qe_norm_arg="--no-enable_qe_norm"
  else
    echo "Invalid QE_NORM=${QE_NORM}. Use on|off." >&2
    exit 1
  fi
  if [[ "${SIGMOID_GATING}" == "on" ]]; then
    sigmoid_gating_arg="--enable_sigmoid_gating"
  elif [[ "${SIGMOID_GATING}" == "off" ]]; then
    sigmoid_gating_arg="--no-enable_sigmoid_gating"
  else
    echo "Invalid SIGMOID_GATING=${SIGMOID_GATING}. Use on|off." >&2
    exit 1
  fi
  if [[ "${AUXFREE_BIAS}" == "on" ]]; then
    auxfree_bias_arg="--enable_auxfree_bias"
  elif [[ "${AUXFREE_BIAS}" == "off" ]]; then
    auxfree_bias_arg="--no-enable_auxfree_bias"
  else
    echo "Invalid AUXFREE_BIAS=${AUXFREE_BIAS}. Use on|off." >&2
    exit 1
  fi
  if [[ "${LIVE_PLOT}" == "on" ]]; then
    live_plot_arg="--live_plot"
  elif [[ "${LIVE_PLOT}" == "off" ]]; then
    live_plot_arg="--no-live_plot"
  else
    echo "Invalid LIVE_PLOT=${LIVE_PLOT}. Use on|off." >&2
    exit 1
  fi

  echo "=== Starting ${run_name} ==="
  echo "device=${DEVICE} steps=${STEPS} batch=${BATCH_SIZE} block=${BLOCK_SIZE} d_model=${D_MODEL} heads(attn/moe)=${ATTN_HEADS}/${MOE_HEADS} d_moe_latent=${D_MOE_LATENT} sparsity=${TOP_K}/${NUM_EXPERTS}"

  local cmd=(
    "${CONDA_BIN}" run -n "${CONDA_ENV}" python "${PY_SCRIPT}"
    --device "${DEVICE}"
    "${deterministic_arg}"
    "${qe_norm_arg}"
    "${sigmoid_gating_arg}"
    "${auxfree_bias_arg}"
    "${lucid_router_arg}"
    --steps "${STEPS}"
    --eval_interval "${EVAL_INTERVAL}"
    --eval_batches "${EVAL_BATCHES}"
    --batch_size "${BATCH_SIZE}"
    --block_size "${BLOCK_SIZE}"
    --d_model "${D_MODEL}"
    --n_layers "${N_LAYERS}"
    --attn_heads "${ATTN_HEADS}"
    --moe_heads "${MOE_HEADS}"
    --d_moe_latent "${D_MOE_LATENT}"
    --num_experts "${NUM_EXPERTS}"
    --top_k "${TOP_K}"
    --expert_hidden "${EXPERT_HIDDEN}"
    --kv_block_size "${KV_BLOCK_SIZE}"
    --lr "${LR}"
    --lr_embedding "${LR_EMBEDDING}"
    --lr_router_embedding "${LR_ROUTER_EMBEDDING}"
    --lr_linear "${LR_LINEAR}"
    --lr_lm_head "${LR_LM_HEAD}"
    --min_lr_ratio "${MIN_LR_RATIO}"
    --beta1 "${BETA1}"
    --beta2 "${BETA2}"
    --eps "${EPS}"
    --weight_decay "${WEIGHT_DECAY}"
    --auxfree_bias_lr "${AUXFREE_BIAS_LR}"
    --auxfree_bias_clip "${AUXFREE_BIAS_CLIP}"
    --seed "${SEED}"
    --data_seed "${DATA_SEED}"
    --json_out "${json_out}"
  )

  if [[ "${PLOT_LOSS}" == "on" ]]; then
    cmd+=(--plot_losses "${live_plot_arg}" --plot_out "${plot_out}")
  elif [[ "${PLOT_LOSS}" != "off" ]]; then
    echo "Invalid PLOT_LOSS=${PLOT_LOSS}. Use off|on." >&2
    exit 1
  fi

  "${cmd[@]}"
}

run_both() {
  local run_name="mh_lmoe_lucid_both_flex_on_${timestamp}"
  local json_out="${OUT_DIR}/${run_name}.json"
  local plot_out="${OUT_DIR}/${run_name}.png"
  local deterministic_arg
  local qe_norm_arg
  local sigmoid_gating_arg
  local auxfree_bias_arg
  local live_plot_arg

  local d_moe_latent=$D_MOE_LATENT
  local d_head_attn=$((D_MODEL / ATTN_HEADS))
  if (( D_MODEL % ATTN_HEADS != 0 )); then
    echo "Invalid config: D_MODEL (${D_MODEL}) must be divisible by ATTN_HEADS (${ATTN_HEADS})." >&2
    exit 1
  fi
  if (( d_moe_latent <= 0 )); then
    echo "Invalid config: D_MOE_LATENT (${d_moe_latent}) must be > 0." >&2
    exit 1
  fi
  if (( (d_moe_latent & (d_moe_latent - 1)) != 0 )); then
    echo "Invalid config for MoE FlexAttention: d_moe_latent=${d_moe_latent} is not a power of 2." >&2
    echo "Set D_MOE_LATENT so per-head dim is 32/64/128..." >&2
    exit 1
  fi
  if (( d_moe_latent < 16 )); then
    echo "Invalid config for compiled MoE FlexAttention: d_moe_latent=${d_moe_latent} must be >= 16." >&2
    exit 1
  fi
  if (( (d_head_attn & (d_head_attn - 1)) != 0 )); then
    echo "Invalid config for self-attention FlexAttention: d_head_attn=${d_head_attn} is not a power of 2." >&2
    echo "Set D_MODEL/ATTN_HEADS so per-head dim is 32/64/128..." >&2
    exit 1
  fi
  if (( d_head_attn < 16 )); then
    echo "Invalid config for compiled self-attention FlexAttention: d_head_attn=${d_head_attn} must be >= 16." >&2
    exit 1
  fi

  if [[ "${DETERMINISTIC}" == "on" ]]; then
    deterministic_arg="--deterministic"
  elif [[ "${DETERMINISTIC}" == "off" ]]; then
    deterministic_arg="--no-deterministic"
  else
    echo "Invalid DETERMINISTIC=${DETERMINISTIC}. Use on|off." >&2
    exit 1
  fi
  if [[ "${QE_NORM}" == "on" ]]; then
    qe_norm_arg="--enable_qe_norm"
  elif [[ "${QE_NORM}" == "off" ]]; then
    qe_norm_arg="--no-enable_qe_norm"
  else
    echo "Invalid QE_NORM=${QE_NORM}. Use on|off." >&2
    exit 1
  fi
  if [[ "${SIGMOID_GATING}" == "on" ]]; then
    sigmoid_gating_arg="--enable_sigmoid_gating"
  elif [[ "${SIGMOID_GATING}" == "off" ]]; then
    sigmoid_gating_arg="--no-enable_sigmoid_gating"
  else
    echo "Invalid SIGMOID_GATING=${SIGMOID_GATING}. Use on|off." >&2
    exit 1
  fi
  if [[ "${AUXFREE_BIAS}" == "on" ]]; then
    auxfree_bias_arg="--enable_auxfree_bias"
  elif [[ "${AUXFREE_BIAS}" == "off" ]]; then
    auxfree_bias_arg="--no-enable_auxfree_bias"
  else
    echo "Invalid AUXFREE_BIAS=${AUXFREE_BIAS}. Use on|off." >&2
    exit 1
  fi
  if [[ "${LIVE_PLOT}" == "on" ]]; then
    live_plot_arg="--live_plot"
  elif [[ "${LIVE_PLOT}" == "off" ]]; then
    live_plot_arg="--no-live_plot"
  else
    echo "Invalid LIVE_PLOT=${LIVE_PLOT}. Use on|off." >&2
    exit 1
  fi

  if [[ "${LUCID_ROUTER}" != "off" ]]; then
    echo "RUN_BOTH_LUCID_ROUTER=on ignores LUCID_ROUTER=${LUCID_ROUTER}." >&2
  fi

  echo "=== Starting ${run_name} ==="
  echo "device=${DEVICE} steps=${STEPS} batch=${BATCH_SIZE} block=${BLOCK_SIZE} d_model=${D_MODEL} heads(attn/moe)=${ATTN_HEADS}/${MOE_HEADS} d_moe_latent=${D_MOE_LATENT} sparsity=${TOP_K}/${NUM_EXPERTS}"

  local cmd=(
    "${CONDA_BIN}" run -n "${CONDA_ENV}" python "${PY_SCRIPT}"
    --device "${DEVICE}"
    "${deterministic_arg}"
    "${qe_norm_arg}"
    "${sigmoid_gating_arg}"
    "${auxfree_bias_arg}"
    --run_both_lucid_router
    --steps "${STEPS}"
    --eval_interval "${EVAL_INTERVAL}"
    --eval_batches "${EVAL_BATCHES}"
    --batch_size "${BATCH_SIZE}"
    --block_size "${BLOCK_SIZE}"
    --d_model "${D_MODEL}"
    --n_layers "${N_LAYERS}"
    --attn_heads "${ATTN_HEADS}"
    --moe_heads "${MOE_HEADS}"
    --d_moe_latent "${D_MOE_LATENT}"
    --num_experts "${NUM_EXPERTS}"
    --top_k "${TOP_K}"
    --expert_hidden "${EXPERT_HIDDEN}"
    --kv_block_size "${KV_BLOCK_SIZE}"
    --lr "${LR}"
    --lr_embedding "${LR_EMBEDDING}"
    --lr_router_embedding "${LR_ROUTER_EMBEDDING}"
    --lr_linear "${LR_LINEAR}"
    --lr_lm_head "${LR_LM_HEAD}"
    --min_lr_ratio "${MIN_LR_RATIO}"
    --beta1 "${BETA1}"
    --beta2 "${BETA2}"
    --eps "${EPS}"
    --weight_decay "${WEIGHT_DECAY}"
    --auxfree_bias_lr "${AUXFREE_BIAS_LR}"
    --auxfree_bias_clip "${AUXFREE_BIAS_CLIP}"
    --seed "${SEED}"
    --data_seed "${DATA_SEED}"
    --json_out "${json_out}"
  )

  if [[ "${PLOT_LOSS}" == "on" ]]; then
    cmd+=(--plot_losses "${live_plot_arg}" --plot_out "${plot_out}")
  elif [[ "${PLOT_LOSS}" != "off" ]]; then
    echo "Invalid PLOT_LOSS=${PLOT_LOSS}. Use off|on." >&2
    exit 1
  fi

  "${cmd[@]}"
}

case "${RUN_BOTH_LUCID_ROUTER}" in
  on)
    run_both
    ;;
  off)
    case "${LUCID_ROUTER}" in
      on)
        run_one "on"
        ;;
      off)
        run_one "off"
        ;;
      *)
        echo "Invalid LUCID_ROUTER=${LUCID_ROUTER}. Use off|on." >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Invalid RUN_BOTH_LUCID_ROUTER=${RUN_BOTH_LUCID_ROUTER}. Use off|on." >&2
    exit 1
    ;;
esac
