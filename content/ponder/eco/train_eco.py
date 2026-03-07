#!/usr/bin/env python3
"""
Minimal ECO training test on Tiny Shakespeare with a residual MLP language model.

This script compares:
1) `reference`: FP32 master weights with `adamw`/`muon`/`shampoo`/`psgd`, but forward/backward
   always use FP8-quantized weights via STE for quantized linear layers.
2) `eco_fp8_nomaster`: FP8 materialized weights only (no master weights),
   optimizer family + ECO compensation + stochastic rounding on weight re-quantization.
3) `fp8_nomaster_noeco`: FP8 materialized weights only (no master weights),
   optimizer family without ECO compensation.

Both runs use FP8-quantized weights in forward and gradient computation.

Example (from repo root, in nanogpt env):
  conda run -n nanogpt python content/ponder/eco/train_eco.py \
      --steps 200 --eval_interval 50 --batch_size 64 --block_size 128 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ScaleMode = Literal["none", "tensor", "row"]
DEFAULT_SCALE_EMA_DECAY = 0.99

# E4M3
FP8_STORAGE_DTYPE = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2.0 ** -6
FP8_E4M3_MIN_SUBNORMAL = 2.0 ** -9
RMS_NORM_EPS = 1e-5

# Newton-Schulz coefficients for matrix inverse roots.
NTH_ROOT_COEFS: List[Optional[List[Tuple[float, float, float]]]] = [
    None,  # r = 0
    None,  # r = 1
    [  # r = 2
        (7.424865680309214, -18.39581635618996, 12.896720413604342),
        (3.4877256051546017, -2.3300436563986993, 0.4404692168431095),
        (2.7766085124882527, -2.070643152532662, 0.46302261050004967),
        (1.9913142104341506, -1.373936700681269, 0.3875934979568538),
        (1.8754637749479246, -1.2505152090010534, 0.37505152463617264),
        (1.875, -1.25, 0.375),
    ],
    None,  # r = 3
    [  # r = 4
        (3.85003181724939, -10.853860241993278, 8.618933773002455),
        (1.8099210622771318, -0.5877777285425438, 0.06478521007149429),
        (1.5039396714850155, -0.594515590829229, 0.12116149581658857),
        (1.4086233134294281, -0.5637769238099195, 0.1551787660660711),
        (1.4062500496446348, -0.562500027067629, 0.15624997742300542),
        (1.40625, -0.5625, 0.15625),
    ],
]

# Muon orthogonalization coefficients.
MUON_NS_COEFFS: List[Tuple[float, float, float]] = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
]

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        gen = torch.Generator(device="cuda")
    else:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _orthogonalize(M: Tensor, niter: int = len(MUON_NS_COEFFS)) -> Tensor:
    # Computes msign(M) = M (M^T M)^{-1/2}
    transpose = M.shape[0] > M.shape[1]
    if transpose:
        M = M.mT
    norm = torch.linalg.norm(M, dim=(-2, -1), keepdim=True)
    M = M / (norm + 1e-20)
    for a, b, c in MUON_NS_COEFFS[:niter]:
        M = a * M + (b * (U := M @ M.mT) + c * (U @ U)) @ M
    if transpose:
        M = M.mT
    return M


def _supported_root_or_raise(r: int) -> int:
    if r <= 0 or r >= len(NTH_ROOT_COEFS) or NTH_ROOT_COEFS[r] is None:
        raise ValueError(f"Unsupported root r={r}. Supported roots: 1, 2 and 4.")
    return r


def _normalize_supported_root(root: float) -> int:
    r = int(round(float(root)))
    if abs(float(root) - float(r)) > 1e-8:
        raise ValueError(f"root must be an integer in {{2, 4}}, got {root}")
    return _supported_root_or_raise(r)


def abc(
    r: int = 2,
    steps: Optional[int] = None,
    scale: float = 1.0,
):
    r = _supported_root_or_raise(r)
    w = NTH_ROOT_COEFS[r]
    assert w is not None
    max_steps = steps or len(w)
    for a, b, c in w[:max_steps] + w[-1:] * max(max_steps - len(w), 0):
        yield a / scale, b / (scale ** (r + 1)), c / (scale ** (2 * r + 1))


def _sym(M: Tensor) -> Tensor:
    return 0.5 * (M + M.mT)


def matrix_root(
    P: Tensor,
    *,
    r: int,
    s: int = 1,
    steps: int = 8,
    eps: float = 1e-6,
    scale: float = 1.01,
) -> Tensor:
    # Computes P^{s/r}
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"
    norm = torch.linalg.norm(P)
    norm_value = float(norm.item())
    if norm_value <= eps:
        return torch.zeros_like(P)
    Y = YZ = P / norm
    I_n = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    for a, b, c in abc(r=r, steps=steps, scale=scale):
        W = a * I_n + b * YZ + c * (YZ @ YZ)
        W1 = torch.linalg.matrix_power(W, s)
        W2 = torch.linalg.matrix_power(W, r)
        Y, YZ = W1 @ Y, W2 @ YZ
    return Y * norm**(float(s) / float(r))


def double_sided_matmul_root(
    Q: Tensor,
    G: Tensor,
    P: Tensor,
    *,
    r: int,
    s: int = 1,
    steps: int = 8,
    eps: float = 1e-6,
    scale: float = 1.01,
) -> Tensor:
    """
    Computes Q^(s/r) @ G @ P^(s/r) using matrix roots on each side.
    """
    assert Q.ndim == 2 and Q.shape[0] == Q.shape[1], "Q must be square"
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"
    assert G.ndim == 2 and G.shape[0] == Q.shape[0] and G.shape[1] == P.shape[0], "shape mismatch"

    left_root = matrix_root(Q, r=r, s=s, steps=steps, eps=eps, scale=scale)
    right_root = matrix_root(P, r=r, s=s, steps=steps, eps=eps, scale=scale)
    return left_root @ G @ right_root


def matrix_invroot(
    P: Tensor,
    *,
    r: int,
    s: int = 1,
    steps: Optional[int] = None,
    eps: float = 1e-5,
    scale: float = 1.001,
) -> Tensor:
    """
    Computes P^(-s/r) using Newton-Schulz iterations.
    """
    r = _supported_root_or_raise(r)
    if s < 0:
        raise ValueError("matrix_invroot expects s >= 0")
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"

    I_n = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    t = torch.linalg.norm(P)
    t_value = float(t.item())
    t_safe = t if t_value > eps else torch.tensor(1.0, device=P.device, dtype=P.dtype)
    Pn = P / t_safe + eps * I_n
    out = I_n
    for a, b, c in abc(r=r, steps=steps, scale=scale):
        W = a * I_n + b * Pn + c * (Pn @ Pn)
        W1 = torch.linalg.matrix_power(W, s)
        W2 = torch.linalg.matrix_power(W, r)
        out = out @ W1
        Pn = _sym(Pn @ W2)
    return out * ((t ** (-float(s) / float(r))) if t_value > eps else 0.0)


def double_sided_matmul_invroot(
    Q: Tensor,
    G: Tensor,
    P: Tensor,
    *,
    r: int,
    s: int = 1,
    steps: Optional[int] = None,
    eps: float = 1e-5,
    scale: float = 1.001,
) -> Tensor:
    """
    Computes Q^(-s/r) @ G @ P^(-s/r) using coupled Newton-Schulz iterations.
    """
    r = _supported_root_or_raise(r)
    assert Q.ndim == 2 and Q.shape[0] == Q.shape[1], "Q must be square"
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"
    assert G.ndim == 2 and G.shape[0] == Q.shape[0] and G.shape[1] == P.shape[0], "shape mismatch"

    I_m = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    I_n = torch.eye(G.shape[1], device=G.device, dtype=G.dtype)

    t_q = torch.linalg.norm(Q)
    t_p = torch.linalg.norm(P)
    t_q_value = float(t_q.item())
    t_p_value = float(t_p.item())
    t_q_safe = t_q if t_q_value > eps else torch.tensor(1.0, device=G.device, dtype=G.dtype)
    t_p_safe = t_p if t_p_value > eps else torch.tensor(1.0, device=G.device, dtype=G.dtype)

    Qn = Q / t_q_safe + eps * I_m
    Pn = P / t_p_safe + eps * I_n
    out = G
    for a, b, c in abc(r=r, steps=steps, scale=scale):
        W_q = a * I_m + b * Qn + c * (Qn @ Qn)
        W_p = a * I_n + b * Pn + c * (Pn @ Pn)
        W_q1 = torch.linalg.matrix_power(W_q, s)
        W_q2 = torch.linalg.matrix_power(W_q, r)
        W_p1 = torch.linalg.matrix_power(W_p, s)
        W_p2 = torch.linalg.matrix_power(W_p, r)
        Qn = _sym(Qn @ W_q2)
        out = W_q1 @ out @ W_p1
        Pn = _sym(Pn @ W_p2)

    q_scale = (t_q ** (-float(s) / float(r))) if t_q_value > eps else 0.0
    p_scale = (t_p ** (-float(s) / float(r))) if t_p_value > eps else 0.0
    return out * q_scale * p_scale


def _matrix_update_scale(p: Tensor) -> float:
    if p.ndim != 2:
        return 1.0
    fan_out, fan_in = p.shape
    return math.sqrt(float(fan_out) / float(fan_in))


def norm_lower_bound(A: Tensor) -> Tensor:
    """
    Cheap lower bound for spectral norm used by Muon-style PSGD preconditioner updates.
    """
    max_abs = torch.max(torch.abs(A))
    if float(max_abs.item()) <= 0.0:
        return max_abs

    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), dim=0)
    value1, j = torch.max(torch.sum(aa, dim=1), dim=0)
    A_h = A.transpose(-2, -1).conj()
    eps = 1.2e-38

    if value0 > value1:
        x = A[:, int(i.item())].conj() @ A
        x = x / (torch.linalg.vector_norm(x) + eps)
        return max_abs * torch.linalg.vector_norm(x @ A_h)

    x = A @ A[int(j.item())].conj()
    x = x / (torch.linalg.vector_norm(x) + eps)
    return max_abs * torch.linalg.vector_norm(A_h @ x)


def _psgd_apply(
    M: Tensor,
    Q: Tensor,
    invQ: Tensor,
    *,
    lr_preconditioner: float,
    preconditioner_update_probability: float,
    generator: Optional[torch.Generator],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Muon-style PSGD by Xilin Li, generalized to both tall and short matrices.
    """
    m, n = M.shape
    k = min(m, n)
    assert Q.shape == (k, k), f"Q shape {tuple(Q.shape)} incompatible with M shape {tuple(M.shape)}"
    assert invQ.shape == (k, k), f"invQ shape {tuple(invQ.shape)} incompatible with M shape {tuple(M.shape)}"

    if m >= n:
        A = M @ Q.mT
        precond_M = A @ Q
    else:
        M_t = M.mT
        A = M_t @ Q.mT
        precond_M = (A @ Q).mT

    rand = torch.rand((), device=M.device, generator=generator)
    update_precond = bool(rand < preconditioner_update_probability)

    if update_precond:
        inv_q_h_inv_q = invQ.mT @ invQ
        ah_a = A.mT @ A
        lr_den = norm_lower_bound(ah_a + inv_q_h_inv_q) + 1.2e-38
        lr = lr_preconditioner / float(lr_den.item())
        Q = Q - lr * (torch.triu(ah_a - inv_q_h_inv_q) @ Q)
        I_k = torch.eye(k, device=M.device, dtype=M.dtype)
        invQ = torch.linalg.solve_triangular(Q, I_k, upper=True)

    return precond_M, Q, invQ


def _psgd_inverse_apply(E: Tensor, invQ: Tensor) -> Tensor:
    """
    Applies the inverse of Muon-style PSGD linear map used in ECO pullback.
    """
    m, n = E.shape
    inv_s = invQ @ invQ.mT
    if m >= n:
        return E @ inv_s
    return inv_s @ E


def _quantize_fp8_e4m3_no_scale(
    x: Tensor,
    *,
    stochastic_rounding: bool,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    FP8-E4M3 quantization for values already in-scale.
    Returns dequantized float tensor (same dtype as input).
    """
    x32 = x.float()
    x_sat = torch.nan_to_num(x32, nan=0.0, posinf=FP8_E4M3_MAX, neginf=-FP8_E4M3_MAX)
    x_sat = torch.clamp(x_sat, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    if not stochastic_rounding:
        # Fast RNE path: use native FP8 cast and dequantize back.
        return x_sat.to(dtype=FP8_STORAGE_DTYPE).to(dtype=x.dtype)

    # Deterministic SR path (given generator): vectorized, avoids gather/scatter masks.
    sign = torch.sign(x_sat)
    ax = x_sat.abs()
    zero_mask = ax == 0
    normal_mask = ax >= FP8_E4M3_MIN_NORMAL
    # Use separate clamps for normal and subnormal computations.
    # Subnormal path must use raw ax (not clamped to min subnormal) so tiny
    # values can still stochastically round to zero.
    ax_for_normal = torch.clamp(ax, min=FP8_E4M3_MIN_NORMAL)

    if generator is None:
        rand = torch.rand_like(ax)
    else:
        rand = torch.rand(ax.shape, device=ax.device, dtype=ax.dtype, generator=generator)

    exp = torch.clamp(torch.floor(torch.log2(ax_for_normal)), min=-6.0, max=7.0)
    base = torch.exp2(exp)
    mant_scaled = (ax_for_normal / base - 1.0) * 8.0
    mant_i = torch.floor(mant_scaled + rand)
    mant_i = torch.clamp(mant_i, min=0.0, max=8.0)
    carry = mant_i >= 8.0
    exp = torch.clamp(exp + carry.to(dtype=exp.dtype), min=-6.0, max=7.0)
    mant_i = torch.where(carry, torch.zeros_like(mant_i), mant_i)
    q_normal = (1.0 + torch.clamp(mant_i, min=0.0, max=7.0) / 8.0) * torch.exp2(exp)

    sub_scaled = ax / FP8_E4M3_MIN_SUBNORMAL
    sub_i = torch.floor(sub_scaled + rand)
    q_sub = torch.clamp(sub_i, min=0.0, max=7.0) * FP8_E4M3_MIN_SUBNORMAL

    q_abs = torch.where(normal_mask, q_normal, q_sub)
    q_abs = torch.where(zero_mask, torch.zeros_like(q_abs), q_abs)
    return (sign * q_abs).to(dtype=x.dtype)


def _quantize_fp8_with_scale(
    x: Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    x32 = x.float()
    scale = compute_fp8_scale(x32, scale_mode=scale_mode)
    return quantize_fp8_e4m3_with_scale(
        x32,
        scale=scale,
        stochastic_rounding=stochastic_rounding,
        generator=generator,
    ).to(dtype=x.dtype)


def compute_fp8_scale(
    x: Tensor,
    *,
    scale_mode: ScaleMode,
) -> Tensor:
    x32 = x.float()
    if scale_mode == "tensor" or x32.ndim < 2:
        amax = x32.abs().amax()
    elif scale_mode == "row":
        reduce_dims = tuple(range(1, x32.ndim))
        amax = x32.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")
    return torch.clamp(amax / FP8_E4M3_MAX, min=1e-12)


def quantize_fp8_e4m3_with_scale(
    x: Tensor,
    *,
    scale: Tensor,
    stochastic_rounding: bool,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    x32 = x.float()
    q_unit = _quantize_fp8_e4m3_no_scale(
        x32 / scale,
        stochastic_rounding=stochastic_rounding,
        generator=generator,
    )
    return (q_unit * scale).to(dtype=x.dtype)


def quantize_fp8_e4m3(
    x: Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode = "row",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if scale_mode == "none":
        return _quantize_fp8_e4m3_no_scale(
            x,
            stochastic_rounding=stochastic_rounding,
            generator=generator,
        )
    return _quantize_fp8_with_scale(
        x,
        stochastic_rounding=stochastic_rounding,
        scale_mode=scale_mode,
        generator=generator,
    )


def quantize_ste(
    x: Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Straight-through quantization: forward uses quantized value, backward uses identity.
    """
    q = quantize_fp8_e4m3(
        x,
        stochastic_rounding=stochastic_rounding,
        scale_mode=scale_mode,
        generator=generator,
    )
    return x + (q - x).detach()


@dataclass
class QuantForwardConfig:
    weight_scale_mode: ScaleMode = "row"
    enable_weight_quantization: bool = True
    weight_stochastic_rounding: bool = True
    activation_stochastic_rounding: bool = False
    activation_scale_mode: ScaleMode = "row"
    sr_generator: Optional[torch.Generator] = None


def quantize_weight_ste(x: Tensor, qcfg: QuantForwardConfig) -> Tensor:
    if not qcfg.enable_weight_quantization:
        return x
    return quantize_ste(
        x,
        stochastic_rounding=qcfg.weight_stochastic_rounding,
        scale_mode=qcfg.weight_scale_mode,
        generator=qcfg.sr_generator,
    )


def quantize_activation_ste(x: Tensor, qcfg: QuantForwardConfig) -> Tensor:
    return quantize_ste(
        x,
        stochastic_rounding=qcfg.activation_stochastic_rounding,
        scale_mode=qcfg.activation_scale_mode,
        generator=qcfg.sr_generator,
    )


def norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_NORM_EPS)


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        qcfg: QuantForwardConfig,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qcfg = qcfg
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        with torch.no_grad():
            self.weight.copy_(_orthogonalize(self.weight.data, niter=8))

    def forward(self, x: Tensor) -> Tensor:
        x_q = quantize_activation_ste(x, self.qcfg)
        w = quantize_weight_ste(self.weight, self.qcfg)
        out = F.linear(x_q, w, None)
        return out


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, mlp_hidden: int, n_layers: int, *, qcfg: QuantForwardConfig):
        super().__init__()
        self.alpha = 1. / n_layers
        self.qcfg = qcfg
        self.fc1 = QuantizedLinear(d_model, mlp_hidden, qcfg=qcfg)
        self.fc2 = QuantizedLinear(mlp_hidden, d_model, qcfg=qcfg)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(F.gelu(self.fc1(norm(x))))
        return (1. - self.alpha) * x + self.alpha * h


class ResidualMLPLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        mlp_hidden: int,
        qcfg: QuantForwardConfig,
        lm_head_quantized: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.qcfg = qcfg
        self.lm_head_quantized = lm_head_quantized

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Parameter(torch.empty(block_size, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model, mlp_hidden, n_layers, qcfg=qcfg) for _ in range(n_layers)]
        )
        if self.lm_head_quantized:
            self.lm_head = QuantizedLinear(d_model, vocab_size, qcfg=qcfg)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, seqlen = idx.shape
        if seqlen > self.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.block_size}.")

        tok = self.token_embedding(idx)
        pos = self.position_embedding[:seqlen]
        x = tok + pos.unsqueeze(0)
        x = norm(x)

        for block in self.blocks:
            x = block(x)

        x = norm(x)
        logits = self.lm_head(x)
        logits = logits.float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class FP8QuantizedParamMixin:
    sr_generator: Optional[torch.Generator]

    def _ensure_fp8_scale_state(
        self,
        p: Tensor,
        state: Dict[str, Tensor],
        *,
        scale_mode: ScaleMode,
    ) -> None:
        if "fp8_scale" not in state:
            state["fp8_scale"] = compute_fp8_scale(p.data, scale_mode=scale_mode).to(torch.float32)

    def _quantize_with_group_config(
        self,
        p: Tensor,
        x: Tensor,
        *,
        group: Dict[str, object],
        state: Dict[str, Tensor],
    ) -> Tensor:
        scale_mode = group["fp8_scale_mode"]
        sr = group["fp8_stochastic_rounding"]
        use_ema_scales = group["use_ema_scales"]
        scale_ema_decay = group["scale_ema_decay"]

        if use_ema_scales:
            scale_now = compute_fp8_scale(x, scale_mode=scale_mode).to(torch.float32)
            scale_ema = state.get("fp8_scale")
            if scale_ema is None:
                scale_ema = scale_now.detach().clone()
            else:
                scale_ema = scale_ema * scale_ema_decay + scale_now * (1.0 - scale_ema_decay)
            state["fp8_scale"] = scale_ema
            return quantize_fp8_e4m3_with_scale(
                x,
                scale=scale_ema,
                stochastic_rounding=sr,
                generator=self.sr_generator,
            )

        return quantize_fp8_e4m3(
            x,
            stochastic_rounding=sr,
            scale_mode=scale_mode,
            generator=self.sr_generator,
        )

    @torch.no_grad()
    def _quantize_all_params(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                q = self._quantize_with_group_config(
                    p,
                    p.data,
                    group=group,
                    state=state,
                )
                p.copy_(q.to(dtype=p.dtype))


class FP8AdamWNoMasterECO(FP8QuantizedParamMixin, torch.optim.Optimizer):
    """
    FP8 AdamW without master weights, optionally with ECO momentum compensation.

    Update:
      m_tilde = beta1 * m + (1 - beta1) * g
      v_tilde = beta2 * v + (1 - beta2) * g^2
      w_tilde = (1 - lr*wd) * w_q - lr * (m_tilde/(1-beta1^t)) / (sqrt(v_tilde/(1-beta2^t))+eps)
      w_q_next = q(w_tilde)
      e = w_tilde - w_q_next
      m_next = m_tilde + ((1-lr*wd)*(1-beta1^t)/lr) * (1 - 1/beta1) * denom * e
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.9),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        fp8_scale_mode: ScaleMode = "row",
        fp8_stochastic_rounding: bool = True,
        eco_compensation: bool = True,
        use_ema_scales: bool = True,
        scale_ema_decay: float = DEFAULT_SCALE_EMA_DECAY,
        sr_generator: Optional[torch.Generator] = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError("betas must be in (0, 1)")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if not (0.0 <= scale_ema_decay < 1.0):
            raise ValueError("scale_ema_decay must be in [0, 1)")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            fp8_scale_mode=fp8_scale_mode,
            fp8_stochastic_rounding=fp8_stochastic_rounding,
            eco_compensation=eco_compensation,
            use_ema_scales=use_ema_scales,
            scale_ema_decay=scale_ema_decay,
        )
        super().__init__(params, defaults)
        self.sr_generator = sr_generator
        self._quantize_all_params()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]

            eco_coef = ((1.0 - lr * wd) / lr) * (1.0 - 1.0 / beta1)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if use_ema_scales:
                        self._ensure_fp8_scale_state(
                            p,
                            state,
                            scale_mode=group["fp8_scale_mode"],
                        )

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta1 * m + (1.0 - beta1) * g
                v_tilde = beta2 * v + (1.0 - beta2) * (g * g)

                bias_c1 = 1.0 - beta1**t
                bias_c2 = 1.0 - beta2**t
                denom = torch.sqrt(v_tilde / bias_c2) + eps
                adam_update = (m_tilde / bias_c1) / denom

                w_q = p.detach().to(torch.float32)
                w_tilde = (1.0 - lr * wd) * w_q - lr * adam_update
                w_q_next = self._quantize_with_group_config(
                    p,
                    w_tilde,
                    group=group,
                    state=state,
                )
                e = w_tilde - w_q_next

                if eco_comp:
                    comp = bias_c1 * denom * e
                    m_next = m_tilde + eco_coef * comp
                else:
                    m_next = m_tilde

                p.copy_(w_q_next.to(dtype=p.dtype))
                m.copy_(m_next)
                v.copy_(v_tilde)

        return loss


class MuonOptimizer(torch.optim.Optimizer):
    """
    FP32 Muon baseline:
    - momentum in FP32
    - matrix params use Newton-Schulz orthogonalized updates
    - non-matrix params use SGDM-style momentum update
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        ns_steps: int = 8,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be > 0")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"MuonOptimizer expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                update = _matrix_update_scale(p) * _orthogonalize(m_tilde, niter=ns_steps)

                w_next = (1.0 - lr * wd) * p.detach().to(torch.float32) - lr * update
                p.copy_(w_next.to(dtype=p.dtype))
                m.copy_(m_tilde)

        return loss


class PSGDOptimizer(torch.optim.Optimizer):
    """
    Reference Muon-style PSGD with FP32 states and decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        lr_preconditioner: float = 1.0,
        preconditioner_update_probability: float = 1.0,
        precond_generator: Optional[torch.Generator] = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if lr_preconditioner <= 0:
            raise ValueError("lr_preconditioner must be > 0")
        if not (0.0 <= preconditioner_update_probability <= 1.0):
            raise ValueError("preconditioner_update_probability must be in [0, 1]")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            lr_preconditioner=lr_preconditioner,
            preconditioner_update_probability=preconditioner_update_probability,
        )
        super().__init__(params, defaults)
        self.precond_generator = precond_generator

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            lr_precond = group["lr_preconditioner"]
            precond_prob = group["preconditioner_update_probability"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"PSGDOptimizer expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    k = min(p.shape)
                    state["q_factor"] = torch.eye(k, device=p.device, dtype=torch.float32)
                    state["inv_q_factor"] = torch.eye(k, device=p.device, dtype=torch.float32)

                m = state["momentum"]
                q_factor = state["q_factor"]
                inv_q_factor = state["inv_q_factor"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                precond_m, q_next, inv_q_next = _psgd_apply(
                    m_tilde,
                    q_factor,
                    inv_q_factor,
                    lr_preconditioner=lr_precond,
                    preconditioner_update_probability=precond_prob,
                    generator=self.precond_generator,
                )
                update = _matrix_update_scale(p) * precond_m

                w_next = (1.0 - lr * wd) * p.detach().to(torch.float32) - lr * update
                p.copy_(w_next.to(dtype=p.dtype))
                m.copy_(m_tilde)
                q_factor.copy_(q_next.to(dtype=q_factor.dtype))
                inv_q_factor.copy_(inv_q_next.to(dtype=inv_q_factor.dtype))

        return loss


class FP8PSGDNoMasterECO(FP8QuantizedParamMixin, torch.optim.Optimizer):
    """
    FP8 Muon-style PSGD without master weights, with optional ECO compensation.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        lr_preconditioner: float = 1.0,
        preconditioner_update_probability: float = 1.0,
        fp8_scale_mode: ScaleMode = "row",
        fp8_stochastic_rounding: bool = True,
        eco_compensation: bool = True,
        use_ema_scales: bool = True,
        scale_ema_decay: float = DEFAULT_SCALE_EMA_DECAY,
        sr_generator: Optional[torch.Generator] = None,
        precond_generator: Optional[torch.Generator] = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if lr_preconditioner <= 0:
            raise ValueError("lr_preconditioner must be > 0")
        if not (0.0 <= preconditioner_update_probability <= 1.0):
            raise ValueError("preconditioner_update_probability must be in [0, 1]")
        if not (0.0 <= scale_ema_decay < 1.0):
            raise ValueError("scale_ema_decay must be in [0, 1)")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            lr_preconditioner=lr_preconditioner,
            preconditioner_update_probability=preconditioner_update_probability,
            fp8_scale_mode=fp8_scale_mode,
            fp8_stochastic_rounding=fp8_stochastic_rounding,
            eco_compensation=eco_compensation,
            use_ema_scales=use_ema_scales,
            scale_ema_decay=scale_ema_decay,
        )
        super().__init__(params, defaults)
        self.sr_generator = sr_generator
        self.precond_generator = precond_generator
        self._quantize_all_params()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            lr_precond = group["lr_preconditioner"]
            precond_prob = group["preconditioner_update_probability"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]

            eco_coef = ((1.0 - lr * wd) / lr) * (1.0 - 1.0 / beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"FP8PSGDNoMasterECO expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    k = min(p.shape)
                    state["q_factor"] = torch.eye(k, device=p.device, dtype=torch.float32)
                    state["inv_q_factor"] = torch.eye(k, device=p.device, dtype=torch.float32)
                    if use_ema_scales:
                        self._ensure_fp8_scale_state(
                            p,
                            state,
                            scale_mode=group["fp8_scale_mode"],
                        )

                m = state["momentum"]
                q_factor = state["q_factor"]
                inv_q_factor = state["inv_q_factor"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                precond_m, q_next, inv_q_next = _psgd_apply(
                    m_tilde,
                    q_factor,
                    inv_q_factor,
                    lr_preconditioner=lr_precond,
                    preconditioner_update_probability=precond_prob,
                    generator=self.precond_generator,
                )
                matrix_scale = _matrix_update_scale(p)
                u = matrix_scale * precond_m

                w_q = p.detach().to(torch.float32)
                w_tilde = (1.0 - lr * wd) * w_q - lr * u
                w_q_next = self._quantize_with_group_config(
                    p,
                    w_tilde,
                    group=group,
                    state=state,
                )

                e = w_tilde - w_q_next
                if eco_comp:
                    comp = _psgd_inverse_apply(e, inv_q_next)
                    m_next = m_tilde + (eco_coef / matrix_scale) * comp
                else:
                    m_next = m_tilde

                p.copy_(w_q_next.to(dtype=p.dtype))
                m.copy_(m_next)
                q_factor.copy_(q_next.to(dtype=q_factor.dtype))
                inv_q_factor.copy_(inv_q_next.to(dtype=inv_q_factor.dtype))

        return loss


class ShampooOptimizer(torch.optim.Optimizer):
    """
    Reference Shampoo with momentum and decoupled weight decay.

    Matrix Shampoo update:
      U_t = L_t^{-1/r} @ M_tilde @ R_t^{-1/r}
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        precond_beta: float = 0.9,
        root: float = 4.0,
        ns_steps: int = 8,
        ns_eps: float = 1e-6,
        ns_scale: float = 1.001,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if not (0.0 <= precond_beta < 1.0):
            raise ValueError("precond_beta must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be > 0")
        if ns_eps <= 0:
            raise ValueError("ns_eps must be > 0")
        if ns_scale <= 0:
            raise ValueError("ns_scale must be > 0")
        _normalize_supported_root(root)

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            precond_beta=precond_beta,
            root=root,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            ns_scale=ns_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            precond_beta = group["precond_beta"]
            root = _normalize_supported_root(group["root"])
            ns_steps = group["ns_steps"]
            ns_eps = group["ns_eps"]
            ns_scale = group["ns_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"ShampooOptimizer expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    m_dim, n_dim = p.shape
                    state["left_precond"] = torch.zeros((m_dim, m_dim), device=p.device, dtype=torch.float32)
                    state["right_precond"] = torch.zeros((n_dim, n_dim), device=p.device, dtype=torch.float32)
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                left = state["left_precond"]
                right = state["right_precond"]
                left_t = precond_beta * left + (1.0 - precond_beta) * (g @ g.mT)
                right_t = precond_beta * right + (1.0 - precond_beta) * (g.mT @ g)
                update = _matrix_update_scale(p) * double_sided_matmul_invroot(
                    left_t,
                    m_tilde,
                    right_t,
                    r=root,
                    s=1,
                    steps=ns_steps,
                    eps=ns_eps,
                    scale=ns_scale,
                )
                state["left_precond"] = left_t
                state["right_precond"] = right_t

                w_next = (1.0 - lr * wd) * p.detach().to(torch.float32) - lr * update
                p.copy_(w_next.to(dtype=p.dtype))
                m.copy_(m_tilde)

        return loss


class FP8ShampooNoMasterECO(FP8QuantizedParamMixin, torch.optim.Optimizer):
    """
    FP8 Shampoo without master weights, with optional ECO compensation.

    Uses matrix Shampoo preconditioners:
      U_t = L_t^{-1/r} @ M_tilde @ R_t^{-1/r}
      M_next = M_tilde + coef * L_t^{1/r} @ E @ R_t^{1/r}
    where coef = ((1 - lr*wd)/lr) * (1 - 1/beta).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        precond_beta: float = 0.9,
        root: float = 4.0,
        ns_steps: int = 8,
        ns_eps: float = 1e-6,
        ns_scale: float = 1.001,
        fp8_scale_mode: ScaleMode = "row",
        fp8_stochastic_rounding: bool = True,
        eco_compensation: bool = True,
        use_ema_scales: bool = True,
        scale_ema_decay: float = DEFAULT_SCALE_EMA_DECAY,
        sr_generator: Optional[torch.Generator] = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if not (0.0 <= precond_beta < 1.0):
            raise ValueError("precond_beta must be in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be > 0")
        if ns_eps <= 0:
            raise ValueError("ns_eps must be > 0")
        if ns_scale <= 0:
            raise ValueError("ns_scale must be > 0")
        _normalize_supported_root(root)
        if not (0.0 <= scale_ema_decay < 1.0):
            raise ValueError("scale_ema_decay must be in [0, 1)")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            precond_beta=precond_beta,
            root=root,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            ns_scale=ns_scale,
            fp8_scale_mode=fp8_scale_mode,
            fp8_stochastic_rounding=fp8_stochastic_rounding,
            eco_compensation=eco_compensation,
            use_ema_scales=use_ema_scales,
            scale_ema_decay=scale_ema_decay,
        )
        super().__init__(params, defaults)
        self.sr_generator = sr_generator
        self._quantize_all_params()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            precond_beta = group["precond_beta"]
            root = _normalize_supported_root(group["root"])
            ns_steps = group["ns_steps"]
            ns_eps = group["ns_eps"]
            ns_scale = group["ns_scale"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]

            eco_coef = ((1.0 - lr * wd) / lr) * (1.0 - 1.0 / beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"FP8ShampooNoMasterECO expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    m_dim, n_dim = p.shape
                    state["left_precond"] = torch.zeros((m_dim, m_dim), device=p.device, dtype=torch.float32)
                    state["right_precond"] = torch.zeros((n_dim, n_dim), device=p.device, dtype=torch.float32)
                    if use_ema_scales:
                        self._ensure_fp8_scale_state(
                            p,
                            state,
                            scale_mode=group["fp8_scale_mode"],
                        )
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                left = state["left_precond"]
                right = state["right_precond"]
                matrix_scale = _matrix_update_scale(p)
                left_t = precond_beta * left + (1.0 - precond_beta) * (g @ g.mT)
                right_t = precond_beta * right + (1.0 - precond_beta) * (g.mT @ g)
                update = matrix_scale * double_sided_matmul_invroot(
                    left_t,
                    m_tilde,
                    right_t,
                    r=root,
                    s=1,
                    steps=ns_steps,
                    eps=ns_eps,
                    scale=ns_scale,
                )

                w_q = p.detach().to(torch.float32)
                w_tilde = (1.0 - lr * wd) * w_q - lr * update
                w_q_next = self._quantize_with_group_config(
                    p,
                    w_tilde,
                    group=group,
                    state=state,
                )

                e = w_tilde - w_q_next
                if eco_comp:
                    comp = double_sided_matmul_root(
                        left_t,
                        e,
                        right_t,
                        r=root,
                        s=1,
                        steps=ns_steps,
                        eps=ns_eps,
                        scale=ns_scale,
                    )
                    m_next = m_tilde + (eco_coef / matrix_scale) * comp
                else:
                    m_next = m_tilde

                p.copy_(w_q_next.to(dtype=p.dtype))
                m.copy_(m_next)
                state["left_precond"] = left_t
                state["right_precond"] = right_t

        return loss


class FP8MuonNoMasterECO(FP8QuantizedParamMixin, torch.optim.Optimizer):
    """
    FP8 Muon without master weights, with optional ECO compensation.

    - Materialized weights stay quantized to FP8 each step.
    - Momentum stays in FP32.
    - Matrix params use Muon orthogonalization.
    - Optional ECO compensation is applied to the momentum state.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 3e-4,
        beta: float = 0.9,
        weight_decay: float = 0.1,
        ns_steps: int = 8,
        ns_eps: float = 1e-6,
        ns_scale: float = 1.001,
        fp8_scale_mode: ScaleMode = "row",
        fp8_stochastic_rounding: bool = True,
        eco_compensation: bool = True,
        use_ema_scales: bool = True,
        scale_ema_decay: float = DEFAULT_SCALE_EMA_DECAY,
        sr_generator: Optional[torch.Generator] = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be > 0")
        if ns_eps <= 0:
            raise ValueError("ns_eps must be > 0")
        if ns_scale <= 0:
            raise ValueError("ns_scale must be > 0")
        if not (0.0 <= scale_ema_decay < 1.0):
            raise ValueError("scale_ema_decay must be in [0, 1)")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            ns_scale=ns_scale,
            fp8_scale_mode=fp8_scale_mode,
            fp8_stochastic_rounding=fp8_stochastic_rounding,
            eco_compensation=eco_compensation,
            use_ema_scales=use_ema_scales,
            scale_ema_decay=scale_ema_decay,
        )
        super().__init__(params, defaults)
        self.sr_generator = sr_generator
        self._quantize_all_params()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            ns_eps = group["ns_eps"]
            ns_scale = group["ns_scale"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]

            eco_coef = ((1.0 - lr * wd) / lr) * (1.0 - 1.0 / beta)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                assert p.ndim == 2, (
                    f"FP8MuonNoMasterECO expects only 2D weight parameters, got shape {tuple(p.shape)}"
                )

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    if use_ema_scales:
                        self._ensure_fp8_scale_state(
                            p,
                            state,
                            scale_mode=group["fp8_scale_mode"],
                        )
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g
                matrix_scale = _matrix_update_scale(p)
                u = matrix_scale * _orthogonalize(m_tilde, niter=ns_steps)

                w_q = p.detach().to(torch.float32)
                w_tilde = (1.0 - lr * wd) * w_q - lr * u
                w_q_next = self._quantize_with_group_config(
                    p,
                    w_tilde,
                    group=group,
                    state=state,
                )

                e = w_tilde - w_q_next
                if eco_comp:
                    gram = m_tilde.mT @ m_tilde
                    comp = e @ matrix_root(gram, r=2, steps=ns_steps, eps=ns_eps, scale=ns_scale)
                    m_next = m_tilde + (eco_coef / matrix_scale) * comp
                else:
                    m_next = m_tilde

                p.copy_(w_q_next.to(dtype=p.dtype))
                m.copy_(m_next)

        return loss


class CompositeOptimizer:
    """
    Thin wrapper that steps/zeros multiple optimizers as one.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for i, opt in enumerate(self.optimizers):
            if i == 0 and closure is not None:
                loss = opt.step(closure)
            else:
                opt.step()
        return loss


class TinyShakespeareData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.text_path = data_dir / "input.txt"
        self._prepare_if_needed()
        self._load()

    def _prepare_if_needed(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.text_path.exists():
            return
        print(f"Downloading Tiny Shakespeare to {self.text_path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, self.text_path)

    def _load(self) -> None:
        text = self.text_path.read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        encoded = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        split = int(0.9 * len(encoded))
        self.train_data = encoded[:split]
        self.val_data = encoded[split:]

    def get_batch(
        self,
        split: Literal["train", "val"],
        *,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> Tuple[Tensor, Tensor]:
        data = self.train_data if split == "train" else self.val_data
        max_start = data.size(0) - block_size - 1
        if max_start <= 0:
            raise ValueError("Dataset is too small for the chosen block_size.")

        ix = torch.randint(0, max_start, (batch_size,), generator=generator)
        offsets = torch.arange(block_size)
        x = data[ix[:, None] + offsets[None, :]]
        y = data[ix[:, None] + offsets[None, :] + 1]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@dataclass
class TrainConfig:
    steps: int
    eval_interval: int
    eval_batches: int
    batch_size: int
    block_size: int
    grad_clip: float


def cosine_lr_with_floor(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 1:
        return peak_lr
    min_lr = peak_lr * min_lr_ratio
    progress = float(step) / float(total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TinyShakespeareData,
    *,
    device: torch.device,
    cfg: TrainConfig,
    rng: torch.Generator,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split in ("train", "val"):
        losses: List[float] = []
        for _ in range(cfg.eval_batches):
            xb, yb = dataset.get_batch(
                split,  # type: ignore[arg-type]
                batch_size=cfg.batch_size,
                block_size=cfg.block_size,
                device=device,
                generator=rng,
            )
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")
            losses.append(float(loss.item()))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_one_run(
    *,
    run_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: TinyShakespeareData,
    device: torch.device,
    cfg: TrainConfig,
    data_seed: int,
    peak_lr: float,
    min_lr_ratio: float,
) -> List[Dict[str, float]]:
    logs: List[Dict[str, float]] = []
    train_rng = torch.Generator().manual_seed(data_seed)
    eval_rng = torch.Generator().manual_seed(data_seed + 10_000)

    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            stats = evaluate(model, dataset, device=device, cfg=cfg, rng=eval_rng)
            elapsed = time.time() - t0
            current_lr = float(optimizer.param_groups[0]["lr"])
            rec = {
                "step": float(step),
                "train_loss": stats["train"],
                "val_loss": stats["val"],
                "lr": current_lr,
                "elapsed_sec": elapsed,
            }
            logs.append(rec)
            print(
                f"[{run_name}] step={step:5d} "
                f"train_loss={stats['train']:.4f} val_loss={stats['val']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

        if step == cfg.steps:
            break

        xb, yb = dataset.get_batch(
            "train",
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            device=device,
            generator=train_rng,
        )

        current_lr = cosine_lr_with_floor(
            step,
            total_steps=cfg.steps,
            peak_lr=peak_lr,
            min_lr_ratio=min_lr_ratio,
        )
        for param_group in optimizer.param_groups:
            lr_scale = float(param_group.get("lr_scale", 1.0))
            param_group["lr"] = current_lr * lr_scale

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should not be None in training.")
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return logs


def plot_loss_curves(
    all_logs: Dict[str, List[Dict[str, float]]],
    output_path: Path,
    optimizer_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it in the nanogpt env.") from exc

    label_map = {
        "reference": "Reference (FP32 master)",
        "eco_fp8_nomaster": "FP8 no master + ECO",
        "fp8_nomaster_noeco": "FP8 no master (no ECO)",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    all_steps: List[int] = []
    late_horizon_vals: List[float] = []
    for run_name, logs in all_logs.items():
        if not logs:
            continue
        steps = [int(rec["step"]) for rec in logs]
        val_losses = [rec["val_loss"] for rec in logs]
        all_steps.extend(steps)
        label = label_map.get(run_name, run_name)
        plt.plot(steps, val_losses, marker="o", linewidth=2.0, label=label)

    if all_steps:
        max_step = max(all_steps)
        cutoff = 0.25 * float(max_step)
        for logs in all_logs.values():
            for rec in logs:
                if float(rec["step"]) >= cutoff:
                    late_horizon_vals.append(float(rec["val_loss"]))

    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    # plt.yscale("log")
    if late_horizon_vals:
        y_min = min(late_horizon_vals)
        y_top = max(late_horizon_vals)
        if y_top > y_min:
            plt.ylim(y_min - 0.05 * (y_top - y_min), y_top)
    optimizer_name_stylized = {
        "adamw": "AdamW",
        "muon": "Muon",
        "shampoo": "Shampoo",
        "psgd": "PSGD",
    }[optimizer_name]
    plt.yscale("log")
    plt.title(f"{optimizer_name_stylized} on Tiny Shakespeare Residual MLP: Loss vs Training Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_model(
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    n_layers: int,
    mlp_hidden: int,
    weight_scale_mode: ScaleMode,
    enable_weight_quantization: bool,
    lm_head_quantized: bool,
    device: torch.device,
    sr_generator: Optional[torch.Generator],
) -> ResidualMLPLM:
    qcfg = QuantForwardConfig(
        weight_scale_mode=weight_scale_mode,
        enable_weight_quantization=enable_weight_quantization,
        weight_stochastic_rounding=True,
        activation_stochastic_rounding=False,
        activation_scale_mode="row",
        sr_generator=sr_generator,
    )
    model = ResidualMLPLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        n_layers=n_layers,
        mlp_hidden=mlp_hidden,
        qcfg=qcfg,
        lm_head_quantized=lm_head_quantized,
    )
    return model.to(device)


def split_named_params(
    named_params: List[Tuple[str, nn.Parameter]],
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    matrix_params: List[nn.Parameter] = []
    embedding_params: List[nn.Parameter] = []
    lm_head_params: List[nn.Parameter] = []
    non_lm_head_params: List[nn.Parameter] = []
    aux_adamw_params: List[nn.Parameter] = []
    for name, p in named_params:
        if name.startswith("lm_head."):
            lm_head_params.append(p)
            aux_adamw_params.append(p)
            continue

        non_lm_head_params.append(p)
        if "_embedding" in name:
            embedding_params.append(p)
            aux_adamw_params.append(p)
        else:
            matrix_params.append(p)
    return matrix_params, embedding_params, lm_head_params, non_lm_head_params, aux_adamw_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ECO vs reference training on Tiny Shakespeare.")

    parser.add_argument("--data_dir", type=Path, default=Path("data/tinyshakespeare"))
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data_seed", type=int, default=2026)
    parser.add_argument("--sr_seed", type=int, default=None)

    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "both", "reference", "eco_fp8_nomaster", "fp8_nomaster_noeco"],
    )

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_hidden", type=int, default=1536)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "shampoo", "psgd"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--ns_steps", type=int, default=8)
    parser.add_argument("--ns_eps", type=float, default=1e-6)
    parser.add_argument("--ns_scale", type=float, default=1.001)
    parser.add_argument("--shampoo_root", type=float, default=4.0)
    parser.add_argument("--psgd_lr_preconditioner", type=float, default=1.0)
    parser.add_argument("--psgd_update_probability", type=float, default=1.0)

    parser.add_argument("--fp8_scale_mode", type=str, default="row", choices=["none", "tensor", "row"])
    parser.add_argument(
        "--weight_stochastic_rounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic rounding for weight quantization in FP8 runs.",
    )
    # Deprecated alias kept for backward compatibility.
    parser.add_argument(
        "--eco_update_sr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--scale_ema_decay", type=float, default=DEFAULT_SCALE_EMA_DECAY)

    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--plot_out", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.eco_update_sr is not None:
        args.weight_stochastic_rounding = args.eco_update_sr
        print("Deprecated flag --eco_update_sr used; prefer --weight_stochastic_rounding.")

    if not (0.0 < args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in (0, 1].")
    if not (0.0 <= args.scale_ema_decay < 1.0):
        raise ValueError("--scale_ema_decay must be in [0, 1).")
    if args.psgd_lr_preconditioner <= 0:
        raise ValueError("--psgd_lr_preconditioner must be > 0.")
    if not (0.0 <= args.psgd_update_probability <= 1.0):
        raise ValueError("--psgd_update_probability must be in [0, 1].")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or --device auto.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    set_seed(args.seed)
    sr_seed_base = args.seed if args.sr_seed is None else args.sr_seed

    dataset = TinyShakespeareData(args.data_dir)
    print(
        f"Dataset: vocab_size={dataset.vocab_size}, "
        f"train_tokens={dataset.train_data.numel()}, val_tokens={dataset.val_data.numel()}"
    )

    train_cfg = TrainConfig(
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        grad_clip=args.grad_clip,
    )
    print(f"LR schedule: cosine decay from {args.lr:.3e} to {args.lr * args.min_lr_ratio:.3e}")
    print(f"Optimizer family: {args.optimizer}")
    print("Rounding policy: weights=stochastic-rounding, activations=round-to-nearest")
    print(f"FP8 run policy: optimizer-side EMA scales (decay={args.scale_ema_decay:.3f}), lm_head kept in FP32")
    matrix_lr_scale = 0.2 * (max(args.d_model, args.mlp_hidden))**0.5
    if args.optimizer == "muon":
        print(
            f"Muon config: beta={args.beta1:.3f}, "
            f"ns_steps={args.ns_steps}, ns_eps={args.ns_eps:.1e}, ns_scale={args.ns_scale:.3f}"
        )
        print(f"Muon lr scale: {matrix_lr_scale:.3f}")
    elif args.optimizer == "shampoo":
        print(
            f"Shampoo config: beta={args.beta1:.3f}, precond_beta={args.beta2:.3f}, "
            f"root={args.shampoo_root:.1f}, ns_steps={args.ns_steps}, "
            f"ns_eps={args.ns_eps:.1e}, ns_scale={args.ns_scale:.3f} (matrix preconditioners)"
        )
        print(f"Shampoo lr scale: {matrix_lr_scale:.3f}")
    elif args.optimizer == "psgd":
        print(
            f"PSGD config: beta={args.beta1:.3f}, "
            f"lr_preconditioner={args.psgd_lr_preconditioner:.3f}, "
            f"update_probability={args.psgd_update_probability:.3f}"
        )
        print(f"PSGD lr scale: {matrix_lr_scale:.3f}")

    # Build one canonical initialization and reuse it for all runs.
    base_model = build_model(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        mlp_hidden=args.mlp_hidden,
        weight_scale_mode=args.fp8_scale_mode,
        enable_weight_quantization=True,
        lm_head_quantized=True,
        device=torch.device("cpu"),
        sr_generator=None,
    )
    init_state = copy.deepcopy(base_model.state_dict())
    del base_model

    if args.run == "all":
        run_order = ["reference", "eco_fp8_nomaster", "fp8_nomaster_noeco"]
    elif args.run == "both":
        run_order = ["reference", "eco_fp8_nomaster"]
    else:
        run_order = [args.run]

    all_logs: Dict[str, List[Dict[str, float]]] = {}
    summary: Dict[str, float] = {}

    for run_name in run_order:
        print(f"\n=== Starting run: {run_name} ===")
        set_seed(args.seed)
        run_seed_offset = {"reference": 0, "eco_fp8_nomaster": 10_000, "fp8_nomaster_noeco": 20_000}[run_name]
        model_sr_gen = make_torch_generator(sr_seed_base + run_seed_offset + 1, device)
        enable_weight_quantization = (run_name == "reference")

        model = build_model(
            vocab_size=dataset.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden,
            weight_scale_mode=args.fp8_scale_mode,
            enable_weight_quantization=enable_weight_quantization,
            lm_head_quantized=True,
            device=device,
            sr_generator=model_sr_gen,
        )
        model.load_state_dict(init_state)

        if run_name == "reference":
            if args.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
            elif args.optimizer == "muon":
                named_params = list(model.named_parameters())
                muon_params, _, _, _, adamw_params = split_named_params(named_params)
                muon_optimizer = MuonOptimizer(
                    muon_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    ns_steps=args.ns_steps,
                )
                adamw_optimizer = torch.optim.AdamW(
                    adamw_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                for group in muon_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                optimizer = CompositeOptimizer([adamw_optimizer, muon_optimizer])
            elif args.optimizer == "shampoo":
                named_params = list(model.named_parameters())
                shampoo_params, _, _, _, adamw_params = split_named_params(named_params)
                shampoo_optimizer = ShampooOptimizer(
                    shampoo_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    precond_beta=args.beta2,
                    root=args.shampoo_root,
                    ns_steps=args.ns_steps,
                    ns_eps=args.ns_eps,
                    ns_scale=args.ns_scale,
                )
                adamw_optimizer = torch.optim.AdamW(
                    adamw_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                for group in shampoo_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                optimizer = CompositeOptimizer([adamw_optimizer, shampoo_optimizer])
            elif args.optimizer == "psgd":
                named_params = list(model.named_parameters())
                psgd_params, _, _, _, adamw_params = split_named_params(named_params)
                psgd_optimizer = PSGDOptimizer(
                    psgd_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    lr_preconditioner=args.psgd_lr_preconditioner,
                    preconditioner_update_probability=args.psgd_update_probability,
                    precond_generator=make_torch_generator(sr_seed_base + run_seed_offset + 4, device),
                )
                adamw_optimizer = torch.optim.AdamW(
                    adamw_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                for group in psgd_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                optimizer = CompositeOptimizer([adamw_optimizer, psgd_optimizer])
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")
        elif run_name in ("eco_fp8_nomaster", "fp8_nomaster_noeco"):
            eco_comp = run_name == "eco_fp8_nomaster"

            if args.optimizer == "adamw":
                named_params = list(model.named_parameters())
                _, _, lm_head_params, non_lm_head_params, _ = split_named_params(named_params)
                fp8_optimizer = FP8AdamWNoMasterECO(
                    non_lm_head_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                )
                fp32_optimizer = torch.optim.AdamW(
                    lm_head_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, fp32_optimizer])
            elif args.optimizer == "muon":
                named_params = list(model.named_parameters())
                fp8_params, embedding_params, lm_head_params, _, _ = split_named_params(named_params)
                fp8_optimizer = FP8MuonNoMasterECO(
                    fp8_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    ns_steps=args.ns_steps,
                    ns_eps=args.ns_eps,
                    ns_scale=args.ns_scale,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                )
                for group in fp8_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                adamw_fp8_optimizer = FP8AdamWNoMasterECO(
                    embedding_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 3, device),
                )
                adamw_fp32_optimizer = torch.optim.AdamW(
                    lm_head_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, adamw_fp8_optimizer, adamw_fp32_optimizer])
            elif args.optimizer == "shampoo":
                named_params = list(model.named_parameters())
                fp8_params, embedding_params, lm_head_params, _, _ = split_named_params(named_params)
                fp8_optimizer = FP8ShampooNoMasterECO(
                    fp8_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    precond_beta=args.beta2,
                    root=args.shampoo_root,
                    ns_steps=args.ns_steps,
                    ns_eps=args.ns_eps,
                    ns_scale=args.ns_scale,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                )
                for group in fp8_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                adamw_fp8_optimizer = FP8AdamWNoMasterECO(
                    embedding_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 3, device),
                )
                adamw_fp32_optimizer = torch.optim.AdamW(
                    lm_head_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, adamw_fp8_optimizer, adamw_fp32_optimizer])
            elif args.optimizer == "psgd":
                named_params = list(model.named_parameters())
                fp8_params, embedding_params, lm_head_params, _, _ = split_named_params(named_params)
                fp8_optimizer = FP8PSGDNoMasterECO(
                    fp8_params,
                    lr=args.lr,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    lr_preconditioner=args.psgd_lr_preconditioner,
                    preconditioner_update_probability=args.psgd_update_probability,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                    precond_generator=make_torch_generator(sr_seed_base + run_seed_offset + 4, device),
                )
                for group in fp8_optimizer.param_groups:
                    group["lr_scale"] = matrix_lr_scale
                adamw_fp8_optimizer = FP8AdamWNoMasterECO(
                    embedding_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.weight_stochastic_rounding,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=args.scale_ema_decay,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 3, device),
                )
                adamw_fp32_optimizer = torch.optim.AdamW(
                    lm_head_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, adamw_fp8_optimizer, adamw_fp32_optimizer])
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")
        else:
            raise ValueError(f"Unknown run mode: {run_name}")

        logs = train_one_run(
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            device=device,
            cfg=train_cfg,
            data_seed=args.data_seed,
            peak_lr=args.lr,
            min_lr_ratio=args.min_lr_ratio,
        )
        all_logs[run_name] = logs
        summary[run_name] = logs[-1]["val_loss"]

        del model
        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Final validation losses ===")
    for run_name in run_order:
        print(f"{run_name}: {summary[run_name]:.4f}")

    if "reference" in summary and "eco_fp8_nomaster" in summary:
        delta = summary["eco_fp8_nomaster"] - summary["reference"]
        print(f"delta(eco - reference): {delta:+.4f}")
    if "reference" in summary and "fp8_nomaster_noeco" in summary:
        delta = summary["fp8_nomaster_noeco"] - summary["reference"]
        print(f"delta(noeco - reference): {delta:+.4f}")
    if "eco_fp8_nomaster" in summary and "fp8_nomaster_noeco" in summary:
        delta = summary["eco_fp8_nomaster"] - summary["fp8_nomaster_noeco"]
        print(f"delta(eco - noeco): {delta:+.4f}")

    script_dir = Path(__file__).resolve().parent
    runs_dir = script_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    plot_out = (
        args.plot_out
        if args.plot_out is not None
        else runs_dir / f"loss_plot_{args.optimizer}_{run_tag}.png"
    )
    plot_loss_curves(all_logs, plot_out, args.optimizer)
    print(f"Saved plot to {plot_out}")

    json_out = (
        args.json_out
        if args.json_out is not None
        else runs_dir / f"train_data_{args.optimizer}_{run_tag}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    payload = {
        "config": {**config, "device": str(device)},
        "train_config": asdict(train_cfg),
        "summary": summary,
        "logs": all_logs,
    }
    json_out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote logs to {json_out}")


if __name__ == "__main__":
    main()
