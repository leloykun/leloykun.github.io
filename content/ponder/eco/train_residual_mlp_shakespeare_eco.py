#!/usr/bin/env python3
"""
Minimal ECO training test on Tiny Shakespeare with a residual MLP language model.

This script compares:
1) `reference`: FP32 master weights + AdamW/Muon, but forward/backward always use
   FP8-quantized weights via STE.
2) `eco_fp8_nomaster`: FP8 materialized weights only (no master weights),
   AdamW/Muon + ECO compensation + stochastic rounding on weight re-quantization.
3) `fp8_nomaster_noeco`: FP8 materialized weights only (no master weights),
   AdamW/Muon without ECO compensation.

Both runs use FP8-quantized weights in forward and gradient computation.

Example (from repo root, in nanogpt env):
  conda run -n nanogpt python content/ponder/eco/train_residual_mlp_shakespeare_eco.py \
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


ScaleMode = Literal["none", "tensor", "row"]

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2.0 ** -6
FP8_E4M3_MIN_SUBNORMAL = 2.0 ** -9
RMS_NORM_EPS = 1e-5

# Newton-Schulz coefficients for matrix square roots (trace-normalized iteration).
COEFS_R2: List[Tuple[float, float, float]] = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375),
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


def _abc_for_r2(step: int, scale: float) -> Tuple[float, float, float]:
    a, b, c = COEFS_R2[step] if step < len(COEFS_R2) else COEFS_R2[-1]
    # For r=2: (a/scale, b/scale^3, c/scale^5).
    return a / scale, b / (scale**3), c / (scale**5)


def matrix_sqrt_ns(
    P: torch.Tensor,
    *,
    steps: int = 6,
    eps: float = 1e-6,
    scale: float = 1.01,
) -> torch.Tensor:
    # Computes P^{1/2}
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square"
    norm = torch.linalg.norm(P, dim=(-2, -1), keepdim=True)
    if norm <= eps:
        return torch.zeros_like(P)
    Y = YZ = P / norm
    I_n = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    for k in range(steps):
        a, b, c = _abc_for_r2(k, scale)
        W = a * I_n + b * YZ + c * (YZ @ YZ)
        Y, YZ = W @ Y, (W @ W) @ YZ
    return Y * norm**0.5


def _orthogonalize(M: torch.Tensor, niter: int = len(MUON_NS_COEFFS)) -> torch.Tensor:
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


def _muon_update_scale(p: torch.Tensor) -> float:
    if p.ndim != 2:
        return 1.0
    fan_out, fan_in = p.shape
    return math.sqrt(float(fan_out) / float(fan_in))


def _stochastic_round_nonnegative(
    x: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x_floor = torch.floor(x)
    frac = torch.clamp(x - x_floor, min=0.0, max=1.0)
    if generator is None:
        rand = torch.rand_like(frac)
    else:
        rand = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype, generator=generator)
    return x_floor + (rand < frac).to(dtype=x.dtype)


def _quantize_fp8_e4m3_no_scale(
    x: torch.Tensor,
    *,
    stochastic_rounding: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    FP8-E4M3 quantization for values already in-scale.
    Returns dequantized float tensor (same dtype as input).
    """
    x32 = x.float()
    out = torch.zeros_like(x32)

    finite_mask = torch.isfinite(x32)
    if finite_mask.any():
        xf = x32[finite_mask]
        sign = torch.sign(xf)
        ax = torch.clamp(xf.abs(), max=FP8_E4M3_MAX)
        q = torch.zeros_like(ax)

        normal_mask = ax >= FP8_E4M3_MIN_NORMAL
        if normal_mask.any():
            an = ax[normal_mask]
            exp = torch.clamp(torch.floor(torch.log2(an)), min=-6.0, max=7.0)
            base = torch.pow(2.0, exp)
            mant_scaled = (an / base - 1.0) * 8.0
            if stochastic_rounding:
                mant_i = _stochastic_round_nonnegative(mant_scaled, generator=generator)
            else:
                mant_i = torch.round(mant_scaled)
            mant_i = torch.clamp(mant_i, min=0.0, max=8.0)

            carry = mant_i >= 8.0
            if carry.any():
                exp = torch.clamp(exp + carry.to(exp.dtype), min=-6.0, max=7.0)
                mant_i = torch.where(carry, torch.zeros_like(mant_i), mant_i)

            q[normal_mask] = (1.0 + mant_i / 8.0) * torch.pow(2.0, exp)

        sub_mask = (ax > 0.0) & (ax < FP8_E4M3_MIN_NORMAL)
        if sub_mask.any():
            sub_scaled = ax[sub_mask] / FP8_E4M3_MIN_SUBNORMAL
            if stochastic_rounding:
                sub_i = _stochastic_round_nonnegative(sub_scaled, generator=generator)
            else:
                sub_i = torch.round(sub_scaled)
            q[sub_mask] = torch.clamp(sub_i, min=0.0, max=7.0) * FP8_E4M3_MIN_SUBNORMAL

        out[finite_mask] = sign * q

    inf_mask = torch.isinf(x32)
    if inf_mask.any():
        out[inf_mask] = torch.sign(x32[inf_mask]) * FP8_E4M3_MAX

    return out.to(dtype=x.dtype)


def _quantize_fp8_with_scale(
    x: torch.Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x32 = x.float()
    scale = compute_fp8_scale(x32, scale_mode=scale_mode)
    return quantize_fp8_e4m3_with_scale(
        x32,
        scale=scale,
        stochastic_rounding=stochastic_rounding,
        generator=generator,
    ).to(dtype=x.dtype)


def compute_fp8_scale(
    x: torch.Tensor,
    *,
    scale_mode: ScaleMode,
) -> torch.Tensor:
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
    x: torch.Tensor,
    *,
    scale: torch.Tensor,
    stochastic_rounding: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x32 = x.float()
    q_unit = _quantize_fp8_e4m3_no_scale(
        x32 / scale,
        stochastic_rounding=stochastic_rounding,
        generator=generator,
    )
    return (q_unit * scale).to(dtype=x.dtype)


def quantize_fp8_e4m3(
    x: torch.Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode = "row",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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
    x: torch.Tensor,
    *,
    stochastic_rounding: bool,
    scale_mode: ScaleMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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
    scale_mode: ScaleMode = "row"
    weight_stochastic_rounding: bool = True
    activation_stochastic_rounding: bool = False
    activation_scale_mode: ScaleMode = "tensor"
    sr_generator: Optional[torch.Generator] = None


def quantize_weight_ste(x: torch.Tensor, qcfg: QuantForwardConfig) -> torch.Tensor:
    return quantize_ste(
        x,
        stochastic_rounding=qcfg.weight_stochastic_rounding,
        scale_mode=qcfg.scale_mode,
        generator=qcfg.sr_generator,
    )


def quantize_activation_ste(x: torch.Tensor, qcfg: QuantForwardConfig) -> torch.Tensor:
    return quantize_ste(
        x,
        stochastic_rounding=qcfg.activation_stochastic_rounding,
        scale_mode=qcfg.activation_scale_mode,
        generator=qcfg.sr_generator,
    )


def norm(x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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


class FP8AdamWNoMasterECO(torch.optim.Optimizer):
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
        scale_ema_decay: float = 0.99,
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
    def _quantize_all_params(self) -> None:
        for group in self.param_groups:
            scale_mode = group["fp8_scale_mode"]
            sr = group["fp8_stochastic_rounding"]
            use_ema_scales = group["use_ema_scales"]
            for p in group["params"]:
                if use_ema_scales:
                    state = self.state[p]
                    scale = compute_fp8_scale(p.data, scale_mode=scale_mode)
                    state["fp8_scale"] = scale.detach().to(torch.float32).clone()
                    p.copy_(
                        quantize_fp8_e4m3_with_scale(
                            p.data,
                            scale=state["fp8_scale"],
                            stochastic_rounding=sr,
                            generator=self.sr_generator,
                        ).to(dtype=p.dtype)
                    )
                else:
                    p.copy_(
                        quantize_fp8_e4m3(
                            p,
                            stochastic_rounding=sr,
                            scale_mode=scale_mode,
                            generator=self.sr_generator,
                        ).to(dtype=p.dtype)
                    )

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
            scale_mode = group["fp8_scale_mode"]
            sr = group["fp8_stochastic_rounding"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]
            scale_ema_decay = group["scale_ema_decay"]

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
                    if use_ema_scales and "fp8_scale" not in state:
                        state["fp8_scale"] = compute_fp8_scale(p.data, scale_mode=scale_mode).to(torch.float32)

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
                if use_ema_scales:
                    scale_now = compute_fp8_scale(w_tilde, scale_mode=scale_mode).to(torch.float32)
                    scale_ema = state.get("fp8_scale")
                    if scale_ema is None:
                        scale_ema = scale_now.detach().clone()
                    else:
                        scale_ema = scale_ema * scale_ema_decay + scale_now * (1.0 - scale_ema_decay)
                    state["fp8_scale"] = scale_ema
                    w_q_next = quantize_fp8_e4m3_with_scale(
                        w_tilde,
                        scale=scale_ema,
                        stochastic_rounding=sr,
                        generator=self.sr_generator,
                    )
                else:
                    w_q_next = quantize_fp8_e4m3(
                        w_tilde,
                        stochastic_rounding=sr,
                        scale_mode=scale_mode,
                        generator=self.sr_generator,
                    )
                e = w_tilde - w_q_next

                if eco_comp:
                    eco_coef = ((1.0 - lr * wd) * bias_c1 * denom / lr) * (1.0 - 1.0 / beta1)
                    m_next = m_tilde + eco_coef * e
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

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g

                if p.ndim == 2:
                    update = _muon_update_scale(p) * _orthogonalize(m_tilde, niter=ns_steps)
                else:
                    update = m_tilde

                w_next = (1.0 - lr * wd) * p.detach().to(torch.float32) - lr * update
                p.copy_(w_next.to(dtype=p.dtype))
                m.copy_(m_tilde)

        return loss


class FP8MuonNoMasterECO(torch.optim.Optimizer):
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
        scale_ema_decay: float = 0.99,
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
    def _quantize_all_params(self) -> None:
        for group in self.param_groups:
            scale_mode = group["fp8_scale_mode"]
            sr = group["fp8_stochastic_rounding"]
            use_ema_scales = group["use_ema_scales"]
            for p in group["params"]:
                if use_ema_scales:
                    state = self.state[p]
                    scale = compute_fp8_scale(p.data, scale_mode=scale_mode)
                    state["fp8_scale"] = scale.detach().to(torch.float32).clone()
                    p.copy_(
                        quantize_fp8_e4m3_with_scale(
                            p.data,
                            scale=state["fp8_scale"],
                            stochastic_rounding=sr,
                            generator=self.sr_generator,
                        ).to(dtype=p.dtype)
                    )
                else:
                    p.copy_(
                        quantize_fp8_e4m3(
                            p,
                            stochastic_rounding=sr,
                            scale_mode=scale_mode,
                            generator=self.sr_generator,
                        ).to(dtype=p.dtype)
                    )

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
            scale_mode = group["fp8_scale_mode"]
            sr = group["fp8_stochastic_rounding"]
            eco_comp = group["eco_compensation"]
            use_ema_scales = group["use_ema_scales"]
            scale_ema_decay = group["scale_ema_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
                    if use_ema_scales and "fp8_scale" not in state:
                        state["fp8_scale"] = compute_fp8_scale(p.data, scale_mode=scale_mode).to(torch.float32)
                m = state["momentum"]

                g = p.grad.detach().to(torch.float32)
                m_tilde = beta * m + (1.0 - beta) * g

                if p.ndim == 2:
                    muon_scale = _muon_update_scale(p)
                    u = muon_scale * _orthogonalize(m_tilde, niter=ns_steps)
                    gram = m_tilde.mT @ m_tilde
                else:
                    muon_scale = 1.0
                    u = m_tilde
                    gram = None

                w_q = p.detach().to(torch.float32)
                w_tilde = (1.0 - lr * wd) * w_q - lr * u
                if use_ema_scales:
                    scale_now = compute_fp8_scale(w_tilde, scale_mode=scale_mode).to(torch.float32)
                    scale_ema = state.get("fp8_scale")
                    if scale_ema is None:
                        scale_ema = scale_now.detach().clone()
                    else:
                        scale_ema = scale_ema * scale_ema_decay + scale_now * (1.0 - scale_ema_decay)
                    state["fp8_scale"] = scale_ema
                    w_q_next = quantize_fp8_e4m3_with_scale(
                        w_tilde,
                        scale=scale_ema,
                        stochastic_rounding=sr,
                        generator=self.sr_generator,
                    )
                else:
                    w_q_next = quantize_fp8_e4m3(
                        w_tilde,
                        stochastic_rounding=sr,
                        scale_mode=scale_mode,
                        generator=self.sr_generator,
                    )

                e = w_tilde - w_q_next
                if eco_comp:
                    eco_coef = ((1.0 - lr * wd) / lr) * (1.0 - 1.0 / beta)
                    if p.ndim == 2 and gram is not None:
                        comp = e @ matrix_sqrt_ns(gram, steps=ns_steps, eps=ns_eps, scale=ns_scale)
                        m_next = m_tilde + (eco_coef / muon_scale) * comp
                    else:
                        m_next = m_tilde + eco_coef * e
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            param_group["lr"] = current_lr

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
        cutoff = 0.10 * float(max_step)
        for logs in all_logs.values():
            for rec in logs:
                if float(rec["step"]) >= cutoff:
                    late_horizon_vals.append(float(rec["val_loss"]))

    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    # plt.yscale("log")
    if late_horizon_vals:
        y_min, _ = plt.ylim()
        y_top = max(late_horizon_vals)
        if y_top > y_min:
            plt.ylim(None, y_top)
    optimizer_name_stylized = {
        "adamw": "AdamW",
        "muon": "Muon",
    }[optimizer_name]
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
    scale_mode: ScaleMode,
    lm_head_quantized: bool,
    device: torch.device,
    sr_generator: Optional[torch.Generator],
) -> ResidualMLPLM:
    qcfg = QuantForwardConfig(
        scale_mode=scale_mode,
        weight_stochastic_rounding=True,
        activation_stochastic_rounding=False,
        activation_scale_mode="tensor",
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
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--muon_ns_steps", type=int, default=8)

    parser.add_argument("--fp8_scale_mode", type=str, default="row", choices=["none", "tensor", "row"])
    parser.add_argument("--eco_update_sr", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--plot_out", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in (0, 1].")

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
    print("FP8 run policy: optimizer-side EMA scales (decay=0.99), lm_head kept in FP32")
    if args.optimizer == "muon":
        print(f"Muon config: beta={args.beta1:.3f}, ns_steps={args.muon_ns_steps}")
        muon_lr_scale = 0.2 * (max(args.d_model, args.mlp_hidden))**0.5
        print(f"Muon lr scale: {muon_lr_scale:.3f}")

    # Build one canonical initialization and reuse it for all runs.
    base_model = build_model(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        mlp_hidden=args.mlp_hidden,
        scale_mode=args.fp8_scale_mode,
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

        model = build_model(
            vocab_size=dataset.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden,
            scale_mode=args.fp8_scale_mode,
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
            else:
                named_params = list(model.named_parameters())
                adamw_params = [p for name, p in named_params if name.startswith("lm_head.") or "_embedding" in name]
                muon_params = [p for name, p in named_params if not name.startswith("lm_head.") and "_embedding" not in name]
                muon_optimizer = MuonOptimizer(
                    muon_params,
                    lr=args.lr * muon_lr_scale,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    ns_steps=args.muon_ns_steps,
                )
                adamw_optimizer = torch.optim.AdamW(
                    adamw_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([adamw_optimizer, muon_optimizer])
        elif run_name in ("eco_fp8_nomaster", "fp8_nomaster_noeco"):
            eco_comp = run_name == "eco_fp8_nomaster"

            if args.optimizer == "adamw":
                named_params = list(model.named_parameters())
                fp32_params = [p for name, p in named_params if name.startswith("lm_head.")]
                fp8_params = [p for name, p in named_params if not name.startswith("lm_head.")]
                fp8_optimizer = FP8AdamWNoMasterECO(
                    fp8_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.eco_update_sr,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=0.99,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                )
                fp32_optimizer = torch.optim.AdamW(
                    fp32_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, fp32_optimizer])
            else:
                named_params = list(model.named_parameters())
                embedding_params = [p for name, p in named_params if "_embedding" in name]
                fp32_params = [p for name, p in named_params if name.startswith("lm_head.") and "_embedding" not in name]
                fp8_params = [p for name, p in named_params if not name.startswith("lm_head.") and "_embedding" not in name]
                fp8_optimizer = FP8MuonNoMasterECO(
                    fp8_params,
                    lr=args.lr * muon_lr_scale,
                    beta=args.beta1,
                    weight_decay=args.weight_decay,
                    ns_steps=args.muon_ns_steps,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.eco_update_sr,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=0.99,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 2, device),
                )
                adamw_fp8_optimizer = FP8AdamWNoMasterECO(
                    embedding_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    fp8_scale_mode=args.fp8_scale_mode,
                    fp8_stochastic_rounding=args.eco_update_sr,
                    eco_compensation=eco_comp,
                    use_ema_scales=True,
                    scale_ema_decay=0.99,
                    sr_generator=make_torch_generator(sr_seed_base + run_seed_offset + 3, device),
                )
                adamw_fp32_optimizer = torch.optim.AdamW(
                    fp32_params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                )
                optimizer = CompositeOptimizer([fp8_optimizer, adamw_fp8_optimizer, adamw_fp32_optimizer])
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
    plot_out = args.plot_out if args.plot_out is not None else script_dir / "loss_vs_training_steps_shakespeare_eco.png"
    plot_loss_curves(all_logs, plot_out, args.optimizer)
    print(f"Saved plot to {plot_out}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        config = {}
        for key, value in vars(args).items():
            config[key] = str(value) if isinstance(value, Path) else value
        payload = {
            "config": {**config, "device": str(device)},
            "train_config": asdict(train_cfg),
            "summary": summary,
            "logs": all_logs,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote logs to {args.json_out}")


if __name__ == "__main__":
    main()
