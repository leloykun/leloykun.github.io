from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RMS_NORM_EPS = 1e-5
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
    )
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception:
        create_block_mask = None
    try:
        from torch.nn.attention.flex_attention import AuxRequest
    except Exception:
        AuxRequest = None
except Exception:
    AuxRequest = None
    BlockMask = None
    create_block_mask = None
    flex_attention = None

if flex_attention is not None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("This implementation requires torch.compile for FlexAttention.")
    flex_attention = torch.compile(flex_attention)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
        torch.backends.cudnn.allow_tf32 = not enabled
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not enabled
    if enabled and hasattr(torch, "set_float32_matmul_precision"):
        # Highest precision avoids backend-dependent reduced precision paths.
        torch.set_float32_matmul_precision("highest")


def make_torch_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        gen = torch.Generator(device="cuda")
    else:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


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


def _orthogonalize(M: torch.Tensor, niter: int = len(MUON_NS_COEFFS)) -> torch.Tensor:
    transpose = M.shape[0] > M.shape[1]
    if transpose:
        M = M.mT
    norm_value = torch.linalg.norm(M, dim=(-2, -1), keepdim=True)
    M = M / (norm_value + 1e-20)
    for a, b, c in MUON_NS_COEFFS[:niter]:
        M = a * M + (b * (U := M @ M.mT) + c * (U @ U)) @ M
    if transpose:
        M = M.mT
    return M


def _matrix_update_scale(p: torch.Tensor) -> float:
    assert p.ndim >= 2, "Expected weight matrix of rank 2 or higher, got shape {}".format(p.shape)
    fan_out, fan_in = p.shape[-2], p.shape[-1]
    return math.sqrt(float(fan_out) / float(fan_in))


class MuonOptimizer(torch.optim.Optimizer):
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
                if p.ndim < 2:
                    raise RuntimeError(
                        f"MuonOptimizer expects rank-2 or higher parameters; got {p.ndim} for parameter with shape {p.shape}."
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


class CompositeOptimizer:
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


def _extract_lse(aux: object) -> torch.Tensor:
    if hasattr(aux, "lse"):
        return aux.lse
    if isinstance(aux, dict) and "lse" in aux:
        return aux["lse"]
    raise RuntimeError("Could not read `lse` from FlexAttention aux output.")


def _score_mod_gelu(score, batch, head, q_idx, k_idx):
    del batch, head, q_idx, k_idx
    return torch.log1p(F.gelu(score))


@torch.no_grad()
def _batched_bincount(in_tensor: torch.Tensor, minlength: int) -> torch.Tensor:
    """
    Batched bincount over rank-2 int64 tensor.
    """
    if in_tensor.ndim != 2:
        raise ValueError("in_tensor must be rank-2.")
    if in_tensor.dtype != torch.int64:
        raise ValueError("in_tensor must be int64.")
    if minlength <= 0:
        raise ValueError("minlength must be > 0.")

    dim_0, dim_1 = in_tensor.shape
    offsets = minlength * torch.arange(dim_0, dtype=torch.int64, device=in_tensor.device)
    flat = (in_tensor + offsets.view(dim_0, 1)).reshape(dim_0 * dim_1)
    out = torch.bincount(flat, minlength=dim_0 * minlength)
    return out.view(dim_0, minlength)


@torch.no_grad()
def _get_block_mask(
    block_level_expert_assign: torch.Tensor,
    *,
    num_expert: int,
    expert_size: int,
    q_block_size: int,
    kv_block_size: int,
) -> "BlockMask":
    """
    Official-style block mask construction with full_kv_* tensors.
    """
    if BlockMask is None:
        raise RuntimeError("BlockMask is unavailable in this PyTorch build.")
    if block_level_expert_assign.ndim != 2:
        raise ValueError("block_level_expert_assign must be rank-2 (num_head, num_block_q).")
    if num_expert <= 0 or expert_size <= 0 or q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("num_expert/expert_size/q_block_size/kv_block_size must be > 0.")
    if expert_size % kv_block_size != 0:
        raise ValueError("expert_size must be divisible by kv_block_size.")

    num_head, num_block_q = block_level_expert_assign.shape
    num_block_per_expert = expert_size // kv_block_size
    num_block_k = num_expert * num_block_per_expert
    device = block_level_expert_assign.device

    block_level_expert_assign = block_level_expert_assign.to(torch.int32)

    kv_num_blocks = torch.zeros((1, num_head, num_block_q), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((1, num_head, num_block_q, num_block_k), dtype=torch.int32, device=device)
    full_kv_num_blocks = torch.full(
        (1, num_head, num_block_q),
        num_block_per_expert,
        dtype=torch.int32,
        device=device,
    )

    full_kv_indices = torch.arange(num_block_k, dtype=torch.int32, device=device)
    full_kv_indices = full_kv_indices.view(1, 1, 1, num_block_k).expand(1, num_head, 1, num_block_k)
    offsets = (num_block_per_expert * block_level_expert_assign).view(1, num_head, num_block_q, 1)
    full_kv_indices = full_kv_indices + offsets

    block_size = (q_block_size, kv_block_size)
    try:
        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=None,
        )
    except TypeError:
        if create_block_mask is None:
            raise RuntimeError("Neither BlockMask full_kv path nor create_block_mask fallback is available.")

        q_len = num_block_q * q_block_size
        kv_len = num_expert * expert_size
        block_level_expert_assign_i64 = block_level_expert_assign.to(torch.int64)

        def mask_mod(batch, head, q_idx, kv_idx):
            del batch
            block_idx = q_idx // q_block_size
            expert_id = block_level_expert_assign_i64[head, block_idx]
            return expert_id == (kv_idx // expert_size)

        return create_block_mask(
            mask_mod=mask_mod,
            B=1,
            H=num_head,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
            BLOCK_SIZE=block_size,
        )


@torch.no_grad()
def _prepare_packing(
    expert_assign: torch.Tensor,
    expert_bincount: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Prepare deterministic packing metadata.
    """
    if expert_assign.ndim != 2 or expert_assign.dtype != torch.int64:
        raise ValueError("expert_assign must be rank-2 int64.")
    if expert_bincount.ndim != 2 or expert_bincount.dtype != torch.int64:
        raise ValueError("expert_bincount must be rank-2 int64.")
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")

    num_head, pool_size = expert_assign.shape
    if expert_bincount.shape[0] != num_head:
        raise ValueError("expert_bincount first dim must match expert_assign.")
    num_expert = expert_bincount.size(1)
    device = expert_assign.device

    padding_needed = (-expert_bincount) % block_size
    padding_size_all = padding_needed.sum(dim=1, keepdim=False)
    padding_size = int(padding_size_all.max().item())
    if padding_size < 0:
        raise RuntimeError("padding_size must be non-negative.")

    padding_needed = padding_needed.clone()
    padding_needed[:, -1] += padding_size - padding_size_all

    offsets = num_expert * torch.arange(num_head, device=device, dtype=torch.int64).view(num_head, 1)
    expert_assign = (expert_assign + offsets).view(num_head * pool_size)

    padding_tensor = torch.arange(num_head * num_expert, device=device, dtype=torch.int64)
    padding_tensor = torch.repeat_interleave(padding_tensor, padding_needed.view(num_head * num_expert))

    expert_assign = torch.concat((expert_assign, padding_tensor), dim=0)
    expert_assign, mapping = torch.sort(expert_assign, stable=False)

    expert_assign = expert_assign.view(num_head, pool_size + padding_size)
    expert_assign = expert_assign - offsets

    mapping_inv = torch.empty_like(mapping)
    mapping_inv[mapping] = torch.arange(mapping.size(0), device=device, dtype=torch.int64)

    num_block_q = (pool_size + padding_size) // block_size
    block_level_expert_assign = expert_assign.view(num_head, num_block_q, block_size)[..., 0]
    block_level_expert_assign = block_level_expert_assign.contiguous()

    return mapping, mapping_inv, padding_size, block_level_expert_assign, expert_assign


def _packing(input_tensor: torch.Tensor, padding_size: int, mapping: torch.Tensor) -> torch.Tensor:
    """
    Apply packing permutation after zero-padding tail rows per head.
    """
    if input_tensor.ndim != 3:
        raise ValueError("input_tensor must be rank-3 (num_head, pool_size, emb_size).")
    if mapping.ndim != 1 or mapping.dtype != torch.int64:
        raise ValueError("mapping must be rank-1 int64.")
    if padding_size < 0:
        raise ValueError("padding_size must be >= 0.")

    num_head, pool_size, emb_size = input_tensor.shape
    flat = input_tensor.view(num_head * pool_size, emb_size)
    if padding_size > 0:
        padding_tensor = torch.zeros(
            (num_head * padding_size, emb_size),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        flat = torch.concat((flat, padding_tensor), dim=0)
    flat = flat[mapping]
    return flat.view(num_head, pool_size + padding_size, emb_size)


def _unpacking(input_tensor: torch.Tensor, padding_size: int, mapping_inv: torch.Tensor) -> torch.Tensor:
    """
    Inverse of _packing.
    """
    if input_tensor.ndim != 3:
        raise ValueError("input_tensor must be rank-3 (num_head, pool_size+padding_size, emb_size).")
    if mapping_inv.ndim != 1 or mapping_inv.dtype != torch.int64:
        raise ValueError("mapping_inv must be rank-1 int64.")
    if padding_size < 0:
        raise ValueError("padding_size must be >= 0.")

    num_head = input_tensor.size(0)
    packed_pool_size = input_tensor.size(1)
    emb_size = input_tensor.size(2)
    pool_size = packed_pool_size - padding_size
    if pool_size < 0:
        raise ValueError("padding_size is larger than packed sequence length.")

    flat = input_tensor.view(num_head * packed_pool_size, emb_size)
    flat = flat[mapping_inv]
    flat = flat[: num_head * pool_size]
    return flat.view(num_head, pool_size, emb_size)


def flex_expert_ffn_assignments_directmask_multihead(
    q_by_head: torch.Tensor,
    expert_ids_by_head: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    *,
    kv_block: int = 128,
    q_block: int = 1,
) -> torch.Tensor:
    """
    Multi-head FlexAttention path implementing exact expert FFN for assignment rows:
      y[h, i] = GELU(q[h, i] @ W1[h, e]) @ W2[h, e]

    Weight layouts (swapped for consistency):
      W1: (H, E, m, d)
      W2: (H, E, d, m)
    """
    if flex_attention is None:
        raise RuntimeError("FlexAttention is not available in this PyTorch build.")
    if not q_by_head.is_cuda:
        raise RuntimeError("FlexAttention path requires CUDA tensors.")
    if q_by_head.ndim != 3 or expert_ids_by_head.ndim != 2:
        raise ValueError("q_by_head must be (H, Q_LEN, d) and expert_ids_by_head must be (H, Q_LEN).")
    if W1.ndim != 4 or W2.ndim != 4:
        raise ValueError("W1/W2 must be rank-4 tensors for multi-head mode.")

    H, q_len, d = q_by_head.shape
    if expert_ids_by_head.shape != (H, q_len):
        raise ValueError("expert_ids_by_head shape must match q_by_head's first two dims.")

    Hw1, E, m, dw1 = W1.shape
    Hw2, Ew2, dw2, mw2 = W2.shape
    if Hw1 != H or Hw2 != H:
        raise ValueError("W1/W2 head count must match q_by_head.")
    if E != Ew2 or dw1 != d or mw2 != m or dw2 != d:
        raise ValueError("W1/W2 dimensions are inconsistent.")
    if not _is_pow2(d):
        raise ValueError(f"FlexAttention requires power-of-2 head dim; got d={d}.")

    dtype = q_by_head.dtype
    m_pad = _ceil_div(m, kv_block) * kv_block
    pad = m_pad - m

    # Use block-level expert assignments (official-style). For q_block=1, keep
    # a compile-friendly query block size.
    q_block_for_mask = q_block if q_block > 1 else 128
    expert_ids_by_head = expert_ids_by_head.to(torch.int64).contiguous()
    expert_bincount = _batched_bincount(
        expert_ids_by_head,
        minlength=E,
    )
    mapping, mapping_inv, padding_size, block_expert_ids, expert_ids_packed = _prepare_packing(
        expert_assign=expert_ids_by_head,
        expert_bincount=expert_bincount,
        block_size=q_block_for_mask,
    )
    q_packed = _packing(q_by_head, padding_size=padding_size, mapping=mapping)
    q_len_packed = int(q_packed.shape[1])

    # KV layout per head: [expert0 neurons][expert1 neurons]...[expertE-1 neurons]
    K = W1.contiguous()  # (H, E, m, d)
    V = W2.transpose(-1, -2).contiguous()  # (H, E, m, d)
    if pad:
        K = F.pad(K, (0, 0, 0, pad))
        V = F.pad(V, (0, 0, 0, pad))

    K_all = K.view(H, E * m_pad, d)
    V_all = V.view(H, E * m_pad, d)

    bias_table = V.sum(dim=2)  # (H, E, d)
    gather_index = expert_ids_packed.to(torch.int64).unsqueeze(-1).expand(H, q_len_packed, d)
    bias_packed = torch.gather(bias_table, dim=1, index=gather_index)  # (H, Q_LEN_PAD, d)

    block_mask = _get_block_mask(
        block_level_expert_assign=block_expert_ids,
        num_expert=E,
        expert_size=m_pad,
        q_block_size=q_block_for_mask,
        kv_block_size=kv_block,
    )
    kv_len = E * m_pad

    q = q_packed.unsqueeze(0)  # (1, H, Q_LEN_PAD, d)
    k = K_all.unsqueeze(0)  # (1, H, KV_LEN, d)
    v = V_all.unsqueeze(0)  # (1, H, KV_LEN, d)

    if AuxRequest is not None:
        out, aux = flex_attention(
            q,
            k,
            v,
            score_mod=_score_mod_gelu,
            block_mask=block_mask,
            scale=1.0,
            return_aux=AuxRequest(lse=True),
        )
        lse = _extract_lse(aux)
    else:
        out, lse = flex_attention(
            q,
            k,
            v,
            score_mod=_score_mod_gelu,
            block_mask=block_mask,
            scale=1.0,
            return_lse=True,
        )

    denom = torch.exp(lse)
    if denom.ndim == out.ndim - 1:
        denom = denom.unsqueeze(-1)
    unnorm = out * denom
    y_packed = unnorm[0] - bias_packed
    y = _unpacking(y_packed, padding_size=padding_size, mapping_inv=mapping_inv)
    if y.shape != (H, q_len, d):
        raise RuntimeError("Unpacking produced an unexpected shape.")
    return y.to(dtype=dtype)


def norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_NORM_EPS)


@dataclass
class MoEConfig:
    d_model: int
    moe_heads: int
    d_moe_latent: int
    num_experts: int
    top_k: int
    expert_hidden: int
    lucid_router_eps: float = 1e-4
    enable_lucid_router_default: bool = False
    enable_sigmoid_gating_default: bool = False
    enable_qe_norm_default: bool = False
    enable_auxfree_bias_default: bool = True
    auxfree_bias_lr: float = 1e-2
    auxfree_bias_clip: float = 10.0
    kv_block_size: int = 128


class MultiHeadLatentMoE(nn.Module):
    """
    Multi-head Latent MoE FFN layer with optional LUCID correction.
    """

    def __init__(
        self,
        cfg: MoEConfig,
        *,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if cfg.d_moe_latent <= 0:
            raise ValueError("d_moe_latent must be > 0.")
        if not (1 <= cfg.top_k <= cfg.num_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        if cfg.kv_block_size <= 0:
            raise ValueError("kv_block_size must be > 0.")

        self.cfg = cfg
        self.d_moe_latent = cfg.d_moe_latent
        if self.d_moe_latent < 16:
            raise ValueError(
                f"Compiled FlexAttention requires per-head dim >= 16; got d_moe_latent={self.d_moe_latent}."
            )

        H, E, d, m = cfg.moe_heads, cfg.num_experts, self.d_moe_latent, cfg.expert_hidden

        self.in_proj = nn.Parameter(torch.empty(H, d, cfg.d_model))
        self.out_proj = nn.Linear(H * d, cfg.d_model, bias=False)
        self.router_embedding = nn.Parameter(torch.empty(H, E, d))
        self.register_buffer("auxfree_bias", torch.zeros(H, E, dtype=torch.float32), persistent=True)

        # Store expert FFN weights as (fan_out, fan_in)-style per expert:
        # W1: (H, E, m, d), W2: (H, E, d, m)
        self.W1 = nn.Parameter(torch.empty(H, E, m, d))
        self.W2 = nn.Parameter(torch.empty(H, E, d, m))
        self.last_routing_entropy: Optional[float] = None
        self.last_max_load_violation: Optional[float] = None
        self.reset_parameters(init_generator=init_generator)

    def reset_parameters(self, *, init_generator: Optional[torch.Generator] = None) -> None:
        nn.init.kaiming_uniform_(self.in_proj, a=math.sqrt(5), generator=init_generator)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5), generator=init_generator)
        nn.init.kaiming_uniform_(self.router_embedding, a=math.sqrt(5), generator=init_generator)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5), generator=init_generator)
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5), generator=init_generator)
        with torch.no_grad():
            self.in_proj.data = _matrix_update_scale(self.in_proj.data) * _orthogonalize(self.in_proj.data)
            self.out_proj.weight.data = _matrix_update_scale(self.out_proj.weight.data) * _orthogonalize(self.out_proj.weight.data)
            self.W1.data = _matrix_update_scale(self.W1.data) * _orthogonalize(self.W1.data)
            self.W2.data = _matrix_update_scale(self.W2.data) * _orthogonalize(self.W2.data)
            if self.cfg.enable_qe_norm_default:
                self.router_embedding.data = norm(self.router_embedding.data)
            self.auxfree_bias.zero_()

    def _expert_ffn_assignments(
        self,
        q_assign_stacked: torch.Tensor,
        expert_ids_stacked: torch.Tensor,
    ) -> torch.Tensor:
        if q_assign_stacked.ndim != 3 or expert_ids_stacked.ndim != 2:
            raise ValueError("q_assign_stacked must be (H, Nassign, d) and expert_ids_stacked must be (H, Nassign).")
        if q_assign_stacked.shape[:2] != expert_ids_stacked.shape:
            raise ValueError("q_assign_stacked and expert_ids_stacked shapes are inconsistent.")

        return flex_expert_ffn_assignments_directmask_multihead(
            q_by_head=q_assign_stacked,
            expert_ids_by_head=expert_ids_stacked,
            W1=self.W1,
            W2=self.W2,
            kv_block=self.cfg.kv_block_size,
            q_block=1,
        )

    def _remove_auxfree_bias(
        self,
        top_logit: torch.Tensor,
        top_idx: torch.Tensor,
    ) -> torch.Tensor:
        if top_logit.ndim != 3 or top_idx.ndim != 3:
            raise ValueError("top_logit/top_idx must be rank-3 (H, Ntok, K).")
        if top_logit.shape != top_idx.shape:
            raise ValueError("top_logit and top_idx shapes must match.")
        H, Ntok, k = top_logit.shape
        selected_bias = torch.gather(
            self.auxfree_bias.to(dtype=top_logit.dtype),
            dim=1,
            index=top_idx.reshape(H, Ntok * k),
        ).reshape(H, Ntok, k)
        return top_logit - selected_bias

    def _update_auxfree_bias(self, topk_idx: torch.Tensor) -> None:
        if not self.cfg.enable_auxfree_bias_default or not self.training or not torch.is_grad_enabled():
            return
        if topk_idx.ndim != 3:
            raise ValueError("topk_idx must be rank-3 (H, Ntok, K).")

        with torch.no_grad():
            H, Ntok, k = topk_idx.shape
            E = self.cfg.num_experts
            if Ntok == 0 or k == 0:
                return

            expert_assign = topk_idx.reshape(H, Ntok * k).to(torch.int64)
            expert_bincount = _batched_bincount(expert_assign, minlength=E)
            load = expert_bincount.to(torch.float32) / float(Ntok * k)

            target = 1.0 / float(E)
            self.auxfree_bias.add_(self.cfg.auxfree_bias_lr * torch.sign(target - load))
            # Keep only per-expert relative offsets for each head.
            self.auxfree_bias.sub_(self.auxfree_bias.mean(dim=1, keepdim=True))

            clip = float(self.cfg.auxfree_bias_clip)
            if clip > 0.0:
                self.auxfree_bias.clamp_(min=-clip, max=clip)

    def _compute_block_expert_load(self, topk_idx: torch.Tensor) -> torch.Tensor:
        if topk_idx.ndim != 3:
            raise ValueError("topk_idx must be rank-3 (H, Ntok, K).")
        H, Ntok, k = topk_idx.shape
        E = self.cfg.num_experts
        if Ntok == 0 or k == 0:
            return torch.zeros((E,), dtype=torch.float32, device=topk_idx.device)

        expert_assign = topk_idx.reshape(H * Ntok * k).to(torch.int64)
        expert_bincount = torch.bincount(expert_assign, minlength=E)
        return expert_bincount.to(torch.float32) / float(H * Ntok * k)

    def _forward_impl(
        self,
        x: torch.Tensor,
        *,
        enable_lucid_router: Optional[bool],
        enable_sigmoid_gating: Optional[bool],
        enable_qe_norm: Optional[bool],
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must be rank-3 (B, T, d_model).")

        if enable_lucid_router is None:
            enable_lucid_router = self.cfg.enable_lucid_router_default
        if enable_sigmoid_gating is None:
            enable_sigmoid_gating = self.cfg.enable_sigmoid_gating_default
        if enable_qe_norm is None:
            enable_qe_norm = self.cfg.enable_qe_norm_default

        B, T, _ = x.shape
        H, d, k = self.cfg.moe_heads, self.d_moe_latent, self.cfg.top_k
        Ntok = B * T

        # (Ntok, H, d) -> (H, Ntok, d)
        q_by_head = torch.einsum("hdm,btm->hbtd", self.in_proj, x).reshape(H, Ntok, d).contiguous()
        if enable_qe_norm:
            q_router = norm(q_by_head)
            router_embedding = norm(self.router_embedding)
            logits_factor = 1.0 / math.sqrt(d)
        else:
            q_router = q_by_head
            router_embedding = self.router_embedding
            logits_factor = 1.0

        # Router logits: (H, Ntok, E)
        logits = logits_factor * torch.einsum("hnd,hed->hne", q_router, router_embedding)
        if self.cfg.enable_auxfree_bias_default:
            dirty_logits = logits + self.auxfree_bias.to(dtype=logits.dtype).unsqueeze(1)
            topk_dirty_values, topk_idx = torch.topk(dirty_logits, k=k, dim=-1)
            topk_values = self._remove_auxfree_bias(topk_dirty_values, topk_idx)
        else:
            topk_values, topk_idx = torch.topk(logits, k=k, dim=-1)
        if enable_sigmoid_gating:
            gate = torch.sigmoid(topk_values)
            alpha = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            alpha = torch.softmax(topk_values, dim=-1)  # (H, Ntok, k)
        self._update_auxfree_bias(topk_idx)
        block_expert_load = self._compute_block_expert_load(topk_idx)
        mean_load = block_expert_load.mean()
        max_load = block_expert_load.max()
        load_violation = (max_load - mean_load) / mean_load.clamp_min(1e-12)
        self.last_max_load_violation = float(load_violation.item())

        # Expand assignments without repeat_interleave.
        q_assign_stacked = norm(q_router).unsqueeze(2).expand(H, Ntok, k, d).reshape(H, Ntok * k, d)
        expert_ids_stacked = topk_idx.reshape(H, Ntok * k).to(torch.int64)

        y_assign_stacked = self._expert_ffn_assignments(
            q_assign_stacked=q_assign_stacked,
            expert_ids_stacked=expert_ids_stacked,
        ).view(H, Ntok, k, d)

        if not enable_lucid_router or k <= 1:
            mixed_by_head = (alpha.unsqueeze(-1) * y_assign_stacked).sum(dim=2)
            beta_for_entropy = alpha
        else:
            # Gather selected router embeddings for each (head, token, top-k).
            h_idx = torch.arange(H, device=x.device, dtype=torch.int64)[:, None, None].expand(H, Ntok, k)
            key_embed = norm(router_embedding[h_idx, topk_idx])  # (H, Ntok, k, d); norm(*) is idempotent
            sim = key_embed @ key_embed.transpose(-1, -2)
            if enable_sigmoid_gating:
                P = 2.0 * torch.sigmoid(sim.to(torch.float32) / math.sqrt(d) - math.sqrt(d))
            else:
                P = torch.exp(sim.to(torch.float32) / math.sqrt(d) - math.sqrt(d))
            eye = torch.eye(k, device=P.device, dtype=P.dtype).view(1, 1, k, k)
            P = P + self.cfg.lucid_router_eps * eye

            beta_for_entropy = torch.linalg.solve(P, alpha.to(torch.float32).unsqueeze(-1)).squeeze(-1)
            y_assign_corr = torch.linalg.solve(P, y_assign_stacked.to(torch.float32))
            mixed_by_head = (alpha.to(torch.float32).unsqueeze(-1) * y_assign_corr).sum(dim=2)
            mixed_by_head = mixed_by_head.to(dtype=y_assign_stacked.dtype)

        beta_mass = (beta_for_entropy.to(torch.float32))**2
        beta_dist = beta_mass / beta_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(beta_dist * beta_dist.clamp_min(1e-12).log()).sum(dim=-1)
        self.last_routing_entropy = float(entropy.mean().item())

        y = mixed_by_head.permute(1, 0, 2).contiguous().view(B, T, H * d)
        return self.out_proj(y)

    def forward(
        self,
        x: torch.Tensor,
        *,
        enable_lucid_router: Optional[bool] = None,
        enable_sigmoid_gating: Optional[bool] = None,
        enable_qe_norm: Optional[bool] = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            x,
            enable_lucid_router=enable_lucid_router,
            enable_sigmoid_gating=enable_sigmoid_gating,
            enable_qe_norm=enable_qe_norm,
        )


class RoPE(nn.Module):
    def __init__(self, head_dim: int, *, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        self.head_dim = head_dim
        self.base = base
        self._cache: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(
        self,
        *,
        device: torch.device,
        seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, seqlen)
        if key in self._cache:
            return self._cache[key]

        inv_freq = self.inv_freq.to(device=device)
        positions = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(positions, inv_freq)  # (T, Dh/2)
        cos = torch.cos(freqs).to(dtype=torch.float32).view(1, 1, seqlen, self.head_dim // 2)
        sin = torch.sin(freqs).to(dtype=torch.float32).view(1, 1, seqlen, self.head_dim // 2)
        self._cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        x_even = x_fp32[..., ::2]
        x_odd = x_fp32[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x_fp32)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out.to(dtype=x_dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("RoPE input last dim must equal head_dim.")
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("RoPE expects q and k to have the same sequence length.")

        seqlen = q.shape[-2]
        cos, sin = self._get_cos_sin(device=q.device, seqlen=seqlen)
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        block_size: int = 128,
        rope_base: float = 10000.0,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads in CausalSelfAttention.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.qkv_proj = nn.Parameter(torch.empty(3, num_heads, self.head_dim, d_model))
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.rope = RoPE(self.head_dim, base=rope_base)
        self._block_mask_cache: Dict[Tuple[torch.device, int, int], "BlockMask"] = {}
        self.reset_parameters(init_generator=init_generator)

    def reset_parameters(self, *, init_generator: Optional[torch.Generator] = None) -> None:
        nn.init.kaiming_uniform_(self.qkv_proj, a=math.sqrt(5), generator=init_generator)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5), generator=init_generator)
        with torch.no_grad():
            self.qkv_proj.data = _matrix_update_scale(self.qkv_proj.data) * _orthogonalize(self.qkv_proj.data)
            self.out_proj.weight.data = _matrix_update_scale(self.out_proj.weight.data) * _orthogonalize(self.out_proj.weight.data)

    def _get_causal_block_mask(
        self,
        *,
        device: torch.device,
        batch_size: int,
        seqlen: int,
    ) -> "BlockMask":
        if create_block_mask is None:
            raise RuntimeError("create_block_mask is unavailable in this PyTorch build.")

        key = (device, batch_size, seqlen)
        if key in self._block_mask_cache:
            return self._block_mask_cache[key]

        def causal_mask(batch, head, q_idx, kv_idx):
            del batch, head
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            mask_mod=causal_mask,
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=device,
            BLOCK_SIZE=(self.block_size, self.block_size),
        )
        self._block_mask_cache[key] = block_mask
        return block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = torch.einsum("ahdm,btm->abhtd", self.qkv_proj, x)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK norm
        q, k = norm(q), norm(k)
        q, k = self.rope(q, k)

        if flex_attention is None:
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build.")
        if not q.is_cuda:
            raise RuntimeError("FlexAttention path requires CUDA tensors.")
        block_mask = self._get_causal_block_mask(
            device=q.device,
            batch_size=B,
            seqlen=T,
        )
        y = flex_attention(q, k, v, block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        attn_heads: int,
        moe_cfg: MoEConfig,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.alpha = 1.0 / (2 * float(n_layers))
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=attn_heads,
            block_size=moe_cfg.kv_block_size,
            init_generator=init_generator,
        )
        self.moe = MultiHeadLatentMoE(
            moe_cfg,
            init_generator=init_generator,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        enable_lucid_router: Optional[bool],
        enable_sigmoid_gating: Optional[bool],
    ) -> torch.Tensor:
        x = (1. - self.alpha) * x + self.alpha * self.self_attn(norm(x))
        x = (1. - self.alpha) * x + self.alpha * self.moe(
            norm(x),
            enable_lucid_router=enable_lucid_router,
            enable_sigmoid_gating=enable_sigmoid_gating,
        )
        return x


class LatentMoEShakespeareLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        attn_heads: int,
        moe_cfg: MoEConfig,
        init_generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if moe_cfg.d_model != d_model:
            raise ValueError("moe_cfg.d_model must match model d_model.")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    d_model=d_model,
                    n_layers=n_layers,
                    attn_heads=attn_heads,
                    moe_cfg=moe_cfg,
                    init_generator=init_generator,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.last_routing_entropy: Optional[float] = None
        self.last_max_load_violation: Optional[float] = None
        self.reset_parameters(init_generator=init_generator)

    def reset_parameters(self, *, init_generator: Optional[torch.Generator] = None) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02, generator=init_generator)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02, generator=init_generator)
        with torch.no_grad():
            self.token_embedding.weight.data = norm(self.token_embedding.weight.data)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        enable_lucid_router: Optional[bool] = None,
        enable_sigmoid_gating: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = idx.shape
        if seqlen > self.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.block_size}.")

        x = norm(self.token_embedding(idx))

        block_entropies: List[float] = []
        block_load_violations: List[float] = []
        for block in self.blocks:
            x = block(
                x,
                enable_lucid_router=enable_lucid_router,
                enable_sigmoid_gating=enable_sigmoid_gating,
            )
            if block.moe.last_routing_entropy is not None:
                block_entropies.append(block.moe.last_routing_entropy)
            if block.moe.last_max_load_violation is not None:
                block_load_violations.append(block.moe.last_max_load_violation)

        self.last_routing_entropy = (
            float(sum(block_entropies) / len(block_entropies)) if block_entropies else None
        )
        self.last_max_load_violation = (
            float(sum(block_load_violations) / len(block_load_violations))
            if block_load_violations
            else None
        )

        logits = self.lm_head(norm(x)).float()

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


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


def constant_then_decay_lr_with_floor(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
    decay_fraction: float = 0.2,
) -> float:
    if total_steps <= 1:
        return peak_lr
    if not (0.0 < decay_fraction <= 1.0):
        raise ValueError("decay_fraction must be in (0, 1].")

    min_lr = peak_lr * min_lr_ratio
    decay_steps = max(1, int(math.ceil(float(total_steps) * decay_fraction)))
    decay_start = max(0, total_steps - decay_steps)

    if step < decay_start:
        return peak_lr
    if decay_steps == 1:
        return min_lr

    progress = float(step - decay_start) / float(decay_steps - 1)
    progress = max(0.0, min(1.0, progress))
    return peak_lr + (min_lr - peak_lr) * progress


def _get_group_lrs(optimizer: torch.optim.Optimizer | CompositeOptimizer) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("group_name", f"group_{idx}"))
        out[group_name] = float(group["lr"])
    return out


def _set_scheduled_lrs(
    optimizer: torch.optim.Optimizer | CompositeOptimizer,
    *,
    step: int,
    total_steps: int,
    min_lr_ratio: float,
) -> None:
    for group in optimizer.param_groups:
        peak_lr = float(group.get("lr_peak", group["lr"]))
        group["lr_peak"] = peak_lr
        group["lr"] = constant_then_decay_lr_with_floor(
            step,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr_ratio=min_lr_ratio,
        )


@torch.no_grad()
def evaluate(
    model: LatentMoEShakespeareLM,
    dataset: TinyShakespeareData,
    *,
    device: torch.device,
    cfg: TrainConfig,
    rng: torch.Generator,
    enable_lucid_router: bool,
    enable_sigmoid_gating: bool,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split in ("train", "val"):
        losses: List[float] = []
        entropies: List[float] = []
        load_violations: List[float] = []
        for _ in range(cfg.eval_batches):
            xb, yb = dataset.get_batch(
                split,  # type: ignore[arg-type]
                batch_size=cfg.batch_size,
                block_size=cfg.block_size,
                device=device,
                generator=rng,
            )
            _, loss = model(
                xb,
                yb,
                enable_lucid_router=enable_lucid_router,
                enable_sigmoid_gating=enable_sigmoid_gating,
            )
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")
            losses.append(float(loss.item()))
            if model.last_routing_entropy is not None:
                entropies.append(model.last_routing_entropy)
            if model.last_max_load_violation is not None:
                load_violations.append(model.last_max_load_violation)
        out[f"{split}_loss"] = sum(losses) / len(losses)
        out[f"{split}_routing_entropy"] = (
            sum(entropies) / len(entropies) if entropies else float("nan")
        )
        out[f"{split}_max_load_violation"] = (
            sum(load_violations) / len(load_violations) if load_violations else float("nan")
        )
    model.train()
    return out


def train_one_run(
    *,
    run_name: str,
    model: LatentMoEShakespeareLM,
    optimizer: torch.optim.Optimizer | CompositeOptimizer,
    dataset: TinyShakespeareData,
    device: torch.device,
    cfg: TrainConfig,
    data_seed: int,
    min_lr_ratio: float,
    enable_lucid_router: bool,
    enable_sigmoid_gating: bool,
    logs: List[Dict[str, float]],
    on_eval: Optional[Callable[[], None]] = None,
) -> List[Dict[str, float]]:
    train_rng = make_torch_generator(data_seed, torch.device("cpu"))
    eval_rng = make_torch_generator(data_seed + 10_000, torch.device("cpu"))

    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            stats = evaluate(
                model,
                dataset,
                device=device,
                cfg=cfg,
                rng=eval_rng,
                enable_lucid_router=enable_lucid_router,
                enable_sigmoid_gating=enable_sigmoid_gating,
            )
            elapsed = time.time() - t0
            group_lrs = _get_group_lrs(optimizer)
            current_lr = group_lrs.get("token_embedding_adamw", float(next(iter(group_lrs.values()))))
            rec = {
                "step": float(step),
                "train_loss": stats["train_loss"],
                "val_loss": stats["val_loss"],
                "train_routing_entropy": stats["train_routing_entropy"],
                "val_routing_entropy": stats["val_routing_entropy"],
                "train_max_load_violation": stats["train_max_load_violation"],
                "val_max_load_violation": stats["val_max_load_violation"],
                "lr": current_lr,
                "lr_embedding": group_lrs.get("token_embedding_adamw", float("nan")),
                "lr_token_embedding": group_lrs.get("token_embedding_adamw", float("nan")),
                "lr_router_embedding": group_lrs.get("router_embedding_adamw", float("nan")),
                "lr_linear": group_lrs.get("linear_muon", float("nan")),
                "lr_lm_head": group_lrs.get("lm_head_muon", float("nan")),
                "elapsed_sec": elapsed,
            }
            logs.append(rec)
            print(
                f"[{run_name}] step={step:5d} "
                f"train_loss={stats['train_loss']:.4f} val_loss={stats['val_loss']:.4f} "
                f"train_entropy={stats['train_routing_entropy']:.4f} "
                f"val_entropy={stats['val_routing_entropy']:.4f} "
                f"train_load_violation={stats['train_max_load_violation']:.4f} "
                f"val_load_violation={stats['val_max_load_violation']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            if on_eval is not None:
                on_eval()

        if step == cfg.steps:
            break

        xb, yb = dataset.get_batch(
            "train",
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            device=device,
            generator=train_rng,
        )

        _set_scheduled_lrs(
            optimizer,
            step=step,
            total_steps=cfg.steps,
            min_lr_ratio=min_lr_ratio,
        )

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(
            xb,
            yb,
            enable_lucid_router=enable_lucid_router,
            enable_sigmoid_gating=enable_sigmoid_gating,
        )
        if loss is None:
            raise RuntimeError("Loss should not be None in training.")
        loss.backward()
        optimizer.step()

    return logs


def _run_label(run_name: str) -> str:
    return {
        "mh_lmoe_lucid_off": "MH-LatentMoE (no LUCID-Routers)",
        "mh_lmoe_lucid_on": "MH-LatentMoE (with LUCID-Routers)",
    }.get(run_name, run_name.replace("_", " "))


def _collect_plot_window_values(
    *,
    logs_by_run: Dict[str, List[Dict[str, float]]],
    key: str,
    require_positive: bool = False,
) -> Tuple[float, float, List[float]]:
    non_empty_logs = [run_logs for run_logs in logs_by_run.values() if run_logs]
    if not non_empty_logs:
        return 0.0, 0.0, []

    max_step = max(float(rec["step"]) for run_logs in non_empty_logs for rec in run_logs)
    cut_step = 0.5 * max_step
    remaining: List[float] = []
    all_values: List[float] = []

    for run_logs in non_empty_logs:
        for rec in run_logs:
            value = float(rec.get(key, float("nan")))
            if not math.isfinite(value):
                continue
            if require_positive and value <= 0.0:
                continue
            all_values.append(value)
            if float(rec["step"]) >= cut_step:
                remaining.append(value)

    y_values = remaining if remaining else all_values
    return max_step, cut_step, y_values


def _apply_plot_y_limits(
    *,
    plt_mod,
    y_values: List[float],
    clamp_zero_min: bool,
) -> None:
    if not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    if not (math.isfinite(y_min) and math.isfinite(y_max)):
        return

    if y_max > y_min:
        y_span = y_max - y_min
        y_low = y_min - 0.05 * y_span
        if clamp_zero_min:
            y_low = max(0.0, y_low)
        plt_mod.ylim(bottom=y_low, top=y_max)
    else:
        wiggle = 0.05 * max(1.0, abs(y_max))
        y_low = y_min - wiggle
        if clamp_zero_min:
            y_low = max(0.0, y_low)
        plt_mod.ylim(bottom=y_low, top=y_max + wiggle)


def plot_loss_curves(
    *,
    logs_by_run: Dict[str, List[Dict[str, float]]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "--plot_losses was requested but matplotlib is unavailable in this environment."
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(run_logs for run_logs in logs_by_run.values()):
        raise ValueError("plot_loss_curves received no logs.")

    plt.figure(figsize=(10.0, 4.0))
    max_step, cut_step, y_values = _collect_plot_window_values(
        logs_by_run=logs_by_run,
        key="val_loss",
        require_positive=True,
    )
    for run_name, run_logs in logs_by_run.items():
        if not run_logs:
            continue
        steps = [rec["step"] for rec in run_logs]
        val_losses = [rec["val_loss"] for rec in run_logs]
        plt.plot(steps, val_losses, linestyle="-", linewidth=2.2, label=_run_label(run_name))

    plt.yscale("log")
    _apply_plot_y_limits(plt_mod=plt, y_values=y_values, clamp_zero_min=True)
    if max_step > cut_step:
        plt.xlim(left=cut_step, right=max_step)

    plt.xlabel("Step")
    plt.ylabel("Validation Loss (log scale)")
    plt.title(
        "LUCID-MoE: Mixture-of-Experts with Preconditioned Routing\n"
        "Multi-Head LatentMoE (base) on Tiny Shakespeare"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_routing_entropy_curves(
    *,
    logs_by_run: Dict[str, List[Dict[str, float]]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "--plot_losses was requested but matplotlib is unavailable in this environment."
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(run_logs for run_logs in logs_by_run.values()):
        raise ValueError("plot_routing_entropy_curves received no logs.")

    plt.figure(figsize=(10.0, 4.0))
    max_step, cut_step, y_values = _collect_plot_window_values(
        logs_by_run=logs_by_run,
        key="val_routing_entropy",
        require_positive=False,
    )
    for run_name, run_logs in logs_by_run.items():
        if not run_logs:
            continue
        steps = [rec["step"] for rec in run_logs]
        val_entropy = [rec.get("val_routing_entropy", float("nan")) for rec in run_logs]
        plt.plot(steps, val_entropy, linestyle="-", linewidth=2.2, label=_run_label(run_name))

    _apply_plot_y_limits(plt_mod=plt, y_values=y_values, clamp_zero_min=True)
    if max_step > cut_step:
        plt.xlim(left=cut_step, right=max_step)

    plt.xlabel("Step")
    plt.ylabel("Validation Routing Entropy")
    plt.title(
        "LUCID-MoE: Routing Distribution Entropy on Tiny Shakespeare"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_max_load_violation_curves(
    *,
    logs_by_run: Dict[str, List[Dict[str, float]]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "--plot_losses was requested but matplotlib is unavailable in this environment."
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(run_logs for run_logs in logs_by_run.values()):
        raise ValueError("plot_max_load_violation_curves received no logs.")

    plt.figure(figsize=(10.0, 4.0))
    max_step, cut_step, y_values = _collect_plot_window_values(
        logs_by_run=logs_by_run,
        key="val_max_load_violation",
        require_positive=False,
    )
    for run_name, run_logs in logs_by_run.items():
        if not run_logs:
            continue
        steps = [rec["step"] for rec in run_logs]
        val_violations = [rec.get("val_max_load_violation", float("nan")) for rec in run_logs]
        plt.plot(steps, val_violations, linestyle="-", linewidth=2.2, label=_run_label(run_name))

    _apply_plot_y_limits(plt_mod=plt, y_values=y_values, clamp_zero_min=True)
    if max_step > cut_step:
        plt.xlim(left=cut_step, right=max_step)

    plt.xlabel("Step")
    plt.ylabel("Validation Max Load Violation")
    plt.title("LUCID-MoE: Validation Max Load Violation on Tiny Shakespeare")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def resolve_plot_out_path(*, plot_out: Optional[Path], json_out: Optional[Path]) -> Path:
    if plot_out is not None:
        return plot_out
    if json_out is not None:
        return json_out.with_suffix(".png")
    return Path("loss_plot.png")


def resolve_entropy_plot_out_path(*, plot_out: Optional[Path], json_out: Optional[Path]) -> Path:
    if plot_out is not None:
        return plot_out.with_name(f"{plot_out.stem}_entropy{plot_out.suffix}")
    if json_out is not None:
        return json_out.with_name(f"{json_out.stem}_entropy.png")
    return Path("routing_entropy_plot.png")


def resolve_load_violation_plot_out_path(*, plot_out: Optional[Path], json_out: Optional[Path]) -> Path:
    if plot_out is not None:
        return plot_out.with_name(f"{plot_out.stem}_load_violation{plot_out.suffix}")
    if json_out is not None:
        return json_out.with_name(f"{json_out.stem}_load_violation.png")
    return Path("routing_load_violation_plot.png")


def build_model(
    *,
    vocab_size: int,
    block_size: int,
    d_model: int,
    n_layers: int,
    attn_heads: int,
    moe_heads: int,
    d_moe_latent: int,
    num_experts: int,
    top_k: int,
    expert_hidden: int,
    lucid_router_eps: float,
    enable_lucid_router_default: bool,
    enable_sigmoid_gating_default: bool,
    enable_qe_norm_default: bool,
    enable_auxfree_bias_default: bool,
    auxfree_bias_lr: float,
    auxfree_bias_clip: float,
    kv_block_size: int,
    device: torch.device,
    init_generator: Optional[torch.Generator] = None,
) -> LatentMoEShakespeareLM:
    moe_cfg = MoEConfig(
        d_model=d_model,
        moe_heads=moe_heads,
        d_moe_latent=d_moe_latent,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden=expert_hidden,
        lucid_router_eps=lucid_router_eps,
        enable_lucid_router_default=enable_lucid_router_default,
        enable_sigmoid_gating_default=enable_sigmoid_gating_default,
        enable_qe_norm_default=enable_qe_norm_default,
        enable_auxfree_bias_default=enable_auxfree_bias_default,
        auxfree_bias_lr=auxfree_bias_lr,
        auxfree_bias_clip=auxfree_bias_clip,
        kv_block_size=kv_block_size,
    )
    model = LatentMoEShakespeareLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        n_layers=n_layers,
        attn_heads=attn_heads,
        moe_cfg=moe_cfg,
        init_generator=init_generator,
    )
    return model.to(device)


def build_optimizer(
    *,
    model: LatentMoEShakespeareLM,
    lr_embedding: float,
    lr_router_embedding: float,
    lr_linear: float,
    lr_lm_head: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    muon_ns_steps: int,
) -> Tuple[
    torch.optim.Optimizer | CompositeOptimizer,
    Dict[str, int],
    Dict[str, List[str]],
]:
    if lr_embedding <= 0.0 or lr_router_embedding <= 0.0 or lr_linear <= 0.0 or lr_lm_head <= 0.0:
        raise ValueError("All learning rates must be > 0.")

    named_params = list(model.named_parameters())
    token_embedding_adamw_params = [
        p for name, p in named_params if name.startswith("token_embedding.")
    ]
    router_embedding_adamw_params = [
        p for name, p in named_params if name.endswith("router_embedding")
    ]
    lm_head_muon_params = [
        p
        for name, p in named_params
        if name.startswith("lm_head.")
        and not name.startswith("token_embedding.")
        and not name.endswith("router_embedding")
    ]
    linear_muon_params = [
        p
        for name, p in named_params
        if not name.startswith("lm_head.")
        and not name.startswith("token_embedding.")
        and not name.endswith("router_embedding")
    ]

    token_embedding_adamw_names = [
        name for name, _ in named_params if name.startswith("token_embedding.")
    ]
    router_embedding_adamw_names = [
        name for name, _ in named_params if name.endswith("router_embedding")
    ]
    lm_head_muon_names = [
        name
        for name, _ in named_params
        if name.startswith("lm_head.")
        and not name.startswith("token_embedding.")
        and not name.endswith("router_embedding")
    ]
    linear_muon_names = [
        name
        for name, _ in named_params
        if not name.startswith("lm_head.")
        and not name.startswith("token_embedding.")
        and not name.endswith("router_embedding")
    ]

    optimizers: List[torch.optim.Optimizer] = []
    if token_embedding_adamw_params:
        token_adamw_opt = torch.optim.AdamW(
            token_embedding_adamw_params,
            lr=lr_embedding,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        for group in token_adamw_opt.param_groups:
            group["group_name"] = "token_embedding_adamw"
            group["lr_peak"] = lr_embedding
        optimizers.append(token_adamw_opt)
    if router_embedding_adamw_params:
        router_adamw_opt = torch.optim.AdamW(
            router_embedding_adamw_params,
            lr=lr_router_embedding,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        for group in router_adamw_opt.param_groups:
            group["group_name"] = "router_embedding_adamw"
            group["lr_peak"] = lr_router_embedding
        optimizers.append(router_adamw_opt)
    if linear_muon_params:
        linear_muon_opt = MuonOptimizer(
            linear_muon_params,
            lr=lr_linear,
            beta=beta1,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
        for group in linear_muon_opt.param_groups:
            group["group_name"] = "linear_muon"
            group["lr_peak"] = lr_linear
        optimizers.append(linear_muon_opt)
    if lm_head_muon_params:
        lm_head_muon_opt = MuonOptimizer(
            lm_head_muon_params,
            lr=lr_lm_head,
            beta=beta1,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
        for group in lm_head_muon_opt.param_groups:
            group["group_name"] = "lm_head_muon"
            group["lr_peak"] = lr_lm_head
        optimizers.append(lm_head_muon_opt)

    if not optimizers:
        raise RuntimeError("No trainable parameters found for optimizer construction.")
    split_counts = {
        "token_embedding_adamw": len(token_embedding_adamw_params),
        "router_embedding_adamw": len(router_embedding_adamw_params),
        "linear_muon": len(linear_muon_params),
        "lm_head_muon": len(lm_head_muon_params),
    }
    split_names = {
        "token_embedding_adamw": token_embedding_adamw_names,
        "router_embedding_adamw": router_embedding_adamw_names,
        "linear_muon": linear_muon_names,
        "lm_head_muon": lm_head_muon_names,
    }
    if len(optimizers) == 1:
        return optimizers[0], split_counts, split_names
    return CompositeOptimizer(optimizers), split_counts, split_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MH-Latent-MoE (+ optional LUCID) on Tiny Shakespeare.")

    parser.add_argument("--data_dir", type=Path, default=Path("data/tinyshakespeare"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data_seed", type=int, default=2026)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--attn_heads", type=int, default=None)
    parser.add_argument("--moe_heads", type=int, default=4)
    parser.add_argument("--d_moe_latent", type=int, default=None)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--expert_hidden", type=int, default=512)
    parser.add_argument("--kv_block_size", type=int, default=128)

    parser.add_argument("--enable_lucid_router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_both_lucid_router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lucid_router_eps", type=float, default=1e-4)
    parser.add_argument("--enable_sigmoid_gating", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_qe_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_auxfree_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auxfree_bias_lr", type=float, default=1e-2)
    parser.add_argument("--auxfree_bias_clip", type=float, default=10.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_embedding", type=float, default=None)
    parser.add_argument("--lr_router_embedding", type=float, default=None)
    parser.add_argument("--lr_linear", type=float, default=None)
    parser.add_argument("--lr_lm_head", type=float, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--muon_ns_steps", type=int, default=8)

    parser.add_argument("--plot_losses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--live_plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_out", type=Path, default=None)
    parser.add_argument("--json_out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be > 0.")
    if args.eval_batches <= 0:
        raise ValueError("--eval_batches must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.block_size <= 0:
        raise ValueError("--block_size must be > 0.")
    if args.d_model <= 0:
        raise ValueError("--d_model must be > 0.")
    if args.n_layers <= 0:
        raise ValueError("--n_layers must be > 0.")
    if args.moe_heads <= 0:
        raise ValueError("--moe_heads must be > 0.")
    if args.num_experts <= 0:
        raise ValueError("--num_experts must be > 0.")
    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0.")
    if args.expert_hidden <= 0:
        raise ValueError("--expert_hidden must be > 0.")
    if args.kv_block_size <= 0:
        raise ValueError("--kv_block_size must be > 0.")

    if not (0.0 < args.min_lr_ratio <= 1.0):
        raise ValueError("--min_lr_ratio must be in (0, 1].")
    if args.auxfree_bias_lr < 0.0:
        raise ValueError("--auxfree_bias_lr must be >= 0.")
    if args.auxfree_bias_clip < 0.0:
        raise ValueError("--auxfree_bias_clip must be >= 0.")
    if args.d_moe_latent is None:
        raise ValueError("--d_moe_latent must be set explicitly.")
    if args.d_moe_latent <= 0:
        raise ValueError("--d_moe_latent must be > 0.")

    configure_determinism(args.deterministic)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or --device auto.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build.")
    if device.type != "cuda":
        raise RuntimeError("FlexAttention-only implementation requires CUDA.")

    print(f"Using device: {device}")
    set_seed(args.seed)

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
    )
    lr_embedding = args.lr if args.lr_embedding is None else args.lr_embedding
    lr_router_embedding = lr_embedding if args.lr_router_embedding is None else args.lr_router_embedding
    lr_linear = args.lr if args.lr_linear is None else args.lr_linear
    lr_lm_head = args.lr if args.lr_lm_head is None else args.lr_lm_head
    attn_heads = args.moe_heads if args.attn_heads is None else args.attn_heads
    if attn_heads <= 0:
        raise ValueError("--attn_heads must be > 0.")
    if args.d_model % attn_heads != 0:
        raise ValueError("--d_model must be divisible by --attn_heads.")
    if args.top_k > args.num_experts:
        raise ValueError("--top_k must be <= --num_experts.")
    print(
        f"LR schedule (constant then decay in last 20%): "
        f"token_embed {lr_embedding:.3e}->{lr_embedding * args.min_lr_ratio:.3e}, "
        f"router_embed {lr_router_embedding:.3e}->{lr_router_embedding * args.min_lr_ratio:.3e}, "
        f"linear {lr_linear:.3e}->{lr_linear * args.min_lr_ratio:.3e}, "
        f"lm_head {lr_lm_head:.3e}->{lr_lm_head * args.min_lr_ratio:.3e} | "
        f"Heads(attn/moe)={attn_heads}/{args.moe_heads}, d_moe_latent={args.d_moe_latent} | "
        f"LUCID_ROUTER={'both (off,on)' if args.run_both_lucid_router else ('on' if args.enable_lucid_router else 'off')} | "
        f"GATING={'sigmoid' if args.enable_sigmoid_gating else 'softmax'} | "
        f"QE_NORM={'on' if args.enable_qe_norm else 'off'} | "
        f"AUXFREE_BIAS={'on' if args.enable_auxfree_bias else 'off'} "
        f"(lr={args.auxfree_bias_lr:.2e}, clip={args.auxfree_bias_clip:.2f}) | "
        f"FlexAttention=on | "
        f"Deterministic={'on' if args.deterministic else 'off'}"
    )

    run_specs: List[Tuple[str, bool]]
    if args.run_both_lucid_router:
        run_specs = [("mh_lmoe_lucid_off", False), ("mh_lmoe_lucid_on", True)]
    else:
        run_specs = [("mh_lmoe_lucid_off", False)] if not args.enable_lucid_router else [("mh_lmoe_lucid_on", True)]
    logs_by_run: Dict[str, List[Dict[str, float]]] = {}
    plot_path: Optional[Path] = None
    entropy_plot_path: Optional[Path] = None
    load_violation_plot_path: Optional[Path] = None
    if args.plot_losses:
        plot_path = resolve_plot_out_path(plot_out=args.plot_out, json_out=args.json_out)
        entropy_plot_path = resolve_entropy_plot_out_path(plot_out=args.plot_out, json_out=args.json_out)
        load_violation_plot_path = resolve_load_violation_plot_out_path(
            plot_out=args.plot_out,
            json_out=args.json_out,
        )
        print(
            "Plotting enabled: "
            f"live_updates={'on' if args.live_plot else 'off'} | "
            f"loss={plot_path} | entropy={entropy_plot_path} | load={load_violation_plot_path}"
        )

    def refresh_plots_live() -> None:
        if not args.plot_losses or not args.live_plot:
            return
        if plot_path is None or entropy_plot_path is None or load_violation_plot_path is None:
            return
        non_empty_logs = {name: run_logs for name, run_logs in logs_by_run.items() if run_logs}
        if not non_empty_logs:
            return
        plot_loss_curves(logs_by_run=non_empty_logs, out_path=plot_path)
        plot_routing_entropy_curves(logs_by_run=non_empty_logs, out_path=entropy_plot_path)
        plot_max_load_violation_curves(logs_by_run=non_empty_logs, out_path=load_violation_plot_path)

    for run_name, run_enable_lucid_router in run_specs:
        set_seed(args.seed)
        init_rng = make_torch_generator(args.seed, torch.device("cpu"))
        model = build_model(
            vocab_size=dataset.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            attn_heads=attn_heads,
            moe_heads=args.moe_heads,
            d_moe_latent=args.d_moe_latent,
            num_experts=args.num_experts,
            top_k=args.top_k,
            expert_hidden=args.expert_hidden,
            lucid_router_eps=args.lucid_router_eps,
            enable_lucid_router_default=run_enable_lucid_router,
            enable_sigmoid_gating_default=args.enable_sigmoid_gating,
            enable_qe_norm_default=args.enable_qe_norm,
            enable_auxfree_bias_default=args.enable_auxfree_bias,
            auxfree_bias_lr=args.auxfree_bias_lr,
            auxfree_bias_clip=args.auxfree_bias_clip,
            kv_block_size=args.kv_block_size,
            device=device,
            init_generator=init_rng,
        )
        optimizer, split_counts, split_names = build_optimizer(
            model=model,
            lr_embedding=lr_embedding,
            lr_router_embedding=lr_router_embedding,
            lr_linear=lr_linear,
            lr_lm_head=lr_lm_head,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            muon_ns_steps=args.muon_ns_steps,
        )
        print(
            f"[{run_name}] optimizer split: "
            f"token_embedding_adamw={split_counts['token_embedding_adamw']}, "
            f"router_embedding_adamw={split_counts['router_embedding_adamw']}, "
            f"linear_muon={split_counts['linear_muon']}, "
            f"lm_head_muon={split_counts['lm_head_muon']}, "
        )
        print(f"[{run_name}] AdamW token_embedding params: {', '.join(split_names['token_embedding_adamw'])}")
        print(f"[{run_name}] AdamW router_embedding params: {', '.join(split_names['router_embedding_adamw'])}")
        print(f"[{run_name}] Muon linear params: {', '.join(split_names['linear_muon'])}")
        print(f"[{run_name}] Muon lm_head params: {', '.join(split_names['lm_head_muon'])}")

        run_logs: List[Dict[str, float]] = []
        logs_by_run[run_name] = run_logs
        train_one_run(
            run_name=run_name,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            device=device,
            cfg=train_cfg,
            data_seed=args.data_seed,
            min_lr_ratio=args.min_lr_ratio,
            enable_lucid_router=run_enable_lucid_router,
            enable_sigmoid_gating=args.enable_sigmoid_gating,
            logs=run_logs,
            on_eval=refresh_plots_live,
        )

    summary = {run_name: logs[-1]["val_loss"] for run_name, logs in logs_by_run.items()}

    print("\n=== Final validation loss ===")
    for run_name, final_val in summary.items():
        print(f"{run_name}: {final_val:.4f}")

    if args.plot_losses:
        plot_loss_curves(logs_by_run=logs_by_run, out_path=plot_path)
        plot_routing_entropy_curves(logs_by_run=logs_by_run, out_path=entropy_plot_path)
        plot_max_load_violation_curves(logs_by_run=logs_by_run, out_path=load_violation_plot_path)
        print(f"Wrote loss plot to {plot_path}")
        print(f"Wrote routing entropy plot to {entropy_plot_path}")
        print(f"Wrote max load violation plot to {load_violation_plot_path}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        config = {}
        for key, value in vars(args).items():
            config[key] = str(value) if isinstance(value, Path) else value
        if plot_path is not None:
            config["plot_out"] = str(plot_path)
        if entropy_plot_path is not None:
            config["plot_out_entropy"] = str(entropy_plot_path)
        if load_violation_plot_path is not None:
            config["plot_out_load_violation"] = str(load_violation_plot_path)
        payload = {
            "config": {**config, "device": str(device)},
            "train_config": {
                "steps": train_cfg.steps,
                "eval_interval": train_cfg.eval_interval,
                "eval_batches": train_cfg.eval_batches,
                "batch_size": train_cfg.batch_size,
                "block_size": train_cfg.block_size,
            },
            "summary": summary,
            "logs": logs_by_run,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote logs to {args.json_out}")


__all__ = [
    "MoEConfig",
    "MultiHeadLatentMoE",
]


if __name__ == "__main__":
    main()
