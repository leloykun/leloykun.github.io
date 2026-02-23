from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


def _load_module():
    module_path = Path(__file__).with_name("mh_lmoe_lucid.py")
    spec = importlib.util.spec_from_file_location("mh_lmoe_lucid", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mh_lmoe_lucid = _load_module()


HAS_FLEX = (
    mh_lmoe_lucid.flex_attention is not None
    and mh_lmoe_lucid.BlockMask is not None
    and mh_lmoe_lucid.create_block_mask is not None
)
HAS_CUDA = torch.cuda.is_available()


def _max_min_avg_vio(scores: torch.Tensor, k: int) -> tuple[float, float, float]:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2 (m, n).")
    m, n = scores.shape
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n.")
    topk = torch.topk(scores, k=k, dim=1).indices
    freq = torch.bincount(topk.reshape(-1), minlength=n).to(torch.float32)
    freq = freq / freq.sum() * float(n) - 1.0
    return float(freq.max().item()), float(freq.min().item()), float(freq.abs().mean().item())


def test_quantile_bias_helper_improves_balance() -> None:
    torch.manual_seed(3)
    m, n, k = 6000, 64, 8
    scores = torch.rand((m, n), dtype=torch.float32) + torch.rand((n,), dtype=torch.float32)

    bias = mh_lmoe_lucid.MultiHeadLatentMoE._compute_quantile_expert_bias(
        scores.unsqueeze(0),
        prev_bias=torch.zeros((1, n), dtype=torch.float32),
        top_k=k,
    )[0]
    assert bias.shape == (n,)

    _, _, before_avg = _max_min_avg_vio(scores, k)
    _, _, after_avg = _max_min_avg_vio(scores + bias.unsqueeze(0), k)
    assert after_avg < before_avg


def _reference_expert_ffn(
    q_assign: torch.Tensor,
    expert_ids: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    # W1: (E, m, d), W2: (E, d, m)
    selected_w1 = W1[expert_ids.to(torch.int64)]  # (Nassign, m, d)
    hidden = torch.bmm(q_assign.unsqueeze(1), selected_w1.transpose(1, 2)).squeeze(1)
    hidden = F.gelu(hidden)
    selected_w2 = W2[expert_ids.to(torch.int64)]  # (Nassign, d, m)
    return torch.bmm(hidden.unsqueeze(1), selected_w2.transpose(1, 2)).squeeze(1)


def _reference_mh_lmoe_forward(
    model,
    x: torch.Tensor,
    *,
    enable_lucid_router: bool,
    enable_qe_norm: bool,
) -> torch.Tensor:
    cfg = model.cfg
    if x.ndim != 3:
        raise ValueError("x must be rank-3 (B, T, d_model).")

    B, T, _ = x.shape
    H, d, k = cfg.moe_heads, model.d_moe_latent, cfg.top_k
    Ntok = B * T
    q_by_head = torch.einsum("hdf,btf->hbtd", model.in_proj, x).reshape(H, Ntok, d).contiguous()
    if enable_qe_norm:
        q_router = mh_lmoe_lucid.norm(q_by_head)
        router_embedding = mh_lmoe_lucid.norm(model.router_embedding)
        logits_factor = 1.0 / math.sqrt(d)
    else:
        q_router = q_by_head
        router_embedding = model.router_embedding
        logits_factor = 1.0

    logits = logits_factor * torch.einsum("hnd,hed->hne", q_router, router_embedding)
    if model._expert_bias_mode() != "none":
        dirty_logits = logits + model.expert_load_bias.to(dtype=logits.dtype).unsqueeze(1)
        topk_dirty_values, topk_idx = torch.topk(dirty_logits, k=k, dim=-1)
        selected_bias = torch.gather(
            model.expert_load_bias.to(dtype=topk_dirty_values.dtype),
            dim=1,
            index=topk_idx.reshape(H, Ntok * k),
        ).reshape(H, Ntok, k)
        topk_values = topk_dirty_values - selected_bias
    else:
        topk_values, topk_idx = torch.topk(logits, k=k, dim=-1)

    alpha = torch.softmax(topk_values, dim=-1)  # (H, Ntok, k)
    q_assign_stacked = mh_lmoe_lucid.norm(q_router).unsqueeze(2).expand(H, Ntok, k, d).reshape(H, Ntok * k, d)
    expert_ids_stacked = topk_idx.reshape(H, Ntok * k).to(torch.int64)

    y_assign_stacked = torch.stack(
        [
            _reference_expert_ffn(
                q_assign_stacked[h],
                expert_ids_stacked[h],
                model.W1[h],
                model.W2[h],
            )
            for h in range(H)
        ],
        dim=0,
    ).view(H, Ntok, k, d)

    if not enable_lucid_router or k <= 1:
        mixed_by_head = (alpha.unsqueeze(-1) * y_assign_stacked).sum(dim=2)
    else:
        h_idx = torch.arange(H, device=x.device, dtype=torch.int64)[:, None, None].expand(H, Ntok, k)
        key_embed = mh_lmoe_lucid.norm(router_embedding[h_idx, topk_idx])
        sim = key_embed @ key_embed.transpose(-1, -2)
        P = torch.exp(sim.to(torch.float32) / math.sqrt(d) - math.sqrt(d))
        eye = torch.eye(k, device=P.device, dtype=P.dtype).view(1, 1, k, k)
        P = P + cfg.lucid_router_eps * eye
        y_assign_corr = torch.linalg.solve(P, y_assign_stacked.to(torch.float32))
        mixed_by_head = (alpha.to(torch.float32).unsqueeze(-1) * y_assign_corr).sum(dim=2)
        mixed_by_head = mixed_by_head.to(dtype=y_assign_stacked.dtype)

    y = mixed_by_head.permute(1, 0, 2).contiguous().view(B, T, H * d)
    return model.out_proj(y)


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
@pytest.mark.parametrize("num_heads", [1, 3], ids=["single_head", "multi_head"])
def test_flex_expert_assignments_match_reference(num_heads: int) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")

    # Match the code path used by MultiHeadLatentMoE expert assignment kernel.
    H, E, d, m = num_heads, 7, 16, 77
    q_len = 193

    q_by_head = torch.randn(H, q_len, d, device=device)
    expert_ids_by_head = torch.randint(0, E, (H, q_len), device=device, dtype=torch.int64)

    W1 = torch.randn(H, E, m, d, device=device)
    W2 = torch.randn(H, E, d, m, device=device)

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=H * d,
        moe_heads=H,
        d_moe_latent=d,
        num_experts=E,
        top_k=2,
        expert_hidden=m,
        kv_block_size=128,
    )
    model = mh_lmoe_lucid.MultiHeadLatentMoE(cfg).to(device)
    with torch.no_grad():
        model.W1.copy_(W1)
        model.W2.copy_(W2)

    y_ref = torch.stack(
        [
            _reference_expert_ffn(
                q_by_head[h],
                expert_ids_by_head[h],
                W1[h],
                W2[h],
            )
            for h in range(H)
        ],
        dim=0,
    )
    y_flex = model._expert_ffn_assignments(
        q_assign_stacked=q_by_head,
        expert_ids_stacked=expert_ids_by_head,
    )

    torch.testing.assert_close(y_flex, y_ref, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
@pytest.mark.parametrize("enable_lucid_router", [False, True])
@pytest.mark.parametrize("enable_qe_norm", [False, True])
@pytest.mark.parametrize("moe_heads", [1, 4], ids=["single_head", "multi_head"])
def test_mh_lmoe_forward_runs(enable_lucid_router: bool, enable_qe_norm: bool, moe_heads: int) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=moe_heads,
        d_moe_latent=64 // moe_heads,
        num_experts=8,
        top_k=2,
        expert_hidden=19,
        lucid_router_eps=1e-4,
        enable_lucid_router_default=False,
        kv_block_size=128,
    )

    model = mh_lmoe_lucid.MultiHeadLatentMoE(cfg).to(device)

    x = torch.randn(8, 32, cfg.d_model, device=device)
    with torch.no_grad():
        y = model(
            x,
            enable_lucid_router=enable_lucid_router,
            enable_qe_norm=enable_qe_norm,
        )

    assert y.shape == (8, 32, cfg.d_model)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
@pytest.mark.parametrize("enable_lucid_router", [False, True])
@pytest.mark.parametrize("enable_qe_norm", [False, True])
@pytest.mark.parametrize("moe_heads", [1, 4], ids=["single_head", "multi_head"])
def test_mh_lmoe_forward_matches_reference(enable_lucid_router: bool, enable_qe_norm: bool, moe_heads: int) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=moe_heads,
        d_moe_latent=64 // moe_heads,
        num_experts=8,
        top_k=2,
        expert_hidden=19,
        lucid_router_eps=1e-4,
        enable_lucid_router_default=False,
        kv_block_size=128,
    )
    model = mh_lmoe_lucid.MultiHeadLatentMoE(cfg).to(device)

    x = torch.randn(4, 16, cfg.d_model, device=device)
    with torch.no_grad():
        y_flex = model(
            x,
            enable_lucid_router=enable_lucid_router,
            enable_qe_norm=enable_qe_norm,
        )
        y_ref = _reference_mh_lmoe_forward(
            model,
            x,
            enable_lucid_router=enable_lucid_router,
            enable_qe_norm=enable_qe_norm,
        )

    torch.testing.assert_close(y_flex, y_ref, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
@pytest.mark.parametrize("moe_heads", [1, 4], ids=["single_head", "multi_head"])
def test_enable_lucid_router_toggle_changes_reference_output(moe_heads: int) -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=moe_heads,
        d_moe_latent=64 // moe_heads,
        num_experts=8,
        top_k=2,
        expert_hidden=19,
        lucid_router_eps=1e-4,
        enable_lucid_router_default=False,
        kv_block_size=128,
    )
    model = mh_lmoe_lucid.MultiHeadLatentMoE(cfg).to(device)
    x = torch.randn(8, 32, cfg.d_model, device=device)

    with torch.no_grad():
        y_off = model(x, enable_lucid_router=False)
        y_on = model(x, enable_lucid_router=True)

    assert not torch.allclose(y_off, y_on), "enable_lucid_router should affect the routed mixture."
