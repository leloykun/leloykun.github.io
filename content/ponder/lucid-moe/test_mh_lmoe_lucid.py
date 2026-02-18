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


def _reference_expert_ffn(
    q_assign: torch.Tensor,
    expert_ids: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    selected_w1 = W1[expert_ids.to(torch.int64)]  # (Nassign, d, m)
    hidden = torch.bmm(q_assign.unsqueeze(1), selected_w1).squeeze(1)
    hidden = F.gelu(hidden)
    selected_w2 = W2[expert_ids.to(torch.int64)]  # (Nassign, m, d)
    return torch.bmm(hidden.unsqueeze(1), selected_w2).squeeze(1)


def _reference_mh_lmoe_forward(
    model,
    x: torch.Tensor,
    *,
    enable_lucid_router: bool,
) -> torch.Tensor:
    cfg = model.cfg
    if x.ndim != 3:
        raise ValueError("x must be rank-3 (B, T, d_model).")

    B, T, _ = x.shape
    H, d, k = cfg.moe_heads, model.d_head, cfg.top_k
    Ntok = B * T

    xh = model.in_proj(x).view(B, T, H, d)
    head_outputs = []
    for h in range(H):
        q = xh[:, :, h, :].reshape(Ntok, d).contiguous()  # (Ntok, d)
        logits = torch.matmul(q, model.router_weight[h].transpose(0, 1))  # (Ntok, E)
        topk = torch.topk(logits, k=k, dim=-1)
        topk_idx = topk.indices  # (Ntok, k)
        alpha = torch.softmax(topk.values, dim=-1)  # (Ntok, k)

        q_assign = q.repeat_interleave(k, dim=0)  # (Ntok*k, d)
        expert_ids = topk_idx.reshape(-1).to(torch.int64)  # (Ntok*k,)

        selected_w1 = model.W1[h][expert_ids]  # (Ntok*k, d, m)
        hidden = torch.bmm(q_assign.unsqueeze(1), selected_w1).squeeze(1)  # (Ntok*k, m)
        hidden = F.gelu(hidden)
        selected_w2 = model.W2[h][expert_ids]  # (Ntok*k, m, d)
        y_assign = torch.bmm(hidden.unsqueeze(1), selected_w2).squeeze(1).view(Ntok, k, d)

        if enable_lucid_router and k > 1:
            router_weight = model.router_weight[h]  # (E, d)
            key_embed = mh_lmoe_lucid.norm(router_weight[topk_idx])  # (Ntok, k, d)
            sim = key_embed @ key_embed.transpose(-1, -2)
            P = torch.exp(sim.to(torch.float32) / math.sqrt(d) - math.sqrt(d))
            eye = torch.eye(k, device=P.device, dtype=P.dtype).unsqueeze(0)
            P = P + cfg.lucid_router_eps * eye
            y_assign_corr = torch.linalg.solve(P, y_assign.to(torch.float32))
            mixed = (alpha.to(torch.float32).unsqueeze(-1) * y_assign_corr).sum(dim=1).to(y_assign.dtype)
        else:
            mixed = (alpha.unsqueeze(-1) * y_assign).sum(dim=1)

        head_outputs.append(mixed.view(B, T, d))

    y = torch.cat(head_outputs, dim=-1)
    return model.out_proj(y)


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
def test_flex_expert_assignments_match_reference() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")

    # Match the code path used by MultiHeadLatentMoE: multi-head assignment kernel.
    H, E, d, m = 3, 7, 16, 77
    q_len = 193

    q_by_head = torch.randn(H, q_len, d, device=device)
    expert_ids_by_head = torch.randint(0, E, (H, q_len), device=device, dtype=torch.int64)

    W1 = torch.randn(H, E, d, m, device=device)
    W2 = torch.randn(H, E, m, d, device=device)

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=H * d,
        moe_heads=H,
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
def test_mh_lmoe_forward_runs(enable_lucid_router: bool) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=4,
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
        y = model(x, enable_lucid_router=enable_lucid_router)

    assert y.shape == (8, 32, cfg.d_model)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
@pytest.mark.parametrize("enable_lucid_router", [False, True])
def test_mh_lmoe_forward_matches_reference(enable_lucid_router: bool) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=4,
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
        y_flex = model(x, enable_lucid_router=enable_lucid_router)
        y_ref = _reference_mh_lmoe_forward(model, x, enable_lucid_router=enable_lucid_router)

    torch.testing.assert_close(y_flex, y_ref, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not HAS_FLEX, reason="FlexAttention is unavailable in this PyTorch build.")
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required for FlexAttention parity tests.")
def test_enable_lucid_router_toggle_changes_reference_output() -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")

    cfg = mh_lmoe_lucid.MoEConfig(
        d_model=64,
        moe_heads=4,
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
