# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch.func import vmap

def _make_qkv(B, H, L, D, device, dtype):
    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    return q, k, v

def _loop_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    outs = []
    for b in range(q.shape[0]):
        outs.append(torch._scaled_dot_product_efficient_attention(  # type: ignore[attr-defined]
            q[b], k[b], v[b],
            attn_mask=attn_mask[b] if attn_mask is not None else None,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale))
    return torch.stack(outs, dim=0)

def _skip_if_unavailable(device):
    # Some builds may not expose the efficient kernel on CPU; skip gracefully.
    try:
        q = torch.randn(1, 1, 2, 4, device=device)
        k = torch.randn(1, 1, 2, 4, device=device)
        v = torch.randn(1, 1, 2, 4, device=device)
        torch._scaled_dot_product_efficient_attention(q, k, v)  # type: ignore[attr-defined]
    except Exception as e:
        pytest.skip(f"efficient SDPA op not available on {device}: {e}")

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vmap_sdpa_efficient_matches_loop(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    _skip_if_unavailable(device)

    B, H, L, D = 3, 4, 16, 64
    dtype = torch.float32 if device == "cpu" else torch.float16
    q, k, v = _make_qkv(B, H, L, D, device, dtype)

    def f(q_, k_, v_):
        return torch._scaled_dot_product_efficient_attention(  # type: ignore[attr-defined]
            q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        )

    out_vmap = vmap(f)(q, k, v)
    out_loop = _loop_sdpa(q, k, v)
    assert torch.allclose(out_vmap, out_loop, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vmap_sdpa_efficient_backward_matches_loop(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    _skip_if_unavailable(device)

    B, H, L, D = 2, 2, 8, 32
    dtype = torch.float32 if device == "cpu" else torch.float16
    q, k, v = _make_qkv(B, H, L, D, device, dtype)

    def loss(q_, k_, v_):
        return torch._scaled_dot_product_efficient_attention(  # type: ignore[attr-defined]
            q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        ).sum()

    # vmap loss over batch, then grads
    lv = vmap(loss)(q, k, v)                      # [B]
    gq_v, gk_v, gv_v = torch.autograd.grad(lv.sum(), (q, k, v))

    # loop loss over batch, then grads
    lo = _loop_sdpa(q, k, v).sum(dim=(1, 2, 3))   # per-sample scalar
    gq_l, gk_l, gv_l = torch.autograd.grad(lo.sum(), (q, k, v))

    assert torch.allclose(gq_v, gq_l, rtol=1e-2, atol=1e-2)
    assert torch.allclose(gk_v, gk_l, rtol=1e-2, atol=1e-2)
    assert torch.allclose(gv_v, gv_l, rtol=1e-2, atol=1e-2)
