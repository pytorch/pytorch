# Owner(s): ["module: dynamo"]

import torch
import torch.nn.functional as F
from torch._dynamo.exc import Unsupported
from torch._higher_order_ops.flex_ep import flex_ep, flex_ep_backward
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton


BATCH = 4
HIDDEN_DIM = 8
INTERMEDIATE_DIM = 16
NUM_EXPERTS = 2
TOPK = 2


def _make_inputs(device, requires_grad=False):
    torch.manual_seed(0)
    x = torch.randn(
        BATCH,
        HIDDEN_DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )
    topk_idx = torch.tensor(
        [[0, 1], [1, 0], [0, 0], [1, 1]],
        device=device,
        dtype=torch.int64,
    )
    w13 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_DIM,
        HIDDEN_DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )
    w2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )
    return x, topk_idx, w13, w2


def _dispatch_fn(x_expanded, topk_idx):
    B, topk, D = x_expanded.shape
    flat_idx = topk_idx.reshape(-1)
    flat_x = x_expanded.reshape(B * topk, D)
    order = torch.argsort(flat_idx, stable=True)
    recv_x = flat_x.index_select(0, order)
    counts = torch.bincount(flat_idx, minlength=NUM_EXPERTS).to(torch.int32)
    local_experts_start = torch.cat(
        (
            torch.zeros(1, device=x_expanded.device, dtype=torch.int32),
            counts.cumsum(0, dtype=torch.int32),
        )
    )
    recv_origin_global_token_id = order.to(torch.int64)
    dest_ranks = torch.zeros_like(recv_origin_global_token_id, dtype=torch.int32)
    dest_offsets = torch.empty_like(recv_origin_global_token_id)
    dest_offsets[order] = torch.arange(
        order.numel(),
        device=x_expanded.device,
        dtype=dest_offsets.dtype,
    )
    recv_total_tokens = torch.tensor(
        order.numel(),
        device=x_expanded.device,
        dtype=torch.int32,
    )
    max_recv_tokens = recv_total_tokens.clone()
    return (
        recv_x,
        recv_origin_global_token_id,
        local_experts_start.clone(),
        dest_ranks,
        dest_offsets,
        max_recv_tokens,
        recv_total_tokens,
        local_experts_start,
        BATCH,
    )


def _combine_fn(
    y3,
    recv_origin_global_token_id,
    expert_begin_offset_per_ep,
    dest_ranks,
    dest_offsets,
    max_recv_tokens,
    recv_total_tokens,
    local_experts_start,
    B,
):
    del expert_begin_offset_per_ep
    del dest_ranks, dest_offsets, max_recv_tokens, recv_total_tokens
    del local_experts_start
    out_flat = y3.new_empty((B * TOPK, y3.shape[-1]))
    out_flat[recv_origin_global_token_id] = y3
    return out_flat.view(B, TOPK, y3.shape[-1]).sum(1)


def _combine_bwd_fn(
    dy,
    recv_origin_global_token_id,
    expert_begin_offset_per_ep,
    dest_ranks,
    dest_offsets,
    max_recv_tokens,
    recv_total_tokens,
    local_experts_start,
    B,
):
    del expert_begin_offset_per_ep
    del dest_ranks, dest_offsets, max_recv_tokens, recv_total_tokens
    del local_experts_start
    dy_flat = (
        dy.unsqueeze(1)
        .expand(B, TOPK, dy.shape[-1])
        .reshape(
            B * TOPK,
            dy.shape[-1],
        )
    )
    return dy_flat.index_select(0, recv_origin_global_token_id)


def _dispatch_bwd_fn(
    dx_recv,
    recv_origin_global_token_id,
    expert_begin_offset_per_ep,
    dest_ranks,
    dest_offsets,
    max_recv_tokens,
    recv_total_tokens,
    local_experts_start,
    B,
):
    del expert_begin_offset_per_ep
    del dest_ranks, dest_offsets, max_recv_tokens, recv_total_tokens
    del local_experts_start
    dxpn = dx_recv.new_empty((B * TOPK, dx_recv.shape[-1]))
    dxpn[recv_origin_global_token_id] = dx_recv
    return dxpn.view(B, TOPK, dx_recv.shape[-1])


def _flex_ep_call(x, topk_idx, w13, w2):
    return flex_ep(
        x,
        topk_idx,
        w13,
        w2,
        _dispatch_fn,
        _combine_fn,
        _combine_bwd_fn,
        _dispatch_bwd_fn,
        num_experts=NUM_EXPERTS,
        ep_rank=0,
        ep_size=1,
        max_tokens=BATCH,
        topk=TOPK,
    )


def _make_backward_inputs(device):
    x, topk_idx, w13, w2 = _make_inputs(device)
    x_expanded = x.unsqueeze(1).expand(-1, TOPK, -1).contiguous()
    dispatch_out = _dispatch_fn(x_expanded, topk_idx)
    recv_x, *tmi_flat = dispatch_out
    offs = tmi_flat[6][1:].to(torch.int32)
    y1 = torch._grouped_mm(recv_x, w13.transpose(-2, -1), offs=offs)
    gate, up = y1.chunk(2, dim=-1)
    y2 = F.silu(gate) * up
    dy = torch.ones((BATCH, HIDDEN_DIM), device=device, dtype=torch.bfloat16)
    return dy, recv_x.clone(), y1, y2, w13, w2, offs, *tmi_flat


def _flex_ep_backward_call(dy, recv_x, y1, y2, w13, w2, offs, *tmi_flat):
    return flex_ep_backward(
        dy,
        recv_x,
        y1,
        y2,
        w13,
        w2,
        offs,
        _combine_bwd_fn,
        _dispatch_bwd_fn,
        *tmi_flat,
    )


def _reference(x, topk_idx, w13, w2):
    outputs = []
    for b in range(x.shape[0]):
        token_out = x.new_zeros((HIDDEN_DIM,))
        for k in range(TOPK):
            expert = int(topk_idx[b, k].item())
            y1 = x[b : b + 1] @ w13[expert].transpose(-2, -1)
            gate, up = y1.chunk(2, dim=-1)
            y2 = F.silu(gate) * up
            token_out = token_out + (y2 @ w2[expert].transpose(-2, -1)).squeeze(0)
        outputs.append(token_out)
    return torch.stack(outputs)


def _run_with_grads(fn, args):
    y = fn(*args)
    y.float().sum().backward()
    return y, args[0].grad, args[2].grad, args[3].grad


class TestFlexEp(TestCase):
    def test_eager_ep1_matches_reference(self, device):
        x, topk_idx, w13, w2 = _make_inputs(device)
        result = _flex_ep_call(x, topk_idx, w13, w2)
        expected = _reference(x, topk_idx, w13, w2)
        self.assertEqual(result, expected, rtol=1e-1, atol=5e-1)

    def test_backward_ep1_matches_eager_reference(self, device):
        args = _make_inputs(device, requires_grad=True)
        result, dx, dw13, dw2 = _run_with_grads(_flex_ep_call, args)

        ref_args = _make_inputs(device, requires_grad=True)
        expected, expected_dx, expected_dw13, expected_dw2 = _run_with_grads(
            _reference,
            ref_args,
        )

        self.assertEqual(result, expected, rtol=1e-1, atol=5e-1)
        self.assertEqual(dx, expected_dx, rtol=1e-1, atol=5e-1)
        self.assertEqual(dw13, expected_dw13, rtol=1e-1, atol=5e-1)
        self.assertEqual(dw2, expected_dw2, rtol=1e-1, atol=5e-1)

    def test_dynamo_preserves_flex_ep_hop(self, device):
        args = _make_inputs(device)
        gm, _ = torch._dynamo.export(_flex_ep_call)(*args)

        self.assertIn("higher_order.flex_ep", gm.code)
        self.assertNotIn("_grouped_mm", gm.code)

    def test_dynamo_rejects_direct_flex_ep_backward_hop(self, device):
        args = _make_backward_inputs(device)

        with self.assertRaisesRegex(Unsupported, "flex_ep_backward"):
            torch._dynamo.export(_flex_ep_backward_call)(*args)

    def test_aot_eager_forward_backward_matches_eager(self, device):
        eager_args = _make_inputs(device, requires_grad=True)
        eager = _run_with_grads(_flex_ep_call, eager_args)

        compiled_args = _make_inputs(device, requires_grad=True)
        compiled_fn = torch.compile(
            _flex_ep_call,
            backend="aot_eager",
            fullgraph=True,
        )
        compiled = _run_with_grads(compiled_fn, compiled_args)

        for actual, expected in zip(compiled, eager):
            self.assertEqual(actual, expected)

    def test_inductor_forward_backward_matches_eager_cuda(self, device):
        if not str(device).startswith("cuda"):
            self.skipTest("flex_ep inductor grouped-mm coverage is CUDA-only")
        if not has_triton():
            self.skipTest("flex_ep inductor test requires Triton")

        eager_args = _make_inputs(device, requires_grad=True)
        eager = _run_with_grads(_flex_ep_call, eager_args)

        compiled_args = _make_inputs(device, requires_grad=True)
        compiled_fn = torch.compile(
            _flex_ep_call,
            backend="inductor",
            fullgraph=True,
        )
        compiled = _run_with_grads(compiled_fn, compiled_args)

        for actual, expected in zip(compiled, eager):
            self.assertEqual(actual, expected, rtol=1e-1, atol=1.0)


instantiate_device_type_tests(TestFlexEp, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
