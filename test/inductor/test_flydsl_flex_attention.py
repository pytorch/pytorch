# Owner(s): ["module: inductor"]

import contextlib
import copy
import functools
import importlib.util
import inspect
import itertools
import math
import operator
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLKernelWrapper
from torch._inductor.kernel.flex import flex_flydsl_attention, flex_flydsl_backward
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _get_supported_bhsd_stride,
    _is_contiguous_shape_stride,
    flex_flydsl_dualwave_forward_template,
    flex_flydsl_forward_template,
    maybe_append_flydsl_dualwave_forward_choice,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_backward import (
    _get_flydsl_flex_attention_backward_config,
    flex_flydsl_backward_template,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import (
    add_block_states,
    lower_flydsl_mask_graph,
)
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import (
    and_masks,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    subtest,
)


class _FakeNode:
    def __init__(self, size, stride, dtype=torch.bfloat16, numel=None, device="cuda"):
        self._device = (
            torch.device(device, 0) if device == "cuda" else torch.device(device)
        )
        self._size = list(size)
        self._stride = list(stride)
        self._dtype = dtype
        self._numel = int(torch.tensor(size).prod().item()) if numel is None else numel

    def get_size(self):
        return self._size

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_device(self):
        return self._device

    def get_numel(self):
        return self._numel


def _contiguous_stride(size):
    stride = [1] * len(size)
    for index in range(len(size) - 2, -1, -1):
        stride[index] = stride[index + 1] * size[index + 1]
    return stride


def _fake_graph():
    return SimpleNamespace(
        sizevars=SimpleNamespace(
            guard_int=lambda value: int(value),
            shape_env=None,
        )
    )


def _supported_fake_forward_inputs(
    *,
    q_size=(1, 64, 512, 128),
    k_size=(1, 4, 1024, 128),
    v_size=None,
    mask_heads=1,
    index_width=8,
    dtype=torch.bfloat16,
    mask_fn=lambda b, h, q, kv: q + 512 >= kv,
):
    v_size = k_size if v_size is None else v_size
    stats_size = q_size[:3]
    mask_counts_size = [1, mask_heads, -(-q_size[2] // 128)]
    mask_indices_size = [*mask_counts_size, index_width]

    def node(size, node_dtype=dtype):
        return _FakeNode(size, _contiguous_stride(size), node_dtype)

    return {
        "query": node(q_size),
        "key": node(k_size),
        "value": node(v_size),
        "logsumexp": node(stats_size, torch.float32),
        "max_scores": node(stats_size, torch.float32),
        "kv_num_blocks": node(mask_counts_size, torch.int32),
        "kv_indices": node(mask_indices_size, torch.int32),
        "full_kv_num_blocks": node(mask_counts_size, torch.int32),
        "full_kv_indices": node(mask_indices_size, torch.int32),
        "subgraph": SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        ),
        "mask_graph": SimpleNamespace(graph_module=torch.fx.symbolic_trace(mask_fn)),
        "score_mod_other_buffers": [],
        "mask_mod_other_buffers": [],
        "scale": q_size[-1] ** -0.5,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
    }


def _supported_fake_backward_inputs(
    *,
    q_size=(1, 2, 256, 128),
    k_size=None,
    v_dim=128,
    mask_heads=1,
    dtype=torch.bfloat16,
    mask_fn=lambda b, h, q, kv: q >= kv,
):
    k_size = q_size if k_size is None else k_size
    inputs = _supported_fake_forward_inputs(
        q_size=q_size,
        k_size=k_size,
        v_size=(*k_size[:3], v_dim),
        mask_heads=mask_heads,
        index_width=-(-k_size[2] // 128),
        dtype=dtype,
        mask_fn=mask_fn,
    )
    output_shape = (*q_size[:3], v_dim)
    inputs["fw_subgraph"] = inputs.pop("subgraph")
    inputs.pop("logsumexp")
    inputs.pop("max_scores")
    inputs.update(
        out=_FakeNode(output_shape, _contiguous_stride(output_shape), dtype),
        grad_out=_FakeNode(output_shape, _contiguous_stride(output_shape), dtype),
        grad_logsumexp=None,
    )
    return inputs


def _backward_config_result(inputs):
    with (
        V.set_graph_handler(_fake_graph()),
        mock.patch.object(
            flex_flydsl_attention, "_is_gfx950_device", return_value=True
        ),
    ):
        return _get_flydsl_flex_attention_backward_config(**inputs)


class _TensorIndexMode(torch.overrides.TorchFunctionMode):
    """Trace ``capture[index]`` with 0-d index tensors to aten.index, as Dynamo does."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.__getitem__:
            index = args[1] if isinstance(args[1], tuple) else (args[1],)
            if all(isinstance(item, torch.Tensor) for item in index):
                return torch.ops.aten.index.Tensor(args[0], list(index))
        return func(*args, **(kwargs or {}))


def _mask_graph(fn, *captures):
    indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
    with _TensorIndexMode():
        return make_fx(fn)(*indices, *captures)


def _capture_node(capture):
    shape = tuple(capture.shape) or (1,)
    return _FakeNode(shape, capture.stride() or (1,), dtype=capture.dtype)


def _lower_mask(fn, *captures):
    with V.set_graph_handler(_fake_graph()):
        return lower_flydsl_mask_graph(
            _mask_graph(fn, *captures), [_capture_node(c) for c in captures]
        )


_REFERENCE_MASK_KINDS = {
    "bool": torch.bool,
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
}
_REFERENCE_MASK_CONSTANTS = {
    "const_bool": torch.bool,
    "const_i32": torch.int32,
    "const_i64": torch.int64,
    "const_f32": torch.float32,
}
_REFERENCE_MASK_BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": lambda a, b: torch.div(a, b, rounding_mode="floor"),
    "remainder": torch.remainder,
    "fmod": torch.fmod,
    "div": operator.truediv,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "eq": operator.eq,
    "ne": operator.ne,
    "and": torch.bitwise_and,
    "or": torch.bitwise_or,
    "xor": torch.bitwise_xor,
    "shl": torch.bitwise_left_shift,
    "shr": torch.bitwise_right_shift,
    "min": torch.minimum,
    "max": torch.maximum,
}


def _load_reference_mask_element(capture, offset, dtype):
    """Read a capture the way the kernels' bounds-checked buffer views do."""
    if not isinstance(capture, torch.Tensor):
        return torch.full_like(offset, capture, dtype=torch.int32)
    extent = 1 + sum(
        (size - 1) * stride for size, stride in zip(capture.shape, capture.stride())
    )
    flat = capture.as_strided((extent,), (1,))
    element = torch.where(
        (offset >= 0) & (offset < extent),
        flat[offset.clamp(0, extent - 1)],
        torch.zeros((), dtype=flat.dtype),
    )
    if dtype == torch.bool:
        return element.to(torch.bool)
    if dtype.is_floating_point:
        return element.to(torch.float32)
    return element.to(torch.int64 if dtype == torch.int64 else torch.int32)


def _evaluate_mask_program(program, b, h, q, kv, captures=()):
    """Python reference for the kernels' mask bytecode evaluator; ``captures``
    are tensors, or ints every load of that buffer returns."""
    values = [torch.as_tensor(value).to(torch.int32) for value in (b, h, q, kv)]
    for instruction in program.instructions:
        op = instruction[0]
        if op in _REFERENCE_MASK_CONSTANTS:
            value = torch.tensor(instruction[1], dtype=_REFERENCE_MASK_CONSTANTS[op])
        elif op in ("load_i32", "load"):
            buffer_index, index_ids = instruction[1:3]
            dtype = torch.int32 if op == "load_i32" else getattr(torch, instruction[3])
            offset = sum(
                values[index].long() * stride
                for index, stride in zip(
                    index_ids, program.buffer_strides[buffer_index]
                )
            )
            offset = torch.remainder(offset + (1 << 31), 1 << 32) - (1 << 31)
            value = _load_reference_mask_element(captures[buffer_index], offset, dtype)
        elif op == "cast":
            source, kind = values[instruction[1]], instruction[2]
            if kind in ("bf16", "fp16"):
                rounded = torch.bfloat16 if kind == "bf16" else torch.float16
                value = source.to(torch.float32).to(rounded).to(torch.float32)
            else:
                value = source.to(_REFERENCE_MASK_KINDS[kind])
        elif op == "where":
            condition, true_value, false_value = (
                values[index] for index in instruction[1:]
            )
            value = torch.where(condition, true_value, false_value)
        elif op == "not":
            value = torch.bitwise_not(values[instruction[1]])
        elif op == "neg":
            value = -values[instruction[1]]
        elif op == "abs":
            value = torch.abs(values[instruction[1]])
        else:
            lhs, rhs = values[instruction[1]], values[instruction[2]]
            value = _REFERENCE_MASK_BINARY[op](lhs, rhs)
        values.append(value)
    result = values[program.output]
    return bool(result) if result.dim() == 0 else result.to(torch.bool)


def _attention_grads_fp64(query, key, value, grad_output, mask_mod):
    """Eager fp64 (dQ, dK, dV) for small shapes; GQA by repeating K/V heads."""
    group = query.size(1) // key.size(1)
    q, k, v, do = (t.detach().double() for t in (query, key, value, grad_output))
    k = k.repeat_interleave(group, dim=1).requires_grad_()
    v = v.repeat_interleave(group, dim=1).requires_grad_()
    q = q.requires_grad_()
    batch, heads, seq_q, _ = q.shape
    seq_kv = k.size(2)
    index = [
        torch.arange(n, device=q.device).view(*shape)
        for n, shape in (
            (batch, (-1, 1, 1, 1)),
            (heads, (1, -1, 1, 1)),
            (seq_q, (1, 1, -1, 1)),
            (seq_kv, (1, 1, 1, -1)),
        )
    ]
    allowed = torch.broadcast_to(mask_mod(*index), (batch, heads, seq_q, seq_kv))
    scores = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
    probability = scores.masked_fill(~allowed, -float("inf")).softmax(-1)
    # Rows without a key give zero output and gradients, as in flex attention.
    probability = torch.where(allowed.any(-1, keepdim=True), probability, 0)
    (probability @ v).backward(do)
    shape = (batch, heads // group, group)
    return (
        q.grad,
        k.grad.view(*shape, seq_kv, -1).sum(2),
        v.grad.view(*shape, seq_kv, -1).sum(2),
    )


def _log2_lse_backward(
    query, key, value, out, lse2, grad_out, grad_lse2, scale, allowed, score_mod
):
    """(dQ, dK, dV) using the FlyDSL backward's log2-LSE math; K/V per Q head."""
    pre_mod = ((query @ key.transpose(-2, -1)) * scale).requires_grad_()
    post_mod = score_mod(pre_mod)
    probability = torch.exp2(post_mod.detach() * math.log2(math.e) - lse2[..., None])
    probability = probability.masked_fill(~allowed, 0.0)
    delta = (grad_out * out).sum(-1) - grad_lse2 * math.log2(math.e)
    grad_post = probability * (grad_out @ value.transpose(-2, -1) - delta[..., None])
    (grad_scores,) = torch.autograd.grad(post_mod, pre_mod, grad_post)
    return (
        grad_scores @ key * scale,
        grad_scores.transpose(-2, -1) @ query * scale,
        probability.transpose(-2, -1) @ grad_out,
    )


def _large_logit_backward(value, dtype, *, prescale_query, fused_backward):
    """(P, dS) of the FlyDSL exponent math for Q = K = V = value, one key, dO = 1;
    the D = 128, scale = 1/8 dot products are exact, so only the exponent's
    rounding can move P off 1."""
    scale_log2 = torch.tensor(0.125 * math.log2(math.e))
    query = torch.full((1, 128), value, dtype=dtype)
    key = value_tensor = query.float()
    scores = query.float() @ key.T
    if prescale_query:
        forward = (query.float() * scale_log2).to(dtype).float() @ key.T
    else:
        forward = scores * scale_log2
    maximum = forward.amax(-1, keepdim=True)
    weights = torch.exp2(forward - maximum)
    lse = maximum + weights.sum(-1, keepdim=True).log2()
    output = (weights @ value_tensor / weights.sum(-1, keepdim=True)).to(dtype)
    if fused_backward:
        exponent = (scores.double() * scale_log2.double() - lse.double()).float()
    else:
        exponent = scores * scale_log2 - lse
    probability = torch.exp2(exponent)
    grad_output = torch.ones_like(value_tensor)
    delta = (grad_output * output.float()).sum(-1, keepdim=True)
    grad_scores = probability * (grad_output @ value_tensor.T - delta)
    return probability.item(), grad_scores.item()


def _has_gfx950_flydsl():
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(
            torch.cuda.get_device_properties(0),
            "gcnArchName",
            "",
        ).split(":", 1)[0]
        == "gfx950"
        and flydsl_utils.runtime_available()
    )


def _fake_choice_result(inputs, graph=None):
    choices = []
    with (
        V.set_graph_handler(_fake_graph() if graph is None else graph),
        mock.patch(
            "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
            return_value=True,
        ),
        mock.patch.object(
            flex_flydsl_forward_template, "maybe_append_choice"
        ) as append,
    ):
        _, reason = maybe_append_flydsl_flex_attention_choice(
            choices,
            layout=mock.Mock(),
            **inputs,
        )
    kwargs = append.call_args.kwargs if append.called else None
    return kwargs, reason


@torch.library.custom_op("flydsl_flex_test::packed_decode_union", mutates_args=())
def _packed_decode_union_op(
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: torch.Tensor,
    full_kv_indices: torch.Tensor,
    num_kv_heads: int,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise AssertionError("only the Inductor lowering under test may run this op")


@_packed_decode_union_op.register_fake
def _(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, hkv, kv_blocks):
    batch, _, q_blocks = kv_num_blocks.shape
    counts = kv_num_blocks.new_empty(batch, hkv, q_blocks)
    indices = kv_indices.new_empty(batch, hkv, q_blocks, kv_blocks)
    return counts, indices, counts.clone(), indices.clone()


@torch._inductor.lowering.register_lowering(
    torch.ops.flydsl_flex_test.packed_decode_union.default
)
def _lower_packed_decode_union(
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, hkv, kv_blocks
):
    return flex_flydsl_attention._create_packed_decode_union_metadata(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        num_kv_heads=hkv,
        num_kv_blocks=kv_blocks,
    )


def _reference_packed_decode_union(
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, num_kv_heads
):
    batch, num_q_heads, q_blocks = kv_num_blocks.shape
    group = num_q_heads // num_kv_heads
    partial_lists = {}
    full_lists = {}
    for b in range(batch):
        for kv_head in range(num_kv_heads):
            for q in range(q_blocks):
                heads = range(kv_head * group, (kv_head + 1) * group)
                partial_sets = [
                    set(kv_indices[b, h, q, : kv_num_blocks[b, h, q]].tolist())
                    for h in heads
                ]
                full_sets = [
                    set(
                        full_kv_indices[b, h, q, : full_kv_num_blocks[b, h, q]].tolist()
                    )
                    for h in heads
                ]
                always_full = set.intersection(*full_sets)
                listed = set.union(*full_sets, *partial_sets)
                full_lists[b, kv_head, q] = sorted(always_full)
                partial_lists[b, kv_head, q] = sorted(listed - always_full)
    return partial_lists, full_lists


@torch.library.custom_op("flydsl_flex_test::split_block_mask", mutates_args=())
def _split_block_mask_op(
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: torch.Tensor,
    full_kv_indices: torch.Tensor,
    q_factor: int,
    kv_factor: int,
    num_q_blocks: int,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise AssertionError("only the Inductor lowering under test may run this op")


@_split_block_mask_op.register_fake
def _(counts, indices, full_counts, full_indices, q_factor, kv_factor, q_blocks, _):
    batch, heads, _ = counts.shape
    return (
        counts.new_empty(batch, heads, q_blocks),
        indices.new_empty(batch, heads, q_blocks, indices.size(-1) * kv_factor),
        counts.new_empty(batch, heads, q_blocks),
        indices.new_empty(batch, heads, q_blocks, full_indices.size(-1) * kv_factor),
    )


@torch._inductor.lowering.register_lowering(
    torch.ops.flydsl_flex_test.split_block_mask.default
)
def _lower_split_block_mask(
    counts, indices, full_counts, full_indices, q_factor, kv_factor, q_blocks, kv_blocks
):
    return flex_flydsl_attention._expand_block_mask_metadata(
        counts,
        indices,
        full_counts,
        full_indices,
        q_factor=q_factor,
        kv_factor=kv_factor,
        num_q_blocks=q_blocks,
        num_kv_blocks=kv_blocks,
    )


@torch.library.custom_op("flydsl_flex_test::merge_block_mask", mutates_args=())
def _merge_block_mask_op(
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: torch.Tensor,
    full_kv_indices: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    seq_q: int,
    seq_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise AssertionError("only the Inductor lowering under test may run this op")


def _merged_shapes(batch, heads, q_block_size, kv_block_size, seq_q, seq_kv):
    q_blocks, kv_blocks = -(-seq_q // 128), -(-seq_kv // 128)
    q_step, kv_step = map(
        flex_flydsl_attention._state_block_size, (q_block_size, kv_block_size)
    )
    states = (
        batch,
        heads,
        -(-q_blocks * 128 // q_step),
        -(-kv_blocks * 128 // kv_step),
    )
    return (batch, heads, q_blocks), (batch, heads, q_blocks, kv_blocks), states


@_merge_block_mask_op.register_fake
def _(counts, indices, full_counts, full_indices, q_size, kv_size, seq_q, seq_kv):
    count_shape, index_shape, state_shape = _merged_shapes(
        *counts.shape[:2], q_size, kv_size, seq_q, seq_kv
    )
    return (
        counts.new_empty(count_shape),
        indices.new_empty(index_shape),
        counts.new_empty(count_shape),
        indices.new_empty(index_shape),
        counts.new_empty(state_shape, dtype=torch.int8),
    )


@torch._inductor.lowering.register_lowering(
    torch.ops.flydsl_flex_test.merge_block_mask.default
)
def _lower_merge_block_mask(
    counts, indices, full_counts, full_indices, q_size, kv_size, seq_q, seq_kv
):
    return flex_flydsl_attention._merge_block_mask_metadata(
        counts,
        indices,
        full_counts,
        full_indices,
        block_sizes=(q_size, kv_size),
        num_q_blocks=-(-seq_q // 128),
        num_kv_blocks=-(-seq_kv // 128),
        seq_kv=seq_kv,
        wide_counts=-(-seq_kv // 128) > flex_flydsl_attention.MAX_UNION_KV_BLOCKS,
    )


def _dense_block_states(counts, indices, full_counts, full_indices, kv_blocks):
    """BLOCK_STATE_* per listed (b, h, q block, kv block) of BlockMask lists;
    blocks at or past ``kv_blocks`` are dropped."""
    states = torch.zeros(*counts.shape, kv_blocks + 1, dtype=torch.int8)
    for lists, value in (((counts, indices), 1), ((full_counts, full_indices), 2)):
        count, index = lists
        slots = torch.arange(index.size(-1))
        listed = (slots < count[..., None]) & (index >= 0) & (index < kv_blocks)
        states.scatter_(-1, torch.where(listed, index, kv_blocks).long(), value)
        states[..., kv_blocks] = 0
    return states[..., :kv_blocks]


def _reference_merged_block_mask(metadata, block_sizes, seq_q, seq_kv):
    """(partial lists, full lists, padded user states) of a merged BlockMask;
    the states repeat over blocks of ``_state_block_size``."""
    user_kv_blocks = -(-seq_kv // block_sizes[1])
    user = _dense_block_states(*metadata, user_kv_blocks)
    batch, heads, user_q_blocks, _ = user.shape
    _, index_shape, state_shape = _merged_shapes(
        batch, heads, *block_sizes, seq_q, seq_kv
    )
    expanded = user
    for dim, size in ((2, block_sizes[0]), (3, block_sizes[1])):
        repeats = size // flex_flydsl_attention._state_block_size(size)
        expanded = expanded.repeat_interleave(repeats, dim)
    expanded = expanded[:, :, : state_shape[2], : state_shape[3]]
    states = torch.zeros(state_shape, dtype=torch.int8)
    states[:, :, : expanded.size(2), : expanded.size(3)] = expanded
    partial_lists, full_lists = {}, {}
    for b, h, q, kv in itertools.product(*map(range, index_shape)):
        rows, columns = (
            range(block * 128 // size, -(-(block + 1) * 128 // size))
            for block, size in ((q, block_sizes[0]), (kv, block_sizes[1]))
        )
        covered = user[b, h, rows.start : rows.stop, columns.start : columns.stop]
        partial_lists.setdefault((b, h, q), [])
        full_lists.setdefault((b, h, q), [])
        if (covered == 2).all():
            full_lists[b, h, q].append(kv)
        elif (covered != 0).any():
            partial_lists[b, h, q].append(kv)
    return partial_lists, full_lists, states


def _normalized_attention(scores, v):
    row_max = scores.detach().amax(-1, keepdim=True).nan_to_num(0.0, neginf=0.0)
    weights = torch.exp(scores - row_max)
    return (weights @ v) / weights.sum(-1, keepdim=True).clamp_min(1e-30)


def _masked_scores(q, k, v, keep, scale, score_mod=None):
    """fp32 scores of BHSD tensors, -inf where not kept, and V per query head."""
    q, k, v = (t.float() for t in (q, k, v))
    group = q.size(1) // k.size(1)
    k, v = (
        t.expand(q.size(0), -1, -1, -1).repeat_interleave(group, dim=1) for t in (k, v)
    )
    scores = q @ k.transpose(-1, -2) * scale
    if score_mod is not None:
        scores = score_mod(
            scores,
            *(
                torch.arange(size, device=q.device).view(
                    [-1 if dim == axis else 1 for dim in range(4)]
                )
                for axis, size in enumerate(scores.shape)
            ),
        )
    return scores.masked_fill(~keep, float("-inf")), v


def _random_block_lists(generator, batch, heads, q_blocks, kv_blocks, garbage=-7):
    """Disjoint, unordered full/partial lists with garbage past each count."""
    shape = (batch, heads, q_blocks)
    counts = torch.zeros(shape, dtype=torch.int32)
    full_counts = torch.zeros_like(counts)
    indices = torch.full((*shape, kv_blocks), garbage, dtype=torch.int32)
    full_indices = torch.full_like(indices, garbage)
    for b, h, q in torch.cartesian_prod(*(torch.arange(n) for n in shape)).tolist():
        order = torch.randperm(kv_blocks, generator=generator)
        listed = int(torch.randint(0, kv_blocks + 1, (), generator=generator))
        num_full = int(torch.randint(0, listed + 1, (), generator=generator))
        full_counts[b, h, q] = num_full
        counts[b, h, q] = listed - num_full
        full_indices[b, h, q, :num_full] = order[:num_full]
        indices[b, h, q, : listed - num_full] = order[num_full:listed]
    return counts, indices, full_counts, full_indices


def _add_duplicate_listings(generator, lists):
    """Fill free partial slots with copies of listed full and partial blocks.
    A block listed as both full and partial counts as full."""
    counts, indices, full_counts, full_indices = (t.clone() for t in lists)
    for b, h, q in itertools.product(*map(range, counts.shape)):
        count = int(counts[b, h, q])
        listed = torch.cat(
            (full_indices[b, h, q, : full_counts[b, h, q]], indices[b, h, q, :count])
        )
        free = indices.size(-1) - count
        if not listed.numel() or not free:
            continue
        copies = int(torch.randint(1, free + 1, (), generator=generator))
        picks = torch.randint(0, listed.numel(), (copies,), generator=generator)
        indices[b, h, q, count : count + copies] = listed[picks]
        counts[b, h, q] = count + copies
    return counts, indices, full_counts, full_indices


def _sorted_block_lists(lists):
    """``lists`` with the listed blocks of each row in ascending order. Triton
    can mis-mask a block that extends past Sk unless it is last in its row."""
    sorted_lists = []
    for counts, indices in (lists[:2], lists[2:]):
        listed = torch.arange(indices.size(-1)) < counts[..., None]
        last = torch.iinfo(indices.dtype).max
        ordered = torch.where(listed, indices, last).sort(-1).values
        sorted_lists += [counts, torch.where(listed, ordered, 0)]
    return sorted_lists


def _triton_tile_options(block_sizes):
    """Triton flex tiles that divide BlockMask ``block_sizes`` its defaults do
    not fit."""
    if all(
        size >= 16 and (size & (size - 1) == 0 or size % 128 == 0)
        for size in block_sizes
    ):
        return {}
    common = math.gcd(*block_sizes)
    tile = min(16, common & -common)
    names = ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
    return dict.fromkeys(names, tile)


def _random_block_mask_metadata(
    generator, batch, heads, num_kv_blocks, capacity, garbage=99
):
    counts = torch.zeros(batch, heads, 1, dtype=torch.int32)
    full_counts = torch.zeros_like(counts)
    indices = torch.full((batch, heads, 1, capacity), garbage, dtype=torch.int32)
    full_indices = torch.full_like(indices, garbage)
    for b in range(batch):
        for h in range(heads):
            order = torch.randperm(num_kv_blocks, generator=generator)
            listed = int(torch.randint(0, capacity + 1, (), generator=generator))
            num_full = int(torch.randint(0, listed + 1, (), generator=generator))
            full = order[:num_full].sort().values
            partial = order[num_full:listed].sort().values
            full_counts[b, h, 0] = full.numel()
            counts[b, h, 0] = partial.numel()
            full_indices[b, h, 0, : full.numel()] = full
            # Unsorted partial lists are legal BlockMask metadata.
            indices[b, h, 0, : partial.numel()] = partial.flip(0)
    return counts, indices, full_counts, full_indices


def _dynamic_case(shapes, mask=None, score_mod=False, graphs=1, heads=(4, 2)):
    return (*heads, shapes, mask, score_mod, graphs)


_PREFILLS = [(3, 1000, 1500), (5, 600, 900)]
# (Q heads, KV heads, (B, Sq, Sk) of each call, mask, score_mod, compiled graphs).
# Kernel choices only change between decode and prefill.
_DYNAMIC_SHAPE_CASES = {
    "ragged_gqa": _dynamic_case([*_PREFILLS, (4, 300, 2001)]),
    "decode_prefill": _dynamic_case(
        [(3, 4, 900), (5, 7, 1300), (3, 1000, 900), (4, 700, 1300)],
        "causal",
        graphs=2,
        heads=(8, 2),
    ),
    "captures": _dynamic_case(_PREFILLS, "document", score_mod=True),
    "small_blocks": _dynamic_case(_PREFILLS, "causal_64"),
    "unaligned_decode": _dynamic_case(
        [(3, 4, 900), (5, 7, 1300)], "causal_24x192", heads=(8, 2)
    ),
    "kv_batch_1": _dynamic_case(_PREFILLS, "kv_batch_1"),
    # 512 CTAs of 256 rows and 16 to 32 KV blocks take the paired prefill.
    "paired": _dynamic_case(
        [(2, 2048, 2048), (4, 1024, 3072)], "causal", heads=(32, 8)
    ),
    # Four document captures, the 64x64 block states and a score_mod capture.
    "capture_table": _dynamic_case(_PREFILLS, "documents_64", score_mod=True),
    # The causal dual-wave forward with Sk % 128 below 64.
    "dualwave_tail": _dynamic_case(
        [(3, 1000, 1050), (2, 700, 1960)], "dualwave_causal", heads=(8, 2)
    ),
}


_DYNAMIC_BLOCK_SIZES = {
    "causal_64": 64,
    "documents_64": 64,
    "causal_24x192": (24, 192),
}


def _dynamic_shape_runs(case, device, backend, dynamic, score_mods=None):
    """Outputs and Q/K/V gradients of each shape of ``case`` through one compile;
    a None ``backend`` runs eagerly in fp64. Appends the score_mod to ``score_mods``."""
    q_heads, kv_heads, shapes, mask, use_score_mod, _ = _DYNAMIC_SHAPE_CASES[case]
    documents = (torch.arange(4096, device=device) * 7 // 900).to(torch.int32)
    d0, d1, d2, d3 = (
        (torch.arange(4096, device=device) * 7 // length).to(torch.int32)
        for length in (500, 700, 900, 1100)
    )
    bias = torch.linspace(-0.5, 0.5, q_heads, device=device)
    mask_mods = {
        "causal": lambda b, h, q, kv: q >= kv,
        "causal_64": lambda b, h, q, kv: q >= kv,
        "causal_24x192": lambda b, h, q, kv: q >= kv,
        "document": lambda b, h, q, kv: documents[q] == documents[kv],
        "documents_64": lambda b, h, q, kv: (
            ((d0[q] == d0[kv]) | (d1[q] == d1[kv]))
            & ((d2[q] == d2[kv]) | (d3[q] == d3[kv]))
        ),
        "kv_batch_1": lambda b, h, q, kv: kv >= 0,
        "dualwave_causal": lambda b, h, q, kv: q >= kv,
    }
    dualwave = mask == "dualwave_causal"

    def score_mod(score, b, h, q, kv):
        return score + bias[h]

    if score_mods is not None:
        score_mods.append(score_mod if use_score_mod else None)

    block_size = _DYNAMIC_BLOCK_SIZES.get(mask, 128)
    kernel_options = None if backend is None else {"BACKEND": backend}
    if backend == "TRITON" and isinstance(block_size, tuple):
        kernel_options.update(_triton_tile_options(block_size))
    patches = []
    if dualwave and backend == "FLYDSL":
        # Offer only the dual-wave candidate.
        kernel_options["FLYDSL_DUALWAVE_CAUSAL"] = True
        patches = [
            torch._inductor.config.patch(flydsl_enable_autotuning=True),
            mock.patch.object(
                flex_flydsl_attention,
                "maybe_append_flydsl_flex_attention_choice",
                return_value=(True, ""),
            ),
        ]

    def attention(q, k, v, block_mask):
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=score_mod if use_score_mod else None,
            scale=128**-0.5,
            enable_gqa=True,
            kernel_options=kernel_options,
        )

    torch._dynamo.reset()
    compiled = torch.compile(
        attention,
        fullgraph=True,
        dynamic=dynamic,
        backend="eager" if backend is None else "inductor",
    )
    results = []
    for batch, seq_q, seq_kv in shapes:
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=seq_q + seq_kv,
        )
        if mask == "kv_batch_1":
            key, value = key[:1].clone(), value[:1].clone()
        if dualwave:
            # BHSD views of [B, S, H, D] memory.
            query, key, value = (
                t.transpose(1, 2).contiguous().transpose(1, 2)
                for t in (query, key, value)
            )
        block_mask = None
        if mask is not None:
            block_mask = create_block_mask(
                mask_mods[mask],
                batch if mask == "kv_batch_1" else 1,
                1,
                seq_q,
                seq_kv,
                device=device,
                BLOCK_SIZE=block_size,
            )
        dtype = torch.float64 if backend is None else query.dtype
        inputs = [
            t.detach().to(dtype, copy=True).requires_grad_()
            for t in (query, key, value)
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            output = compiled(*inputs, block_mask)
        torch.manual_seed(seq_q)
        output.backward(torch.randn_like(output, dtype=query.dtype).to(dtype))
        results.append((output.detach(), *(t.grad for t in inputs)))
    return results


def _dualwave_mod_case(
    batch,
    q_heads,
    kv_heads,
    seq_q,
    seq_kv,
    mask=None,
    score_mod=None,
    block=128,
    dtype=torch.bfloat16,
    dims=(128, 128),
):
    return batch, q_heads, kv_heads, seq_q, seq_kv, mask, score_mod, block, dtype, dims


# Dual-wave forwards with mods; "kv_batch_1" cases read one K/V batch.
_DUALWAVE_MOD_CASES = {
    "alibi": _dualwave_mod_case(1, 8, 2, 1000, 1050, score_mod="alibi"),
    "relative_bias_table": _dualwave_mod_case(
        1, 4, 4, 512, 512, score_mod="relative_bias_table"
    ),
    "random_lists_per_kv_head": _dualwave_mod_case(2, 8, 2, 640, 1024, "random"),
    "captured_bias_qk64_v128": _dualwave_mod_case(
        2, 4, 4, 512, 640, score_mod="bias", dtype=torch.float16, dims=(64, 128)
    ),
    "window_gqa_cross_d96": _dualwave_mod_case(
        2, 8, 2, 333, 1500, "window", dtype=torch.float16, dims=(96, 96)
    ),
    "softcap_fp16_qk128_v256": _dualwave_mod_case(
        1, 8, 8, 1024, 1024, score_mod="softcap", dtype=torch.float16, dims=(128, 256)
    ),
    "causal_64_d256": _dualwave_mod_case(
        1, 4, 2, 1024, 1024, "causal", block=64, dims=(256, 256)
    ),
    "four_documents_d192": _dualwave_mod_case(
        1, 4, 4, 512, 768, "documents", dims=(192, 192)
    ),
    "kv_batch_1_qk192_v128": _dualwave_mod_case(
        3, 4, 2, 512, 768, "document", dims=(192, 128)
    ),
}


def _dualwave_wide(dims):
    """Whether (QK, V) ``dims`` need FLYDSL_DUALWAVE_WIDE."""
    return max(dims) > 128


def _dualwave_mod_inputs(case, device):
    """BHSD views of [B, S, H, D] Q/K/V, the BlockMask (or None), the BlockMask
    for Triton, and the score_mod of a ``_DUALWAVE_MOD_CASES`` case."""
    batch, q_heads, kv_heads, seq_q, seq_kv, mask, score_mod, block, dtype, dims = (
        _DUALWAVE_MOD_CASES[case]
    )
    qk_dim, v_dim = dims
    query, key, value = _make_qkv(
        device=device,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_dim=qk_dim,
        v_dim=v_dim,
        seed=seq_q + seq_kv,
        transposed=True,
        dtype=dtype,
    )
    if case.startswith("kv_batch_1"):
        key, value = key[:1], value[:1]
    torch.manual_seed(11)
    if score_mod == "alibi":
        score_mod = _alibi_mod
    elif score_mod == "softcap":
        score_mod = _softcap_mod
    elif score_mod == "bias":
        bias = torch.randn(batch, q_heads, seq_q, seq_kv, device=device)
        score_mod = lambda score, b, h, q, kv: score + bias[b, h, q, kv]  # noqa: E731
    elif score_mod == "relative_bias_table":
        table = torch.randn(q_heads, seq_q + seq_kv - 1, device=device).bfloat16()
        score_mod = lambda score, b, h, q, kv: score + table[h, q - kv + seq_kv - 1]  # noqa: E731
    length = max(seq_q, seq_kv)
    documents = [
        (torch.arange(length, device=device) * 7 // size).to(torch.int64)
        for size in (900, 500, 700, 1100)
    ]
    mask_mods = {
        "causal": lambda b, h, q, kv: q >= kv,
        "document": lambda b, h, q, kv: (documents[0][q] == documents[0][kv])
        & (q >= kv),
        "window": lambda b, h, q, kv: (q >= kv) & (q - kv < 300),
        "documents": lambda b, h, q, kv: functools.reduce(
            operator.and_, (doc[q] == doc[kv] for doc in documents[1:]), q >= kv
        )
        | (documents[0][q] == documents[0][kv]),
        # Disagrees with the random lists and depends on the head.
        "random": lambda b, h, q, kv: (q + 2 * kv + h) % 3 != 0,
    }
    if mask is None:
        return query, key, value, None, None, score_mod
    if mask != "random":
        block_mask = create_block_mask(
            mask_mods[mask],
            batch if case.startswith("kv_batch_1") else 1,
            1,
            seq_q,
            seq_kv,
            device=device,
            BLOCK_SIZE=block,
        )
        return query, key, value, block_mask, block_mask, score_mod
    # A per-KV-head BlockMask serves every query head of its group.
    generator = torch.Generator().manual_seed(seq_q)
    lists = _random_block_lists(
        generator, batch, kv_heads, -(-seq_q // block), -(-seq_kv // block), garbage=0
    )

    def from_lists(lists):
        return BlockMask.from_kv_blocks(
            *(t.to(device) for t in lists),
            BLOCK_SIZE=block,
            mask_mod=mask_mods["random"],
            seq_lengths=(seq_q, seq_kv),
        )

    group = q_heads // kv_heads
    reference = from_lists(
        _sorted_block_lists([t.repeat_interleave(group, dim=1) for t in lists])
    )
    return query, key, value, from_lists(lists), reference, score_mod


def _symbolic_graph(*hints):
    """A graph handler whose sizevars reason about backed size symbols with
    ``hints``; returns it and the symbols."""
    from torch._dynamo.source import ConstantSource
    from torch._inductor.sizevars import SizeVarAllocator
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    shape_env = ShapeEnv()
    symbols = [
        shape_env.create_symbol(
            hint, ConstantSource(f"size_{index}"), dynamic_dim=DimDynamic.DYNAMIC
        )
        for index, hint in enumerate(hints)
    ]
    return SimpleNamespace(sizevars=SizeVarAllocator(shape_env)), symbols


def _symbolic_node(size, dtype=torch.bfloat16):
    """A contiguous fake node of sympy ``size``."""
    stride = [sympy.Integer(1)] * len(size)
    for index in range(len(size) - 2, -1, -1):
        stride[index] = stride[index + 1] * size[index + 1]
    return _FakeNode(size, stride, dtype, numel=sympy.Mul(*size))


def _symbolic_forward_inputs(
    batch, seq_q, seq_kv, *, q_heads=8, kv_heads=2, kv_batch=None
):
    from torch.utils._sympy.functions import CeilDiv

    inputs = _supported_fake_forward_inputs()
    kv_batch = batch if kv_batch is None else kv_batch
    q_blocks = (
        1 if not isinstance(seq_q, sympy.Expr) and seq_q < 128 else CeilDiv(seq_q, 128)
    )
    counts = [1, 1, q_blocks]
    indices = [*counts, CeilDiv(seq_kv, 128)]
    inputs.update(
        query=_symbolic_node([batch, q_heads, seq_q, 128]),
        key=_symbolic_node([kv_batch, kv_heads, seq_kv, 128]),
        value=_symbolic_node([kv_batch, kv_heads, seq_kv, 128]),
        logsumexp=_symbolic_node([batch, q_heads, seq_q], torch.float32),
        max_scores=_symbolic_node([batch, q_heads, seq_q], torch.float32),
        kv_num_blocks=_symbolic_node(counts, torch.int32),
        kv_indices=_symbolic_node(indices, torch.int32),
        full_kv_num_blocks=_symbolic_node(counts, torch.int32),
        full_kv_indices=_symbolic_node(indices, torch.int32),
        mask_graph=SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q >= kv)
        ),
    )
    return inputs


def _bshd_stride(batch, heads, sequence, dimension):
    del batch
    return [sequence * heads * dimension, dimension, heads * dimension, 1]


def _dualwave_fake_inputs(
    *,
    batch=1,
    q_heads=8,
    kv_heads=None,
    seq_q=1024,
    seq_kv=None,
    qk_dim=128,
    v_dim=None,
    dtype=torch.bfloat16,
    bshd=True,
    mask_fn=lambda b, h, q, kv: q >= kv,
    block_mask_is_dense=False,
    partial_width=None,
    full_width=None,
    mask_batch=1,
    mask_heads=1,
):
    """Fake dual-wave gate inputs; the default metadata is a causal 128x128
    ``create_block_mask``."""
    kv_heads = q_heads if kv_heads is None else kv_heads
    seq_kv = seq_q if seq_kv is None else seq_kv
    v_dim = qk_dim if v_dim is None else v_dim

    def node(heads, sequence, dimension):
        size = (batch, heads, sequence, dimension)
        stride = _bshd_stride(*size) if bshd else _contiguous_stride(size)
        return _FakeNode(size, stride, dtype)

    def meta(size):
        return _FakeNode(size, _contiguous_stride(size), torch.int32)

    q_blocks = -(-seq_q // 128)
    kv_blocks = -(-seq_kv // 128)
    if block_mask_is_dense:
        mask_fn = lambda b, h, q, kv: torch.full((), True)  # noqa: E731
        partial_width = 1 if partial_width is None else partial_width
    partial_width = kv_blocks if partial_width is None else partial_width
    full_width = kv_blocks if full_width is None else full_width
    counts = [mask_batch, mask_heads, q_blocks]
    return {
        "query": node(q_heads, seq_q, qk_dim),
        "key": node(kv_heads, seq_kv, qk_dim),
        "value": node(kv_heads, seq_kv, v_dim),
        "logsumexp": meta([batch, q_heads, seq_q]),
        "max_scores": meta([batch, q_heads, seq_q]),
        "kv_num_blocks": meta(counts),
        "kv_indices": meta([*counts, partial_width]),
        "full_kv_num_blocks": meta(counts),
        "full_kv_indices": meta([*counts, full_width]),
        "subgraph": SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        ),
        "mask_graph": SimpleNamespace(graph_module=_mask_graph(mask_fn)),
        "score_mod_other_buffers": [],
        "mask_mod_other_buffers": [],
        "scale": qk_dim**-0.5,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
        "block_mask_is_dense": block_mask_is_dense,
    }


def _dualwave_choice_result(inputs, *, autotuning=True, graph=None, **append_kwargs):
    choices = []
    with (
        V.set_graph_handler(_fake_graph() if graph is None else graph),
        mock.patch.object(
            flex_flydsl_attention, "_is_gfx950_device", return_value=True
        ),
        mock.patch.object(
            flex_flydsl_dualwave_forward_template, "maybe_append_choice"
        ) as append,
        torch._inductor.config.patch(flydsl_enable_autotuning=autotuning),
    ):
        _, reason = maybe_append_flydsl_dualwave_forward_choice(
            choices, layout=mock.Mock(), **inputs, **append_kwargs
        )
    kwargs = append.call_args.kwargs if append.called else None
    return kwargs, reason


def _make_qkv(
    *,
    device="cuda",
    batch=1,
    q_heads=2,
    kv_heads=None,
    seq_q=256,
    seq_kv=None,
    qk_dim=128,
    v_dim=128,
    seed=0,
    transposed=False,
    dtype=torch.bfloat16,
):
    kv_heads = q_heads if kv_heads is None else kv_heads
    seq_kv = seq_q if seq_kv is None else seq_kv
    torch.manual_seed(seed)

    def make(heads, sequence, dimension):
        if not transposed:
            return torch.randn(
                batch,
                heads,
                sequence,
                dimension,
                device=device,
                dtype=dtype,
            )
        return torch.randn(
            batch,
            sequence,
            heads,
            dimension,
            device=device,
            dtype=dtype,
        ).transpose(1, 2)

    return (
        make(q_heads, seq_q, qk_dim),
        make(kv_heads, seq_kv, qk_dim),
        make(kv_heads, seq_kv, v_dim),
    )


def _listed_blocks(counts, indices, kv_blocks):
    """The [..., Q blocks, KV blocks] map of the blocks of one BlockMask list."""
    listed = torch.arange(indices.size(-1), device=indices.device) < counts[..., None]
    listed &= (indices >= 0) & (indices < kv_blocks)
    dense = torch.zeros(
        *indices.shape[:-1], kv_blocks + 1, dtype=torch.bool, device=indices.device
    )
    dense.scatter_(-1, torch.where(listed, indices, kv_blocks).long(), True)
    return dense[..., :kv_blocks]


def _with_listed_blocks(block_mask):
    """``block_mask`` with its block lists folded into its mask_mod, which eager
    flex_attention evaluates alone: keys of full blocks are kept, of partial
    blocks per the mask_mod, of other blocks dropped."""
    if block_mask is None:
        return None
    q_block, kv_block = block_mask.BLOCK_SIZE
    kv_blocks = -(-block_mask.seq_lengths[1] // kv_block)
    partial = _listed_blocks(block_mask.kv_num_blocks, block_mask.kv_indices, kv_blocks)
    full = torch.zeros_like(partial)
    if block_mask.full_kv_num_blocks is not None:
        full = _listed_blocks(
            block_mask.full_kv_num_blocks, block_mask.full_kv_indices, kv_blocks
        )
    mask_mod = block_mask.mask_mod
    batch, heads = partial.shape[:2]

    def listed_mask_mod(b, h, q_idx, kv_idx):
        block = (b % batch, h % heads, q_idx // q_block, kv_idx // kv_block)
        return full[block] | (partial[block] & mask_mod(b, h, q_idx, kv_idx))

    listed = copy.copy(block_mask)
    listed.mask_mod = listed_mask_mod
    return listed


def _as_gfx950():
    return mock.patch.object(
        flex_flydsl_attention, "_is_gfx950_device", return_value=True
    )


def _fused_backward():
    """Route every FlyDSL backward to the fused template."""
    return mock.patch.object(
        flex_flydsl_backward,
        "_get_flydsl_split_backward_config",
        return_value=(None, "fused backward under test"),
    )


def _bottom_right_mask(seq_q, seq_kv, device, *, batch=1):
    return create_block_mask(
        lambda b, h, q, kv: q + seq_kv - seq_q >= kv,
        batch,
        1,
        seq_q,
        seq_kv,
        device=device,
    )


def _compile_flydsl(options=None, **kwargs):
    return torch.compile(
        lambda q, k, v: flex_attention(
            q, k, v, kernel_options={"BACKEND": "FLYDSL", **(options or {})}, **kwargs
        ),
        fullgraph=True,
    )


def _cases(**cases):
    return [subtest(case, name=name) for name, case in cases.items()]


def _case_values(case, arity):
    """(argument values, name) of a ``parametrize`` case taking ``arity`` arguments."""
    if isinstance(case, subtest):
        if case.decorators:
            raise ValueError(f"{case.name}: decorated cases need parametrize")
        values, name = case.arg_values, case.name
    else:
        values, name = case, None
    values = tuple(values) if arity > 1 else (values,)
    if name is None:
        name = "_".join(str(value) for value in values)
    return values, name


def _paired(*axes):
    """``parametrize`` arguments covering every value of each ``(names, cases)``
    axis once, cycling the shorter axes, instead of their full product."""
    names = [[name.strip() for name in axis_names.split(",")] for axis_names, _ in axes]
    length = max(len(cases) for _, cases in axes)
    paired = []
    for index in range(length):
        values, labels = [], []
        for axis_names, (_, cases) in zip(names, axes):
            case_values, label = _case_values(
                cases[index % len(cases)], len(axis_names)
            )
            values.extend(case_values)
            labels.append(label)
        paired.append(subtest(tuple(values), name="_".join(labels)))
    return ",".join(itertools.chain.from_iterable(names)), paired


def _subtests(arg_names, cases):
    """Like ``parametrize``, but runs each case as a subtest of one test, for
    host-only checks that need no separate compile or device state."""
    names = [name.strip() for name in arg_names.split(",")]

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            for case in cases:
                values, label = _case_values(case, len(names))
                torch._dynamo.reset()
                counters.clear()
                with self.subTest(label):
                    fn(self, *args, **dict(zip(names, values)), **kwargs)

        del wrapper.__wrapped__
        signature = inspect.signature(fn)
        wrapper.__signature__ = signature.replace(
            parameters=[
                param
                for name, param in signature.parameters.items()
                if name not in names
            ]
        )
        return wrapper

    return decorator


@contextlib.contextmanager
def _spy_choices(fake_run=None, kinds=("forward", "backward")):
    """Stub FlyDSL launches with ``fake_run`` and yield the (kind, kwargs) of
    each choice the templates of ``kinds`` register."""
    templates = {
        "forward": flex_flydsl_forward_template,
        "dualwave": flex_flydsl_dualwave_forward_template,
        "backward": flex_flydsl_backward_template,
    }
    captured = []

    def spy(kind, append):
        def spy_append(choices, **kwargs):
            captured.append((kind, kwargs))
            return append(choices, **kwargs)

        return spy_append

    with contextlib.ExitStack() as stack:
        stack.enter_context(_as_gfx950())
        if "backward" in kinds:
            stack.enter_context(_fused_backward())
        stack.enter_context(
            mock.patch.object(FlyDSLKernelWrapper, "run", fake_run or mock.DEFAULT)
        )
        for kind in kinds:
            template = templates[kind]
            stack.enter_context(
                mock.patch.object(
                    template,
                    "maybe_append_choice",
                    spy(kind, template.maybe_append_choice),
                )
            )
        yield captured


@contextlib.contextmanager
def _flydsl_sources():
    """Yield the (kernel name, source, precompile metadata) of each FlyDSL
    module Inductor compiles."""
    from torch._inductor.async_compile import AsyncCompile

    sources = []
    compile_flydsl = AsyncCompile.flydsl

    def capture(self, kernel_name, source_code, precompile_metadata=None):
        sources.append((kernel_name, source_code, precompile_metadata))
        return compile_flydsl(self, kernel_name, source_code, precompile_metadata)

    with mock.patch.object(AsyncCompile, "flydsl", capture):
        yield sources


class TestFlyDSLFlexAttentionBackend(TestCase):
    @parametrize(*_paired(("qk_dim", [128, 192]), ("partial", [False, True])))
    def test_split_kv_reduction(self, device, qk_dim, partial):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(
            device=device,
            q_heads=4,
            seq_q=1,
            seq_kv=2048,
            qk_dim=qk_dim,
            seed=17,
        )
        counts = torch.arange(4, device=device, dtype=torch.int32).view(1, 4, 1)
        zero_counts = torch.zeros_like(counts)
        block_ids = torch.tensor([0, 5, 15], device=device, dtype=torch.int32)
        indices = block_ids.view(1, 1, 1, 3).expand(1, 4, 1, 3).contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx % 2 == 0 if partial else kv_idx >= 0

        block_mask = BlockMask.from_kv_blocks(
            counts if partial else zero_counts,
            indices,
            zero_counts if partial else counts,
            indices,
            BLOCK_SIZE=128,
            mask_mod=mask_mod,
            seq_lengths=(1, 2048),
            compute_q_blocks=False,
        )

        def attention(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLYDSL", "FLYDSL_SPLIT_KV": True},
                return_aux=AuxRequest(lse=True, max_scores=True),
            )

        (output, aux), code = run_and_get_code(
            torch.compile(attention, fullgraph=True), query, key, value
        )
        self.assertIn("kv_splits=KV_SPLITS", "\n".join(code))
        positions = torch.arange(2048, device=device)
        active_blocks = (
            torch.arange(3, device=device)[None, :, None]
            < torch.arange(4, device=device)[:, None, None]
        )
        keep = (
            (
                active_blocks
                & ((positions // 128)[None, None, :] == block_ids[None, :, None])
            )
            .any(dim=1)
            .view(1, 4, 1, 2048)
        )
        if partial:
            keep = keep & (positions % 2 == 0)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attn_mask=keep
        ).to(query.dtype)
        scores = (query.float() @ key.float().transpose(-1, -2)) * qk_dim**-0.5
        scores = scores.masked_fill(~keep, float("-inf"))
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)
        self.assertEqual(aux.lse, scores.logsumexp(-1), atol=0.025, rtol=0.025)
        self.assertEqual(aux.max_scores, scores.amax(-1), atol=0.025, rtol=0.025)
        self.assertEqual(output[:, 0], torch.zeros_like(output[:, 0]), atol=0, rtol=0)
        self.assertTrue(torch.isneginf(aux.lse[:, 0]).all())
        self.assertTrue(torch.isneginf(aux.max_scores[:, 0]).all())

    @_subtests("split_kv", [0, -2, "yes", 1.5])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_invalid_split_kv_option(self, device, split_kv):
        query, key, value = _make_qkv(device=device, seq_q=1, seq_kv=4096)
        compiled = _compile_flydsl({"FLYDSL_SPLIT_KV": split_kv})
        torch._dynamo.reset()
        with (
            _as_gfx950(),
            mock.patch.object(FlyDSLKernelWrapper, "run"),
            self.assertRaisesRegex(Exception, "FLYDSL_SPLIT_KV must be a bool or"),
        ):
            compiled(query, key, value)

    @parametrize("seq_q", [1, 256])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_dense_default_uses_isolated_backend(self, device, seq_q):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(device=device, seq_q=seq_q, seq_kv=256)
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )
        with (
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention._use_flex_decoding",
                side_effect=AssertionError("FlyDSL reached Triton decode selection"),
            ),
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention."
                "flex_attention_template.maybe_append_choice",
                side_effect=AssertionError("FlyDSL registered a Triton candidate"),
            ),
        ):
            output = compiled(query, key, value)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float()
        ).to(query.dtype)
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_computed_mask_capture(self, device):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(device=device)
        ends = torch.arange(256, device=device, dtype=torch.int32)
        counts = torch.full((1, 1, 2), 2, device=device, dtype=torch.int32)
        indices = torch.tensor([[[[0, 1], [0, 1]]]], device=device, dtype=torch.int32)
        full_counts = torch.zeros_like(counts)
        full_indices = torch.zeros_like(indices)

        def attention(q, k, v, document_ends):
            computed_ends = document_ends + 1

            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < computed_ends[q_idx]

            block_mask = BlockMask.from_kv_blocks(
                counts,
                indices,
                full_counts,
                full_indices,
                BLOCK_SIZE=128,
                mask_mod=mask_mod,
                seq_lengths=(256, 256),
                compute_q_blocks=False,
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLYDSL"}
            )

        output = torch.compile(attention, fullgraph=True)(query, key, value, ends)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=True
        ).to(query.dtype)
        # Match the bf16 scaled-Q precision used by the forward kernel.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)


instantiate_device_type_tests(
    TestFlyDSLFlexAttentionBackend, globals(), only_for=("cuda",)
)


class TestFlyDSLFlexAttentionConfig(TestCase):
    @_subtests(
        "strides,expected",
        [
            ((("seq", 8), 128, 8, 1), (128, 128, 8, 1)),
            ((None, 128, 8, 1), (4096, 128, 8, 1)),
            ((("seq", 8), None, 8, 1), (128, 128, 8, 1)),
        ],
    )
    def test_mixed_runtime_strides(self, strides, expected):
        if importlib.util.find_spec("flydsl") is None:
            self.skipTest("requires FlyDSL")
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            flex_attn_utils,
        )

        with mock.patch.object(
            flex_attn_utils, "layout_extents", return_value=((), (4096, 128, 8, 1))
        ):
            self.assertEqual(
                flex_attn_utils.runtime_strides(strides, object(), 16), expected
            )

    def test_gfx950_split_backward_row_descriptors(self):
        if importlib.util.find_spec("flydsl") is None:
            self.skipTest("requires FlyDSL")
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            flex_attn_split_bwd_dq as dq,
        )

        ctx = SimpleNamespace(
            lse_tokens_i32=16384,
            num_head_q=65536,
            batch_idx=3,
            q_head_idx=65535,
            q_row_off=0,
            LSE=object(),
            Delta=object(),
        )
        with (
            mock.patch.object(dq.fx, "Index", int),
            mock.patch.object(dq.ops, "make_buffer_rsrc") as resource,
        ):
            dq.BwdDqKernelContext.init_row_inputs(ctx)
        span = 16384 * 4
        base = (3 * 65536 + 65535) * span
        self.assertEqual(
            [entry.args for entry in resource.call_args_list],
            [(ctx.LSE, base, span), (ctx.Delta, base, span)],
        )
        self.assertEqual((ctx.row_base, ctx.row_oob_off), (0, 16384))

    @_subtests(
        "shape,stride,supported,metadata",
        [
            subtest(
                ((2, 3, 4, 5), (60, 20, 5, 1), (True,) * 2, True), name="contiguous"
            ),
            subtest(
                ((2, 1, 4, 5), (20, 99, 5, 1), (True,) * 2, True),
                name="size_one_dimension",
            ),
            subtest(
                ((2, 3, 4, 5), (120, 40, 5, 1), (True,) * 2, False),
                name="strided",
            ),
            subtest(
                ((2, 3, 4, 5), (60, 20, 1, 4), (False,) * 2, False),
                name="head_dim_stride",
            ),
            subtest(
                ((2, 3, 4, 5), (0, 20, 5, 1), (False, True), False),
                name="zero_batch_stride",
            ),
            subtest(
                ((2, 3, 4, 5), (20, 0, 5, 1), (False,) * 2, False),
                name="zero_head_stride",
            ),
            subtest(
                ((2, 0, 4, 5), (0, 20, 5, 1), (True,) * 2, False),
                name="zero_size_strict",
            ),
            subtest(
                ((2, 0, 4, 5), (20, 20, 5, 1), (True,) * 2, True),
                name="zero_size_metadata",
            ),
            subtest(
                ((2, 3, 4, 5), (20, 5, 1), (False,) * 2, False),
                name="rank_mismatch",
            ),
            subtest(((), (), (False,) * 2, True), name="scalar_metadata"),
        ],
    )
    def test_bhsd_stride_checks(self, shape, stride, supported, metadata):
        with V.set_graph_handler(_fake_graph()):
            for broadcast_batch, expected in zip((False, True), supported):
                self.assertEqual(
                    _get_supported_bhsd_stride(
                        _FakeNode(shape, stride), broadcast_batch=broadcast_batch
                    ),
                    tuple(stride) if expected else None,
                )
            self.assertEqual(_is_contiguous_shape_stride(shape, stride), metadata)

    @_subtests(
        "hints,kv_batch",
        [
            subtest(((3, 1000, 1500), None), name="ragged_prefill"),
            subtest(((3, 1024, 2048), None), name="aligned_prefill"),
            subtest(((3, 4, 1500), None), name="decode"),
            subtest(((3, 1000, 1500), 1), name="kv_batch_1"),
        ],
    )
    def test_symbolic_sizes_reach_the_forward_at_runtime(self, hints, kv_batch):
        graph, (batch, seq_q, seq_kv) = _symbolic_graph(*hints)
        decode = hints[1] < 128
        q_blocks = 1 if decode else None
        inputs = _symbolic_forward_inputs(batch, seq_q, seq_kv, kv_batch=kv_batch)
        if decode:
            for name in ("kv_num_blocks", "full_kv_num_blocks"):
                inputs[name] = _symbolic_node([1, 1, 1], torch.int32)
            for name in ("kv_indices", "full_kv_indices"):
                width = inputs[name].get_size()[-1]
                inputs[name] = _symbolic_node([1, 1, 1, width], torch.int32)
        config, _ = _fake_choice_result(inputs, graph)
        self.assertTrue(config["RUNTIME_SIZES"])
        self.assertEqual(
            [
                config[name]
                for name in (
                    "BATCH_SIZE",
                    "SEQ_Q",
                    "SEQ_KV",
                    "NUM_Q_BLOCKS",
                    "MAX_PARTIAL_BLOCKS",
                    "MAX_FULL_BLOCKS",
                )
            ],
            [None, None, None, q_blocks, None, None],
        )
        self.assertEqual(
            (config["NUM_Q_HEADS"], config["NUM_KV_HEADS"], config["QK_HEAD_DIM"]),
            (8, 2, 128),
        )
        # Strides that span a sequence are multiples of the runtime length.
        query_stride = (("seq", 8 * 128), ("seq", 128), 128, 1)
        self.assertEqual(config["Q_STRIDE"], query_stride)
        self.assertEqual(config["O_STRIDE"], query_stride)
        self.assertEqual(
            config["K_STRIDE"],
            (0 if kv_batch else ("seq", 2 * 128), ("seq", 128), 128, 1),
        )
        plan = config["PLAN"]
        self.assertEqual(plan["decode"], decode)
        self.assertFalse(plan["paired"])
        self.assertEqual(plan["q_ragged"], not decode)
        self.assertEqual(plan["kv_ragged"], hints[2] % 128 != 0)
        # The heads interleaved per CTA group divide the static head count.
        self.assertEqual(plan["heads_per_group"], 2 if decode else 8)
        guards = {str(guard.expr) for guard in graph.sizevars.shape_env.guards}
        self.assertIn(
            f"Ne(Mod({seq_kv}, 128), 0)"
            if hints[2] % 128
            else f"Eq(Mod({seq_kv}, 128), 0)",
            guards,
        )
        self.assertIn(f"{seq_q} < 128" if decode else f"{seq_q} >= 128", guards)
        # B only meets thresholds.
        self.assertTrue(
            all(
                str(batch) not in guard or any(op in guard for op in "<>")
                for guard in guards
            )
        )

    @_subtests(
        "hints,heads,expected,same,different",
        _cases(
            # A graph recompiles only for sizes that change a choice: four-wave
            # owners, an aligned Sk or decode.
            two_wave_ragged_prefill=(
                (3, 1000, 1500),
                (8, 2),
                {"owner_waves": 2, "paired": False},
                (5, 600, 900),
                [(5, 1000, 1500), (3, 1000, 1536), (3, 64, 1500)],
            ),
            # Paired rows need 512 CTAs, Sq % 256 = 0 and Sk % 128 = 0, and
            # unroll over at most 32 KV blocks here.
            paired_prefill=(
                (2, 2048, 2048),
                (32, 8),
                {"owner_waves": 8, "paired": True, "paired_kv_blocks": 32},
                (4, 1024, 3072),
                [
                    (1, 2048, 2048),
                    (2, 2000, 2048),
                    (2, 2048, 1920),
                    (2, 2048, 2000),
                    (2, 2048, 4224),
                ],
            ),
        ),
    )
    def test_forward_plan_guards_its_choices(
        self, hints, heads, expected, same, different
    ):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_plan import (
            forward_plan,
        )

        graph, symbols = _symbolic_graph(*hints)
        shape_env = graph.sizevars.shape_env
        sizes = [shape_env.create_symintnode(symbol, hint=None) for symbol in symbols]
        plan = forward_plan(
            batch_size=sizes[0],
            num_q_heads=heads[0],
            num_kv_heads=heads[1],
            seq_q=sizes[1],
            seq_kv=sizes[2],
            qk_head_dim=128,
            v_head_dim=128,
        )
        self.assertEqual({name: plan[name] for name in expected}, expected)

        def holds(values):
            replacements = dict(zip(symbols, map(sympy.Integer, values)))
            return all(
                bool(guard.expr.xreplace(replacements)) for guard in shape_env.guards
            )

        self.assertTrue(holds(hints))
        self.assertTrue(holds(same))
        for values in different:
            self.assertFalse(holds(values), values)

    @_subtests(
        "sizes,heads,requested,expected",
        _cases(
            # (B, Sq, Sk), (Hq, Hkv): B * Hkv one-wave CTAs fill 256 CUs.
            b1_hkv8_128k=((1, 1, 131072), (8, 8), True, 32),
            b4_gqa_128k=((4, 1, 131072), (32, 8), True, 8),
            # Eight 128-key blocks per split at least.
            min_blocks=((1, 1, 4096), (8, 8), True, 4),
            short_kv=((1, 1, 1000), (8, 8), True, 1),
            full_grid=((64, 1, 131072), (8, 8), True, 1),
            eight_wave_rows=((1, 16, 65536), (64, 4), True, 64),
            most_splits=((1, 1, 1 << 20), (1, 1), True, 256),
            prefill=((1, 128, 131072), (8, 8), True, 1),
            off=((1, 1, 131072), (8, 8), False, 1),
            requested=((1, 1, 131072), (8, 8), 6, 6),
            requested_past_blocks=((1, 1, 300), (8, 8), 6, 3),
        ),
    )
    def test_forward_kv_splits(self, sizes, heads, requested, expected):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_plan import (
            forward_kv_splits,
        )

        batch, seq_q, seq_kv = sizes
        splits = forward_kv_splits(
            batch_size=batch,
            num_q_heads=heads[0],
            num_kv_heads=heads[1],
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_head_dim=128,
            v_head_dim=128,
            requested=requested,
        )
        self.assertEqual(splits, expected)

    def test_target_limits(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_target import (
            FLEX_ARCH,
            GFX950_LIMITS,
            target_limits,
        )

        if torch.version.hip is None or not torch.cuda.is_available():
            self.assertEqual(target_limits(), GFX950_LIMITS)
            return
        props = torch.cuda.get_device_properties(0)
        expected = GFX950_LIMITS
        if props.gcnArchName.split(":", 1)[0] == FLEX_ARCH:
            expected = (props.shared_memory_per_block, props.multi_processor_count)
        self.assertEqual(target_limits(torch.device("cuda", 0)), expected)

    def test_plans_follow_target_limits(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            backward_single_pass,
            forward_value_slice,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_plan import (
            forward_kv_splits,
            forward_plan,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_target import (
            GFX950_LIMITS,
        )

        half_cus = GFX950_LIMITS._replace(num_cus=128)
        small_lds = GFX950_LIMITS._replace(lds_bytes=128 << 10)
        decode = {
            "batch_size": 1,
            "num_q_heads": 8,
            "num_kv_heads": 8,
            "seq_q": 1,
            "seq_kv": 131072,
            "qk_head_dim": 128,
            "v_head_dim": 128,
        }
        # Split-KV fills each CU with one CTA, so it stops once B * Hkv does.
        self.assertEqual(forward_kv_splits(**decode, target=GFX950_LIMITS), 32)
        self.assertEqual(forward_kv_splits(**decode, target=half_cus), 16)
        sixteen_batches = {**decode, "batch_size": 16}
        self.assertTrue(
            forward_plan(**sixteen_batches, target=GFX950_LIMITS)["split_kv"]
        )
        self.assertFalse(forward_plan(**sixteen_batches, target=half_cus)["split_kv"])
        # Narrower V slices and no full-width backward rows where LDS is smaller.
        self.assertEqual(forward_value_slice(384, 512, GFX950_LIMITS), 256)
        self.assertEqual(forward_value_slice(384, 512, small_lds), 128)
        self.assertTrue(backward_single_pass(256, 512, GFX950_LIMITS))
        self.assertFalse(backward_single_pass(256, 512, small_lds))

    def test_split_kv_builder_rejects_prefill(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            build_flex_attn_fwd_module,
        )

        if importlib.util.find_spec("flydsl") is None:
            self.skipTest("requires the FlyDSL package")
        with self.assertRaisesRegex(ValueError, "KV blocks of decode CTAs only"):
            build_flex_attn_fwd_module(
                batch_size=1,
                num_q_heads=8,
                num_kv_heads=8,
                seq_q=256,
                seq_kv=4096,
                qk_head_dim=128,
                v_head_dim=128,
                block_mask_batch=1,
                block_mask_heads=1,
                num_q_blocks=2,
                max_partial_blocks=32,
                max_full_blocks=32,
                sparse_q_block_size=128,
                sparse_kv_block_size=128,
                scale=1.0,
                kv_splits=2,
            )

    @_subtests("block_list", [False, True])
    def test_symbolic_sizes_reach_the_backward_at_runtime(self, block_list):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_plan import (
            backward_plan,
        )

        graph, (batch, seq_q, seq_kv) = _symbolic_graph(3, 1000, 1500)
        inputs = _symbolic_forward_inputs(batch, seq_q, seq_kv)
        inputs["fw_subgraph"] = inputs.pop("subgraph")
        for name in ("logsumexp", "max_scores"):
            inputs.pop(name)
        inputs.update(
            out=_symbolic_node([batch, 8, seq_q, 128]),
            grad_out=_symbolic_node([batch, 8, seq_q, 128]),
            grad_logsumexp=None,
            block_mask_is_dense=not block_list,
        )
        with (
            V.set_graph_handler(graph),
            mock.patch.object(
                flex_flydsl_attention, "_is_gfx950_device", return_value=True
            ),
        ):
            config, reason = _get_flydsl_flex_attention_backward_config(**inputs)
            self.assertEqual(reason, "")
            shape_env = graph.sizevars.shape_env
            sizes = [
                shape_env.create_symintnode(s, hint=None)
                for s in (batch, seq_q, seq_kv)
            ]
            plan = backward_plan(
                batch_size=sizes[0],
                num_heads=8,
                num_kv_heads=2,
                sequence_length=sizes[1],
                kv_sequence_length=sizes[2],
                qk_head_dim=128,
                value_head_dim=128,
                mask_traversal="block_list" if block_list else "direct_range",
                workspace_element_bytes=4,
                strides=(config["Q_STRIDE"],) * 8,
                dq_partitions=8,
            )
        self.assertEqual(
            [config[name] for name in ("BATCH_SIZE", "SEQ_LEN_Q", "SEQ_LEN_KV")],
            [None, None, None],
        )
        self.assertIsNone(config["MAX_PARTIAL_BLOCKS"])
        self.assertEqual((config["NUM_HEADS"], config["NUM_KV_HEADS"]), (8, 2))
        # The slot count is fixed for any Sk: only block lists add into zeroed slots.
        self.assertEqual(plan["dq_partitions"], 8)
        self.assertEqual(plan["dq_accumulates"], block_list)
        self.assertEqual((plan["q_ragged"], plan["kv_ragged"]), (True, True))
        self.assertEqual((plan["workspace_head_group"], plan["cta_head_group"]), (8, 2))

    def test_unrealized_metadata_is_rejected(self):
        inputs = _supported_fake_forward_inputs()
        with (
            V.set_graph_handler(_fake_graph()),
            mock.patch.object(
                inputs["kv_indices"], "get_stride", side_effect=NotImplementedError
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            appended, reason = maybe_append_flydsl_flex_attention_choice(
                [], layout=mock.Mock(), **inputs
            )
        self.assertFalse(appended)
        self.assertIn("requires statically known BlockMask metadata strides", reason)
        append.assert_not_called()

    @_subtests(
        "unsupported,expected",
        (
            ("uneven_gqa", "Q heads divisible by KV heads"),
            ("mask_rows", "one BlockMask row per 128 Q rows"),
            ("block_size_64", "splits or merges into 128x128 blocks"),
            ("block_size_96", "splits or merges into 128x128 blocks"),
            ("block_size_0", "requires positive sparse Q/KV block sizes"),
            ("kv_mismatch", "K and V to share heads and length"),
            ("mask_heads", "must be 1, KV heads, or Q heads"),
            ("head_dim", "supports only"),
            ("grad_lse_shape", "grad_logsumexp shape [B, Hq, Sq]"),
            ("workspace", "workspace larger than 16 GiB"),
            ("workspace_view", "fit a 32-bit buffer range"),
            ("reverse_lists", "at most 16384 query blocks"),
        ),
    )
    def test_backward_config_gates(self, unsupported, expected):
        inputs = _supported_fake_backward_inputs()
        if unsupported == "uneven_gqa":
            inputs = _supported_fake_backward_inputs(
                q_size=(1, 3, 256, 128), k_size=(1, 2, 256, 128)
            )
        elif unsupported.startswith("block_size_"):
            inputs["sparse_q_block_size"] = int(unsupported.rsplit("_", 1)[1])
        elif unsupported == "mask_rows":
            inputs = _supported_fake_backward_inputs(q_size=(1, 2, 300, 128))
            counts = _FakeNode([1, 1, 2], [2, 2, 1], torch.int32)
            inputs["kv_num_blocks"] = inputs["full_kv_num_blocks"] = counts
        elif unsupported == "kv_mismatch":
            shape = (1, 2, 128, 128)
            inputs["value"] = _FakeNode(shape, _contiguous_stride(shape))
        elif unsupported == "mask_heads":
            inputs = _supported_fake_backward_inputs(
                q_size=(1, 4, 256, 128), k_size=(1, 2, 256, 128), mask_heads=3
            )
        elif unsupported == "workspace_view":
            # One dQ slot of a 32-head workspace group exceeds the 32-bit buffer
            # range while the whole workspace is still below 16 GiB.
            inputs = _supported_fake_backward_inputs(
                q_size=(1, 32, 1366 * 128, 192), k_size=(1, 8, 256, 192)
            )
        elif unsupported == "reverse_lists":
            inputs = _supported_fake_backward_inputs(
                q_size=(1, 1, 16385 * 128, 128), k_size=(1, 1, 256, 128)
            )
        elif unsupported == "head_dim":
            shape = (1, 2, 256, 80)
            inputs["query"] = _FakeNode(shape, _contiguous_stride(shape))
            inputs["key"] = _FakeNode(shape, _contiguous_stride(shape))
        elif unsupported == "grad_lse_shape":
            inputs["grad_logsumexp"] = inputs["out"]
        else:
            qk_shape = (32, 64, 16384, 192)
            value_shape = (*qk_shape[:-1], 128)
            inputs["query"] = _FakeNode(qk_shape, _contiguous_stride(qk_shape))
            inputs["key"] = _FakeNode(qk_shape, _contiguous_stride(qk_shape))
            inputs["value"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
            inputs["out"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
            inputs["grad_out"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
        config, reason = _backward_config_result(inputs)
        self.assertIsNone(config)
        self.assertIn(expected, reason)

    @_subtests(
        "key_stride,expected,backward_expected",
        [
            subtest(
                (None, "K/V batch of 1 or B", "K/V batch of 1 or the Q batch"),
                name="kv_batch_2_of_4",
            ),
            subtest(
                ((65536, 0, 128, 1),) + ("contiguous head dimensions",) * 2,
                name="zero_head_stride",
            ),
        ],
    )
    def test_broadcast_kv_batch_rejects(self, key_stride, expected, backward_expected):
        k_size = (2, 2, 512, 128) if key_stride is None else (4, 2, 512, 128)
        forward = _supported_fake_forward_inputs(q_size=(4, 8, 512, 128), k_size=k_size)
        backward = _supported_fake_backward_inputs(
            q_size=(4, 8, 512, 128), k_size=k_size
        )
        if key_stride is not None:
            forward["key"] = backward["key"] = _FakeNode(k_size, key_stride)
        kwargs, reason = _fake_choice_result(forward)
        self.assertIsNone(kwargs)
        self.assertIn(expected, reason)
        config, reason = _backward_config_result(backward)
        self.assertIsNone(config)
        self.assertIn(backward_expected, reason)

    def test_ragged_prefill_requires_ceil_block_mask_rows(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(1, 2, 1000, 128), k_size=(1, 2, 1000, 128)
        )
        counts = _FakeNode([1, 1, 7], [7, 7, 1], torch.int32)
        indices = _FakeNode([1, 1, 7, 8], [56, 56, 8, 1], torch.int32)
        inputs.update(
            kv_num_blocks=counts,
            full_kv_num_blocks=counts,
            kv_indices=indices,
            full_kv_indices=indices,
        )
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn("ceil(Sq / 128) BlockMask rows", reason)

    @_subtests("dtype,expected", [(torch.bfloat16, "bf16"), (torch.float16, "fp16")])
    def test_dtype_config(self, dtype, expected):
        kwargs, reason = _fake_choice_result(
            _supported_fake_forward_inputs(dtype=dtype)
        )
        self.assertIsNotNone(kwargs, reason)
        config, reason = _backward_config_result(
            _supported_fake_backward_inputs(dtype=dtype)
        )
        self.assertIsNotNone(config, reason)
        self.assertEqual((kwargs["DTYPE"], config["DTYPE"]), (expected, expected))

    def test_backward_rejects_mixed_dtypes(self):
        inputs = _supported_fake_backward_inputs(dtype=torch.float16)
        shape = (1, 2, 256, 128)
        inputs["grad_out"] = _FakeNode(shape, _contiguous_stride(shape))
        config, reason = _backward_config_result(inputs)
        self.assertIsNone(config)
        self.assertIn("grad_out in the query dtype", reason)

    @_subtests("dq_accum_fp32", (True, False))
    def test_backward_workspace_bound(self, dq_accum_fp32):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            choose_dq_partitions,
            dq_workspace_head_group,
            MAX_BUFFER_VIEW_BYTES,
            MAX_DQ_WORKSPACE_BYTES,
        )

        element_bytes = 4 if dq_accum_fp32 else 2
        for batch_heads, kv_batch_heads, seq_q, seq_kv, head_dim, key_rows, listed in (
            (1024, 1024, 8192, 8192, 192, 192, False),
            (32, 32, 16384, 16384, 192, 192, False),
            (32, 4, 32768, 32768, 192, 192, False),
            (32, 32, 65536, 65536, 128, 128, True),
            (16, 2, 4096, 131072, 192, 192, False),
            (2, 2, 174720, 256, 192, 128, True),
            (32, 8, 32768, 32768, 256, 64, False),
            (2, 2, 131072, 4096, 256, 64, True),
        ):
            with self.subTest(batch_heads=batch_heads, seq_q=seq_q, seq_kv=seq_kv):
                partitions = choose_dq_partitions(
                    batch_heads,
                    seq_q,
                    key_rows,
                    element_bytes,
                    qk_head_dim=head_dim,
                    key_sequence_length=seq_kv,
                    compute_batch_heads=kv_batch_heads,
                    block_list=listed,
                )
                self.assertGreaterEqual(partitions, 1)
                if listed:
                    # The dQ layout pass tracks written block-list slots in 32 bits.
                    self.assertLessEqual(partitions, 32)
                self.assertLessEqual(partitions, -(-seq_kv // key_rows))
                slot_bytes = seq_q * head_dim * element_bytes
                self.assertLessEqual(
                    batch_heads * partitions * slot_bytes, MAX_DQ_WORKSPACE_BYTES
                )
                self.assertLessEqual(
                    partitions * slot_bytes * dq_workspace_head_group(batch_heads),
                    MAX_BUFFER_VIEW_BYTES,
                )
        # One slot of a 32-head workspace group no longer fits a buffer view.
        with self.assertRaisesRegex(ValueError, "32-bit buffer range"):
            choose_dq_partitions(32, 174848, 192, 4, qk_head_dim=192)

    @_subtests(
        "q_heads,kv_heads,seq_q,seq_kv,empty_rows,softcap",
        [
            subtest((2, 2, 48, 48, 0, False), name="mha"),
            subtest((4, 2, 48, 48, 0, False), name="gqa"),
            subtest((2, 2, 48, 80, 0, False), name="cross"),
            subtest((4, 1, 48, 48, 8, False), name="mqa_empty_rows"),
            subtest((4, 2, 48, 80, 8, True), name="gqa_cross_softcap"),
            subtest((2, 2, 45, 45, 0, False), name="ragged"),
            subtest((4, 2, 37, 77, 0, False), name="gqa_ragged_cross"),
            subtest((4, 2, 45, 61, 8, True), name="ragged_empty_rows_softcap"),
        ],
    )
    def test_grad_logsumexp_delta_matches_autograd(
        self, q_heads, kv_heads, seq_q, seq_kv, empty_rows, softcap
    ):
        torch.manual_seed(0)
        dim, scale = 16, 0.25
        group = q_heads // kv_heads
        # Delta is padded to whole query blocks (16 rows here, 128 in the kernel).
        padded_rows = -seq_q % 16

        def score_mod(scores):
            return 4.0 * torch.tanh(scores / 4.0) if softcap else scores

        def make(*shape):
            return torch.randn(*shape, dtype=torch.float64)

        query, key, value = (
            make(1, q_heads, seq_q, dim),
            make(1, kv_heads, seq_kv, dim),
            make(1, kv_heads, seq_kv, dim),
        )
        grad_out, grad_lse = make(1, q_heads, seq_q, dim), make(1, q_heads, seq_q)
        rows = torch.arange(seq_q)[:, None]
        allowed = (torch.arange(seq_kv)[None, :] <= rows + seq_kv - seq_q) & (
            rows >= empty_rows
        )

        def attention(q, k, v):
            k, v = (t.repeat_interleave(group, dim=1) for t in (k, v))
            scores = score_mod((q @ k.transpose(-2, -1)) * scale)
            scores = scores.masked_fill(~allowed[empty_rows:], -math.inf)
            lse2 = scores.logsumexp(-1) / math.log(2)
            out = torch.exp2(scores / math.log(2) - lse2[..., None]) @ v
            return out, lse2

        # Rows are independent, so autograd only sees the rows that attend to a key.
        inputs = [
            t.clone().requires_grad_() for t in (query[:, :, empty_rows:], key, value)
        ]
        out, lse2 = attention(*inputs)
        # The user-visible LSE is the forward's log2 LSE times ln(2).
        loss = (out * grad_out[:, :, empty_rows:]).sum() + (
            lse2 * math.log(2) * grad_lse[:, :, empty_rows:]
        ).sum()
        expected = list(torch.autograd.grad(loss, inputs))
        expected[0] = torch.cat(
            [torch.zeros_like(query[:, :, :empty_rows]), expected[0]], 2
        )

        full_out = torch.cat(
            [torch.zeros_like(grad_out[:, :, :empty_rows]), out.detach()], 2
        )
        full_lse2 = torch.cat(
            [torch.full_like(grad_lse[:, :, :empty_rows], -math.inf), lse2.detach()], 2
        )
        # The backward receives the gradient of the log2 LSE. Empty rows get an
        # arbitrary finite LSE gradient and must contribute nothing.
        hop_grad_lse = grad_lse * math.log(2)

        def math_grads(grad_lse2):
            k, v = (t.repeat_interleave(group, dim=1) for t in (key, value))

            # Padded rows load zeros and are masked out by the row bound.
            def pad(t, value=0.0):
                return torch.nn.functional.pad(
                    t, (0, 0) * (t.dim() - 3) + (0, padded_rows), value=value
                )

            dq, dk, dv = _log2_lse_backward(
                pad(query),
                k,
                v,
                pad(full_out),
                pad(full_lse2),
                pad(grad_out),
                pad(grad_lse2),
                scale,
                pad(allowed[None, None], False)[0, 0],
                score_mod,
            )
            shape = (1, kv_heads, group, seq_kv, dim)
            return dq[:, :, :seq_q], dk.view(shape).sum(2), dv.view(shape).sum(2)

        actual = math_grads(hop_grad_lse)
        for grad, expected_grad in zip(actual, expected):
            self.assertTrue(grad.isfinite().all())
            self.assertEqual(grad, expected_grad)
        without_dlse = math_grads(torch.zeros_like(hop_grad_lse))
        self.assertNotEqual(without_dlse[0], expected[0])

    def test_backward_builder_validates_heads_and_lengths(self):
        try:
            from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_gfx950 import (
                build_flex_attn_bwd_module,
            )
        except ImportError:
            self.skipTest("requires the FlyDSL Python package")

        def build(**kwargs):
            options = {
                "num_heads": 8,
                "sequence_length": 256,
                "num_kv_heads": 2,
                "kv_sequence_length": 384,
                "block_mask_heads": 1,
                "mask_program": (("ge", 2, 3),),
                "mask_program_output": 4,
            }
            options.update(kwargs)
            return build_flex_attn_bwd_module(
                1,
                options.pop("num_heads"),
                options.pop("sequence_length"),
                128,
                128,
                "bf16",
                128,
                128,
                lse_in_log2=True,
                **options,
            )

        for block_mask_heads in (1, 2, 8):
            self.assertTrue(callable(build(block_mask_heads=block_mask_heads)))
        for sequence_length, kv_sequence_length in ((1000, 200), (1, 77)):
            self.assertTrue(
                callable(
                    build(
                        sequence_length=sequence_length,
                        kv_sequence_length=kv_sequence_length,
                    )
                )
            )
        self.assertTrue(callable(build(mask_program=(), block_mask_is_dense=True)))
        for kwargs, message in (
            ({"num_kv_heads": 3}, "num_heads divisible by num_kv_heads"),
            ({"kv_sequence_length": 0}, "positive kv_sequence_length"),
            ({"sequence_length": 0}, "positive sequence_length"),
            ({"block_mask_heads": 4}, "1, num_kv_heads, or num_heads"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    build(**kwargs)

    def test_unavailable_backward_falls_back_before_allocation(self):
        counters.clear()
        with (
            mock.patch.object(
                flex_flydsl_backward, "runtime_available", return_value=False
            ),
            mock.patch.object(flex_flydsl_backward, "empty_strided") as allocate,
            self.assertLogs(flex_flydsl_backward.perf_hint_log, "INFO") as logs,
        ):
            result = flex_flydsl_backward.create_flydsl_flex_attention_backward_kernel(
                None, None, None, None, None, None, None, 1.0, 128, 128
            )
        self.assertIsNone(result)
        allocate.assert_not_called()
        self.assertEqual(counters["inductor"]["flydsl_flex_bwd_fallback"], 1)
        self.assertIn("FlyDSL runtime is unavailable", "\n".join(logs.output))

    def test_unsupported_backward_logs_fallback_reason(self):
        counters.clear()
        with (
            V.set_graph_handler(_fake_graph()),
            _fused_backward(),
            mock.patch.object(
                flex_flydsl_backward, "runtime_available", return_value=True
            ),
            mock.patch.object(
                flex_flydsl_backward,
                "_get_flydsl_flex_attention_backward_config",
                return_value=(None, "FlyDSL flex bwd currently supports MHA"),
            ),
            mock.patch.object(
                flex_flydsl_attention,
                "prepare_flydsl_mask_captures",
                side_effect=lambda buffers, device: buffers,
            ),
            mock.patch.object(
                flex_flydsl_backward, "maybe_realize", side_effect=lambda x: x
            ),
            mock.patch.object(
                flex_flydsl_backward,
                "_require_bhsd_stride",
                side_effect=lambda x, **kwargs: x,
            ),
            mock.patch.object(flex_flydsl_backward, "empty_strided") as allocate,
            self.assertLogs(flex_flydsl_backward.perf_hint_log, "INFO") as logs,
        ):
            query = _FakeNode((1, 2, 256, 128), (65536, 32768, 128, 1))
            result = flex_flydsl_backward.create_flydsl_flex_attention_backward_kernel(
                query,
                query,
                query,
                query,
                query,
                query,
                None,
                1.0,
                128,
                128,
                fw_subgraph=object(),
                mask_graph=SimpleNamespace(
                    graph_module=_mask_graph(lambda b, h, q, kv: q >= kv)
                ),
            )
        self.assertIsNone(result)
        allocate.assert_not_called()
        self.assertEqual(counters["inductor"]["flydsl_flex_bwd_fallback"], 1)
        self.assertIn("supports MHA", "\n".join(logs.output))

    @_subtests(
        "option,name",
        [
            ("dq_accum_fp32", "DQ_ACCUM_FP32"),
            ("single_pass", "FLYDSL_WIDE_SINGLE_PASS"),
            ("split_backward", "FLYDSL_SPLIT_BACKWARD"),
        ],
    )
    def test_invalid_backward_bool_option(self, option, name):
        with self.assertRaisesRegex(ValueError, f"{name} must be a bool"):
            flex_flydsl_backward.create_flydsl_flex_attention_backward_kernel(
                *[None] * 7, 1.0, 128, 128, **{option: 1}
            )


class _FlyDSLFlexAttentionRuntimeMixin:
    def _require_runtime(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

    def _assert_close_to_fp64(self, actual, reference, golden, *, atol, rtol):
        """``actual`` is within ``atol``/``rtol`` of the fp64 ``golden``, plus
        twice the low-precision ``reference``'s error at each element, and about
        as accurate as ``reference`` overall."""
        # bf16 matmuls can leave outputs silently wrong on gfx950, so a bf16
        # result only scales the error bound.
        self.assertEqual(actual.shape, golden.shape)
        self.assertEqual(reference.shape, golden.shape)
        self.assertEqual(actual.dtype, reference.dtype)
        golden = golden.double()
        self.assertFalse(golden.isnan().any(), "the fp64 reference has NaNs")
        self.assertEqual(actual.isfinite(), golden.isfinite())
        finite = golden.isfinite()
        self.assertEqual(actual.double()[~finite], golden[~finite])
        error = (actual.double() - golden)[finite]
        reference_error = (reference.double() - golden)[finite]
        excess = error.abs() - (
            atol + rtol * golden[finite].abs() + 2 * reference_error.abs()
        )
        if not (excess <= 0).all():
            self.fail(f"error exceeds its bound by up to {excess.max().item()}")
        self.assertLessEqual(
            error.norm().item(),
            1.5 * reference_error.norm().item() + 1e-3 * golden[finite].norm().item(),
        )

    def _compare_forward(
        self,
        query,
        key,
        value,
        *,
        block_mask=None,
        reference_block_mask=None,
        reference_options=None,
        scale=None,
        enable_gqa=False,
        return_aux=False,
        score_mod=None,
        flydsl_options=None,
    ):
        """``reference_options``/``flydsl_options``: Triton/FlyDSL kernel options."""

        def run(q, k, v, backend, selected_mask):
            options = {"TRITON": reference_options, "FLYDSL": flydsl_options}.get(
                backend
            )
            kwargs = {
                "block_mask": selected_mask,
                "enable_gqa": enable_gqa,
                "kernel_options": (
                    None if backend is None else {"BACKEND": backend, **(options or {})}
                ),
                "score_mod": score_mod,
            }
            if scale is not None:
                kwargs["scale"] = scale
            if return_aux:
                kwargs["return_aux"] = AuxRequest(lse=True, max_scores=True)
            return flex_attention(q, k, v, **kwargs)

        actual, code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL", block_mask),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        selected_mask = (
            reference_block_mask if reference_block_mask is not None else block_mask
        )
        expected, reference_code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "TRITON", selected_mask),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        golden = run(
            *(t.double() for t in (query, key, value)),
            None,
            _with_listed_blocks(selected_mask),
        )
        torch.cuda.synchronize()
        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(reference_code))
        if (flydsl_options or {}).get("FLYDSL_SPLIT_KV"):
            self.assertIn("kv_splits=KV_SPLITS", "\n".join(code))

        if not return_aux:
            self._assert_close_to_fp64(actual, expected, golden, atol=0.03, rtol=0.02)
            return actual

        output, aux = actual
        reference, reference_aux = expected
        golden, golden_aux = golden
        self._assert_close_to_fp64(output, reference, golden, atol=0.03, rtol=0.02)
        for name in ("lse", "max_scores"):
            self._assert_close_to_fp64(
                getattr(aux, name),
                getattr(reference_aux, name),
                getattr(golden_aux, name),
                atol=0.03,
                rtol=0.01,
            )
        return output, aux

    def _compare_backward(
        self,
        query,
        key,
        value,
        *,
        block_mask,
        dq_accum_fp32,
        fp64_mask_mod=None,
        score_mod=None,
        grad_lse=False,
        expand_kv=False,
        reference_block_mask=None,
        reference_options=None,
        flydsl_options=None,
        split=False,
    ):
        """``expand_kv`` expands a batch-1 K/V to the query batch in the graph;
        the Triton reference runs on ``reference_block_mask`` (else
        ``block_mask``) with kernel options ``reference_options``, FlyDSL with
        ``flydsl_options``. FlyDSL takes the split backward with ``split``, else
        the fused one."""
        torch.manual_seed(24)
        grad_output = torch.randn(
            *query.shape[:3], value.size(-1), device=query.device, dtype=query.dtype
        )
        grad_logsumexp = torch.randn(*query.shape[:3], device=query.device)
        enable_gqa = query.size(1) != key.size(1)

        def run(backend):
            """A None ``backend`` runs eagerly on fp64 inputs."""
            dtype = torch.float64 if backend is None else query.dtype
            inputs = tuple(
                tensor.detach().to(dtype, copy=True).requires_grad_()
                for tensor in (query, key, value)
            )
            kernel_options = None if backend is None else {"BACKEND": backend}
            selected_mask = block_mask
            if backend == "FLYDSL":
                kernel_options["bwd_FLYDSL_DQ_ACCUM_FP32"] = dq_accum_fp32
                if split is not None:
                    kernel_options["FLYDSL_SPLIT_BACKWARD"] = split
                kernel_options.update(flydsl_options or {})
            elif reference_block_mask is not None:
                selected_mask = reference_block_mask
            if backend == "TRITON":
                kernel_options.update(reference_options or {})
            elif backend is None:
                selected_mask = _with_listed_blocks(selected_mask)

            def attention(q, k, v):
                if expand_kv:
                    k, v = (t.expand(q.size(0), -1, -1, -1) for t in (k, v))
                return flex_attention(
                    q,
                    k,
                    v,
                    score_mod=score_mod,
                    block_mask=selected_mask,
                    enable_gqa=enable_gqa,
                    kernel_options=kernel_options,
                    return_aux=AuxRequest(lse=True) if grad_lse else None,
                )

            compiled = torch.compile(
                attention,
                fullgraph=True,
                backend="eager" if backend is None else "inductor",
            )
            with DeterministicGuard(backend == "FLYDSL"):
                result, code = run_and_get_code(compiled, *inputs)
                if grad_lse:
                    output, aux = result
                    outputs = (output, aux.lse)
                    grads = (grad_output.to(dtype), grad_logsumexp.to(aux.lse.dtype))
                else:
                    output, outputs, grads = result, result, grad_output.to(dtype)
                _, backward_code = run_and_get_code(
                    lambda: torch.autograd.backward(outputs, grads)
                )
            return (
                output.detach(),
                tuple(t.grad.detach().clone() for t in inputs),
                "\n".join(code),
                "\n".join(backward_code),
            )

        output, grads, code, backward_code = run("FLYDSL")
        reference, reference_grads, reference_code, reference_backward_code = run(
            "TRITON"
        )
        golden, golden_grads, _, _ = run(None)
        self.assertIn("flydsl_fused_", code)
        self.assertNotIn("flydsl_fused_", reference_code)
        if (flydsl_options or {}).get("FLYDSL_SPLIT_KV"):
            self.assertIn("kv_splits=KV_SPLITS", code)
        self.assertRegex(backward_code, r"flydsl_fused_\w*flex_attention_backward_")
        self.assertEqual("build_split_bwd_dq" in backward_code, bool(split))
        self.assertEqual("build_split_bwd_dkdv" in backward_code, bool(split))
        self.assertNotIn("flydsl_fused_", reference_backward_code)
        if flydsl_options:
            self.assertEqual(
                "single_pass=SINGLE_PASS" in backward_code,
                bool(flydsl_options.get("bwd_FLYDSL_WIDE_SINGLE_PASS")),
            )
        self._assert_close_to_fp64(output, reference, golden, atol=0.03, rtol=0.02)
        self.assertEqual(len(grads), 3)
        self.assertEqual(len(reference_grads), 3)
        self.assertEqual(len(golden_grads), 3)
        for grad, reference_grad, golden_grad in zip(
            grads, reference_grads, golden_grads, strict=True
        ):
            self._assert_close_to_fp64(
                grad, reference_grad, golden_grad, atol=0.04, rtol=0.03
            )
        if fp64_mask_mod is not None:
            expected = _attention_grads_fp64(
                query, key, value, grad_output, fp64_mask_mod
            )
            for grad, reference_grad, expected_grad in zip(
                grads, reference_grads, expected
            ):
                error = (grad.double() - expected_grad).norm()
                reference_error = (reference_grad.double() - expected_grad).norm()
                # FlyDSL must be about as accurate as Triton against fp64.
                self.assertLess(
                    error.item(),
                    1.5 * reference_error.item() + 1e-3 * expected_grad.norm().item(),
                )

    def _compare_created_mask(
        self,
        mask_mod,
        *,
        device,
        batch=1,
        q_heads=2,
        kv_heads=None,
        seq=256,
        seq_kv=None,
        qk_dim=128,
        v_dim=128,
        seed=0,
        transposed=False,
        mask_heads=1,
        scale=None,
        return_aux=False,
    ):
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            seq_kv=seq_kv,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=seed,
            transposed=transposed,
        )
        block_mask = create_block_mask(
            mask_mod,
            batch,
            mask_heads,
            seq,
            seq if seq_kv is None else seq_kv,
            device=device,
            BLOCK_SIZE=128,
        )
        return self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=kv_heads is not None and kv_heads != q_heads,
            return_aux=return_aux,
        )


_PARITY_BATCH, _PARITY_HEADS, _PARITY_SEQ = 2, 4, 256


class TestFlyDSLAccuracyAssertions(TestCase):
    @_subtests("case", ["shape", "dtype", "nan", "inf", "element", "norm"])
    def test_rejects_invalid_results(self, case):
        golden = torch.ones(4, dtype=torch.float64)
        reference = golden.float()
        actual = reference.clone()
        if case == "shape":
            actual = actual.view(2, 2)
        elif case == "dtype":
            actual = actual.double()
        elif case in ("nan", "inf"):
            actual[0] = float(case)
        elif case == "element":
            actual[0] = 2
        else:
            actual += 0.005
        with self.assertRaises(AssertionError):
            _FlyDSLFlexAttentionRuntimeMixin._assert_close_to_fp64(
                self, actual, reference, golden, atol=0.03, rtol=0.02
            )

    def test_accepts_matching_infinite_aux_rows(self):
        golden = torch.tensor([0, -float("inf")], dtype=torch.float64)
        reference = golden.float()
        _FlyDSLFlexAttentionRuntimeMixin._assert_close_to_fp64(
            self, reference.clone(), reference, golden, atol=0.03, rtol=0.02
        )


def _documents(generator, dtype, scale=1, offset=0):
    ids = torch.randint(0, 6, (_PARITY_SEQ,), generator=generator).sort().values
    return (ids * scale + offset).to(dtype)


# name: (mask_mod, captures(generator)); index captures hold negative and
# out-of-int32 values so narrowing or sign mistakes change the mask.
_MASK_PARITY_CASES = {
    "document_int64": (
        lambda b, h, q, kv, doc: doc[q] == doc[kv],
        lambda g: (_documents(g, torch.int64, 1 << 33, -(1 << 34)),),
    ),
    "document_int32_causal": (
        lambda b, h, q, kv, doc: (doc[q] == doc[kv]) & (q >= kv),
        lambda g: (_documents(g, torch.int32, 7, -20),),
    ),
    "document_int16": (
        lambda b, h, q, kv, doc: doc[q] == doc[kv],
        lambda g: (_documents(g, torch.int16, 10000, -25000),),
    ),
    "document_int8_sum_wraps": (
        lambda b, h, q, kv, doc: doc[q] + doc[kv] > 0,
        lambda g: (_documents(g, torch.int8, 40, -100),),
    ),
    "document_uint8_sum_wraps": (
        lambda b, h, q, kv, doc: doc[q] < doc[kv] + 10,
        lambda g: (_documents(g, torch.uint8, 50),),
    ),
    "uint8_invert": (
        lambda b, h, q, kv, doc: ~doc[kv] > 128,
        lambda g: (_documents(g, torch.uint8, 50),),
    ),
    "bool_flags": (
        lambda b, h, q, kv, flags: flags[q] | flags[kv] | (q == kv),
        lambda g: (torch.rand(_PARITY_SEQ, generator=g) > 0.5,),
    ),
    "bool_not_causal": (
        lambda b, h, q, kv, flags: ~flags[kv] & (q >= kv),
        lambda g: (torch.rand(_PARITY_SEQ, generator=g) > 0.5,),
    ),
    "bool_ordering": (
        lambda b, h, q, kv, flags: flags[q] < flags[kv],
        lambda g: (torch.rand(_PARITY_SEQ, generator=g) > 0.5,),
    ),
    "prefix_lm_int64": (
        lambda b, h, q, kv, prefix: (kv <= prefix[b]) | (q >= kv),
        lambda g: (torch.randint(0, _PARITY_SEQ, (_PARITY_BATCH,), generator=g),),
    ),
    "causal_padding": (
        lambda b, h, q, kv, lengths: (q >= kv) & (kv < lengths[b]),
        lambda g: (
            torch.randint(
                0, _PARITY_SEQ, (_PARITY_BATCH,), generator=g, dtype=torch.int32
            ),
        ),
    ),
    "nested_negative_offsets": (
        lambda b, h, q, kv, doc, offsets: doc[offsets[b] + q] == doc[kv],
        lambda g: (_documents(g, torch.int64, 1 << 33), torch.tensor([-64, 0])),
    ),
    "nested_documents": (
        lambda b, h, q, kv, doc: doc[doc[q] % _PARITY_SEQ] == doc[kv],
        lambda g: (_documents(g, torch.int64, 37, -(1 << 35)),),
    ),
    "document_starts": (
        lambda b, h, q, kv, doc, starts: kv >= starts[doc[q]],
        lambda g: (
            _documents(g, torch.int32),
            torch.randint(0, _PARITY_SEQ, (6,), generator=g),
        ),
    ),
    "alibi_bf16_threshold": (
        lambda b, h, q, kv, slopes: (q - kv).abs() * slopes[h] < 3.0,
        lambda g: (torch.rand(_PARITY_HEADS, generator=g).to(torch.bfloat16),),
    ),
    "alibi_fp16_threshold": (
        lambda b, h, q, kv, slopes: (q - kv).abs() * slopes[h] < 3.0,
        lambda g: (torch.rand(_PARITY_HEADS, generator=g).to(torch.float16),),
    ),
    "alibi_fp32_threshold": (
        lambda b, h, q, kv, slopes: (q - kv).abs().float() * slopes[h] <= 2.5,
        lambda g: (torch.rand(_PARITY_HEADS, generator=g),),
    ),
    "float_table": (
        lambda b, h, q, kv, table: table[q, kv] > 0.25,
        lambda g: (torch.randn(_PARITY_SEQ, _PARITY_SEQ, generator=g),),
    ),
    "transposed_float_table": (
        lambda b, h, q, kv, table: table[h, kv] < table[h, q],
        lambda g: (torch.randn(_PARITY_SEQ, _PARITY_HEADS, generator=g).t(),),
    ),
    "dilated": (lambda b, h, q, kv: (q - kv) % 4 == 0, lambda g: ()),
    "dilated_causal": (
        lambda b, h, q, kv: ((q - kv) % 3 == 0) & (q >= kv),
        lambda g: (),
    ),
    "strided_blocks": (lambda b, h, q, kv: (q // 64) >= (kv // 64), lambda g: ()),
    "trunc_divide": (
        lambda b, h, q, kv: torch.div(q - kv, 3, rounding_mode="trunc") == -1,
        lambda g: (),
    ),
    "fmod": (lambda b, h, q, kv: torch.fmod(q - kv, 5) == -2, lambda g: ()),
    "true_divide": (lambda b, h, q, kv: (q - kv) / 7.0 < 1.5, lambda g: ()),
    "where": (lambda b, h, q, kv: torch.where(q > 100, q >= kv, kv < 10), lambda g: ()),
    "minimum_maximum": (
        lambda b, h, q, kv: torch.minimum(q, kv) > torch.maximum(q - 64, kv * 0),
        lambda g: (),
    ),
    "clamp": (lambda b, h, q, kv: torch.clamp(q - kv, 0, 64) < 32, lambda g: ()),
    "shift_xor": (
        lambda b, h, q, kv: ((q >> 6) == (kv >> 6)) | (((q ^ kv) & 1) == 0),
        lambda g: (),
    ),
    "right_shift_by_index": (
        lambda b, h, q, kv: (kv - 128 >> (q % 40)) < -1,
        lambda g: (),
    ),
    "left_shift_by_index": (
        lambda b, h, q, kv: ((kv + 1 << (q % 40)) & 0xFF0) == 0,
        lambda g: (),
    ),
    "float_cast": (
        lambda b, h, q, kv: (q.to(torch.float32) * 0.5) >= kv.float(),
        lambda g: (),
    ),
    "int8_cast_wraps": (
        lambda b, h, q, kv: q.to(torch.int8) == kv.to(torch.int8),
        lambda g: (),
    ),
    "bf16_cast_rounds": (
        lambda b, h, q, kv: q.to(torch.bfloat16) > (kv + 1).to(torch.bfloat16),
        lambda g: (),
    ),
    "int64_constants": (
        lambda b, h, q, kv: q.to(torch.int64) * (1 << 33)
        >= kv.to(torch.int64) * (1 << 33) + (1 << 32),
        lambda g: (),
    ),
    "logical_ops": (
        lambda b, h, q, kv: torch.logical_xor(
            torch.logical_and(q >= kv, torch.logical_not(kv > 200)), q == 7
        ),
        lambda g: (),
    ),
    "negate": (lambda b, h, q, kv: -q < -kv + 3, lambda g: ()),
    "scalar_captures": (
        lambda b, h, q, kv, window, flag, threshold: (
            (q - kv < window) & flag | ((q - kv).float() < threshold)
        ),
        lambda g: (torch.tensor(96), torch.tensor(True), torch.tensor(-3.5)),
    ),
}


class TestFlyDSLFlexAttentionLowering(TestCase):
    @_subtests("name", list(_MASK_PARITY_CASES))
    def test_mask_lowering_matches_eager(self, name):
        mask_mod, make_captures = _MASK_PARITY_CASES[name]
        generator = torch.Generator().manual_seed(len(name))
        captures = make_captures(generator)
        program, reason = _lower_mask(mask_mod, *captures)
        self.assertIsNotNone(program, reason)
        buffers = [
            capture.reshape(-1) if not capture.dim() else capture
            for capture in captures
        ]
        indices = torch.stack(
            [
                torch.randint(0, extent, (256,), generator=generator)
                for extent in (_PARITY_BATCH, _PARITY_HEADS, _PARITY_SEQ, _PARITY_SEQ)
            ],
            dim=1,
        ).tolist()
        for b, h, q, kv in indices:
            expected = mask_mod(
                *(torch.tensor(i, dtype=torch.int32) for i in (b, h, q, kv)), *captures
            )
            self.assertEqual(
                _evaluate_mask_program(program, b, h, q, kv, buffers),
                bool(expected),
                msg=f"{name} at b={b} h={h} q={q} kv={kv}",
            )

    def test_mask_lowering_float_constant(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNotNone(program, reason)
        self.assertIn(("const_f32", -0.5), program.instructions)
        self.assertTrue(_evaluate_mask_program(program, 0, 0, 3, 2))
        self.assertFalse(_evaluate_mask_program(program, 0, 0, 2, 2))

    @_subtests(
        "mask_mod,captures,reason",
        _cases(
            float64_capture=(
                lambda b, h, q, kv, table: table[q] > 0.5,
                (torch.rand(8, dtype=torch.float64),),
                "captured buffers must be bool, int8",
            ),
            int32_constant_overflow=(
                lambda b, h, q, kv: q + (1 << 40) >= kv,
                (),
                "does not fit int32",
            ),
            int8_constant_overflow=(
                lambda b, h, q, kv: q.to(torch.int8) < 300,
                (),
                "300 does not fit int8",
            ),
            non_finite_constant=(
                lambda b, h, q, kv: q + float("inf") >= kv,
                (),
                "float constant inf is unsupported",
            ),
            float64_value=(
                lambda b, h, q, kv: q.double() > kv,
                (),
                "dtype torch.float64 is unsupported",
            ),
            unsupported_op=(
                lambda b, h, q, kv: torch.sin(q.float()) > 0,
                (),
                "operation aten.sin.default is unsupported",
            ),
            partial_index=(
                lambda b, h, q, kv, table: table[q].sum() > 0,
                (torch.zeros(8, 8),),
                "index rank mismatch",
            ),
        ),
    )
    def test_mask_lowering_rejects(self, mask_mod, captures, reason):
        program, message = _lower_mask(mask_mod, *captures)
        self.assertIsNone(program)
        self.assertIn(reason, message)

    @_subtests("dtype", [torch.uint16, torch.uint32, torch.float8_e4m3fn])
    def test_mask_lowering_rejects_capture_dtype(self, dtype):
        graph_module = _mask_graph(
            lambda b, h, q, kv, doc: doc[q] == doc[kv],
            torch.zeros(8, dtype=torch.int32),
        )
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_mask_graph(
                graph_module, [_FakeNode((8,), (1,), dtype=dtype)]
            )
        self.assertIsNone(program)
        self.assertIn("captured buffers must be bool, int8", reason)

    @_subtests(
        "dtype,elements",
        [
            subtest((torch.int64, 1 << 29), name="int64_4gib"),
            subtest((torch.int8, 1 << 31), name="int8_2g_elements"),
        ],
    )
    def test_mask_lowering_rejects_large_capture(self, dtype, elements):
        graph_module = _mask_graph(
            lambda b, h, q, kv, doc: doc[q] == doc[kv], torch.zeros(8, dtype=dtype)
        )
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_mask_graph(
                graph_module, [_FakeNode((elements,), (1,), dtype=dtype)]
            )
        self.assertIsNone(program)
        self.assertIn("smaller than 4 GiB", reason)
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_mask_graph(
                graph_module, [_FakeNode((elements // 2,), (1,), dtype=dtype)]
            )
        self.assertIsNotNone(program, reason)

    @_subtests(
        "mask,state_heads,block_sizes",
        [
            subtest(("none", 1, (64, 64)), name="noop_shared"),
            subtest(("causal", 4, (16, 32)), name="causal_per_q_head"),
            subtest(("document", 2, (128, 16)), name="document_per_kv_head"),
            subtest(("causal", 2, (384, 64)), name="causal_384x64"),
            subtest(("causal", 2, (96, 24)), name="causal_96x24"),
            subtest(("document", 4, (7, 192)), name="document_7x192"),
        ],
    )
    def test_mask_lowering_block_states(self, mask, state_heads, block_sizes):
        batch, q_heads, seq_q, seq_kv = 2, 4, 300, 200
        documents = torch.arange(max(seq_q, seq_kv), dtype=torch.int16) // 70
        masks = {
            "causal": (lambda b, h, q, kv: q >= kv, ()),
            "document": (lambda b, h, q, kv, docs: docs[q] == docs[kv], (documents,)),
        }
        program, captures = None, ()
        if mask != "none":
            fn, captures = masks[mask]
            program, reason = _lower_mask(fn, *captures)
            self.assertIsNotNone(program, reason)
        state_shape = (
            batch,
            state_heads,
            -(-seq_q // block_sizes[0]),
            -(-seq_kv // block_sizes[1]),
        )
        states = torch.randint(0, 3, state_shape, dtype=torch.int8)
        merged = add_block_states(
            program,
            shape=state_shape,
            strides=states.stride(),
            block_sizes=block_sizes,
            head_group=q_heads // state_heads,
        )
        self.assertEqual(merged.buffer_dtypes[-1], "int8")
        b, h, q, kv = (
            torch.arange(size).view([-1 if dim == axis else 1 for dim in range(4)])
            for axis, size in enumerate((batch, q_heads, seq_q, seq_kv))
        )
        keep = _evaluate_mask_program(merged, b, h, q, kv, (*captures, states))
        state = states[
            b, h // (q_heads // state_heads), q // block_sizes[0], kv // block_sizes[1]
        ]
        allowed = True if program is None else masks[mask][0](b, h, q, kv, *captures)
        expected = (state == 2) | ((state == 1) & allowed)
        self.assertEqual(keep.expand_as(expected), expected)

    @_subtests("count", [8, 16])
    def test_mask_lowering_many_captures(self, count):
        dtypes = (torch.int32, torch.int64, torch.int16, torch.uint8, torch.int8)
        generator = torch.Generator().manual_seed(count)
        captures = tuple(
            torch.randint(0, 3, (64,), generator=generator).to(dtypes[i % 5])
            for i in range(count)
        )

        def mask_mod(b, h, q, kv, *docs):
            allowed = q >= kv
            for index, doc in enumerate(docs):
                same = doc[q] == doc[kv]
                allowed = (allowed | same) if index % 3 == 2 else (allowed & same)
            return allowed

        program, reason = _lower_mask(mask_mod, *captures)
        self.assertIsNotNone(program, reason)
        self.assertEqual(program.buffer_count, count)
        self.assertEqual(
            program.buffer_dtypes,
            tuple(str(c.dtype).removeprefix("torch.") for c in captures),
        )
        for q, kv in torch.randint(0, 64, (64, 2), generator=generator).tolist():
            expected = mask_mod(
                *(torch.tensor(i, dtype=torch.int32) for i in (0, 0, q, kv)), *captures
            )
            self.assertEqual(
                _evaluate_mask_program(program, 0, 0, q, kv, captures), bool(expected)
            )

    @_subtests(
        "dtype,load",
        [
            subtest((torch.int32, ("load_i32", 0, (2,))), name="int32"),
            subtest((torch.int64, ("load", 0, (2,), "int64")), name="int64"),
            subtest((torch.bool, ("load", 0, (2,), "bool")), name="bool"),
            subtest((torch.bfloat16, ("load", 0, (2,), "bfloat16")), name="bf16"),
        ],
    )
    def test_mask_lowering_reads_native_capture_dtype(self, dtype, load):
        program, reason = _lower_mask(
            lambda b, h, q, kv, doc: doc[q] == doc[kv], torch.zeros(8, dtype=dtype)
        )
        self.assertIsNotNone(program, reason)
        self.assertEqual(program.instructions[0], load)
        self.assertFalse(any(op[0] == "cast" for op in program.instructions))

    def test_mask_lowering_direct_scalar_capture(self):
        indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
        graph_module = make_fx(
            lambda b, h, q, kv, w: (q >= kv) & (q - kv < w) & (kv + w >= q)
        )(*indices, torch.tensor(96, dtype=torch.int32))
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_mask_graph(
                graph_module, (_FakeNode((1,), (1,), dtype=torch.int32),)
            )
        self.assertIsNotNone(program, reason)
        (load,) = [op for op in program.instructions if op[0] == "load_i32"]
        self.assertEqual(program.instructions[load[2][0] - 4], ("const_i32", 0))
        self.assertTrue(_evaluate_mask_program(program, 0, 0, 95, 0, (96,)))
        self.assertFalse(_evaluate_mask_program(program, 0, 0, 96, 0, (96,)))
        self.assertTrue(_evaluate_mask_program(program, 0, 0, 96, 0, (97,)))

    @_subtests(
        "shape,device,reason",
        [
            subtest(((2,), "cuda", "is not scalar-lowered"), name="multi_element"),
            subtest(((1,), "cpu", "must live on the attention device"), name="cpu"),
        ],
    )
    def test_mask_lowering_rejects_direct_capture(self, shape, device, reason):
        indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
        graph_module = make_fx(lambda b, h, q, kv, w: q - kv < w)(
            *indices, torch.tensor(8, dtype=torch.int32)
        )
        capture = _FakeNode(shape, (1,), dtype=torch.int32, device=device)
        with V.set_graph_handler(_fake_graph()):
            program, message = lower_flydsl_mask_graph(graph_module, (capture,))
        self.assertIsNone(program)
        self.assertIn(reason, message)

    def test_mask_lowering_full_constant_respects_dtype(self):
        program, reason = lower_flydsl_mask_graph(
            _mask_graph(
                lambda b, h, q, kv: q
                > torch.full((), 3, dtype=torch.bool, device=q.device)
            ),
            (),
        )
        self.assertIsNotNone(program, reason)
        self.assertTrue(_evaluate_mask_program(program, 0, 0, 2, 0))

    @_subtests(
        "mask_mod,q,kv,expected",
        [
            subtest(
                (lambda b, h, q, kv: torch.add(q, 64, alpha=2) >= kv, 0, 100, True),
                name="add_alpha",
            ),
            subtest(
                (
                    lambda b, h, q, kv: torch.sub(q, 64, alpha=2) >= kv,
                    200,
                    100,
                    False,
                ),
                name="sub_alpha",
            ),
            subtest(
                (lambda b, h, q, kv: q + 8188 >= kv, 3, 8191, True),
                name="decode_boundary_inside",
            ),
            subtest(
                (lambda b, h, q, kv: q + 8188 >= kv, 2, 8191, False),
                name="decode_boundary_outside",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    95,
                    0,
                    True,
                ),
                name="window_boundary_inside",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    96,
                    0,
                    False,
                ),
                name="window_boundary_outside",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q - kv) % 8 == 1,
                    1,
                    8,
                    True,
                ),
                name="negative_lhs_remainder",
            ),
        ],
    )
    def test_mask_lowering_edge_cases(self, mask_mod, q, kv, expected):
        program, reason = lower_flydsl_mask_graph(
            _mask_graph(mask_mod),
            (),
        )
        self.assertIsNotNone(program, reason)
        self.assertEqual(
            _evaluate_mask_program(program, 0, 0, q, kv),
            expected,
        )

    @_subtests(
        "input_kwargs,expected",
        [
            subtest(
                (
                    {},
                    (64, 4, 512, 1024, 128, 128, 1),
                ),
                name="bf16_gqa",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 16, 256, 192),
                        "k_size": (1, 16, 256, 192),
                        "v_size": (1, 16, 256, 128),
                        "index_width": 1,
                        "mask_fn": lambda b, h, q, kv: q >= kv,
                    },
                    (16, 16, 256, 256, 192, 128, 1),
                ),
                name="qk192_v128",
            ),
            subtest(
                (
                    {
                        "q_size": (32, 64, 4, 128),
                        "k_size": (32, 4, 8192, 128),
                        "mask_heads": 4,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                    },
                    (64, 4, 4, 8192, 128, 128, 4),
                ),
                name="gqa_decode_q4",
            ),
        ],
    )
    def test_appends_supported_choices(self, input_kwargs, expected):
        keys = (
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "SEQ_Q",
            "SEQ_KV",
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "BLOCK_MASK_HEADS",
        )
        kwargs, reason = _fake_choice_result(
            _supported_fake_forward_inputs(**input_kwargs)
        )
        self.assertIsNotNone(kwargs, reason)
        self.assertEqual(tuple(kwargs[key] for key in keys), expected)
        self.assertNotIn("CAUSAL_PARTIAL_BLOCKS", kwargs)
        self.assertTrue(kwargs["MASK_PROGRAM"])
        self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
        self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    @_subtests(
        "input_kwargs,overrides,expected_reason",
        [
            subtest(
                (
                    {},
                    {"kv_num_blocks": _FakeNode([1, 1, 4], [4, 4, 1], torch.float32)},
                    "requires int32 BlockMask metadata",
                ),
                name="count_dtype",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 8, 256, 128),
                        "k_size": (1, 2, 256, 128),
                        "index_width": 2,
                        "mask_fn": lambda b, h, q, kv: q >= kv,
                    },
                    {"sparse_q_block_size": 256},
                    "requires BlockMask metadata that splits or merges into 128x128",
                ),
                name="q_block_size",
            ),
            subtest(
                (
                    {},
                    {"sparse_kv_block_size": 64},
                    "requires BlockMask metadata that splits or merges into 128x128",
                ),
                name="kv_block_size_64",
            ),
            subtest(
                (
                    {},
                    {"sparse_kv_block_size": 96},
                    "requires BlockMask metadata that splits or merges into 128x128",
                ),
                name="kv_block_size_96",
            ),
            subtest(
                (
                    {},
                    {"sparse_kv_block_size": 0},
                    "requires positive sparse block sizes",
                ),
                name="kv_block_size_0",
            ),
            subtest(
                (
                    {
                        "q_size": (32, 64, 4, 128),
                        "k_size": (32, 4, 8192, 128),
                        "mask_heads": 64,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                    },
                    {},
                    "requires prefill Sq >= 128",
                ),
                name="decode_per_q_head_mask",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 16, 1, 128),
                        "k_size": (1, 1, 1 << 24, 128),
                        "mask_heads": 1,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + (1 << 24) - 1 >= kv,
                    },
                    {},
                    "requires every per-head tensor slice to be smaller than 4 GiB",
                ),
                name="four_gib_head_slice",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 8, 256, 128),
                        "k_size": (1, 8, 256, 128),
                        "dtype": torch.float32,
                    },
                    {},
                    "supports BF16 and FP16 only",
                ),
                name="dtype",
            ),
        ],
    )
    def test_unsupported_choices_fall_back(
        self, input_kwargs, overrides, expected_reason
    ):
        inputs = _supported_fake_forward_inputs(**input_kwargs)
        inputs.update(overrides)
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn(expected_reason, reason)

    @_subtests(
        "head_dims,expected_reason",
        [
            subtest(((64, 64), None), name="d64"),
            subtest(((128, 64), None), name="qk128_v64"),
            subtest(((96, 96), None), name="d96"),
            subtest(((192, 192), None), name="d192"),
            subtest(((256, 128), None), name="qk256_v128"),
            subtest(((256, 256), None), name="d256"),
            subtest(((64, 128), None), name="qk64_v128"),
            subtest(((64, 256), None), name="qk64_v256"),
            subtest(((96, 128), None), name="qk96_v128"),
            subtest(((128, 256), None), name="qk128_v256"),
            subtest(((80, 80), "lowering pads narrower heads"), name="d80"),
            subtest(((160, 160), "lowering pads narrower heads"), name="d160"),
            subtest(((512, 512), None), name="d512"),
            subtest(((384, 384), None), name="d384"),
            subtest(((128, 512), None), name="qk128_v512"),
            subtest(((512, 64), None), name="qk512_v64"),
            subtest(((320, 320), "lowering pads narrower heads"), name="d320"),
            subtest(((576, 512), "head dims up to 512"), name="qk576_v512"),
            subtest(((128, 640), "head dims up to 512"), name="qk128_v640"),
        ],
    )
    def test_head_dim_gates(self, head_dims, expected_reason):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            forward_value_slice,
        )

        qk_dim, v_dim = head_dims
        for seq_q in (512, 4):
            forward_kwargs, reason = _fake_choice_result(
                _supported_fake_forward_inputs(
                    q_size=(1, 8, seq_q, qk_dim),
                    k_size=(1, 2, 1024, qk_dim),
                    v_size=(1, 2, 1024, v_dim),
                )
            )
            if expected_reason is None:
                self.assertIsNotNone(forward_kwargs, reason)
                value_slice = forward_value_slice(*head_dims)
                self.assertEqual(
                    (forward_kwargs["QK_HEAD_DIM"], forward_kwargs["V_HEAD_DIM"]),
                    (qk_dim, value_slice),
                )
                self.assertEqual(
                    forward_kwargs.get("VALUE_SLICES", 1), v_dim // value_slice
                )
            else:
                self.assertIsNone(forward_kwargs)
                self.assertIn(expected_reason, reason)

        inputs = _supported_fake_backward_inputs()
        qk_shape, v_shape = (1, 2, 256, qk_dim), (1, 2, 256, v_dim)
        for name, shape in (
            ("query", qk_shape),
            ("key", qk_shape),
            ("value", v_shape),
            ("out", v_shape),
            ("grad_out", v_shape),
        ):
            inputs[name] = _FakeNode(shape, _contiguous_stride(shape))
        config, reason = _backward_config_result(inputs)
        if expected_reason is None:
            self.assertIsNotNone(config, reason)
        else:
            self.assertIsNone(config)
            self.assertIn(expected_reason, reason)

    def test_wide_value_output_slice_over_4gib(self):
        # BSHD query head slices of 1 GiB; the V-256 output's span over 4 GiB.
        q_size = (1, 32, (1 << 18) + 1024, 64)
        inputs = _supported_fake_forward_inputs(
            q_size=q_size,
            k_size=(1, 4, 1024, 64),
            v_size=(1, 4, 1024, 256),
            mask_fn=lambda b, h, q, kv: kv >= 0,
        )
        inputs["query"] = _FakeNode(q_size, _bshd_stride(*q_size))
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn("per-head output slice", reason)

    @_subtests(
        "head_dims,expected",
        [
            subtest(((128, 128), (128, 128)), name="d128"),
            subtest(((64, 256), (64, 256)), name="qk64_v256"),
            subtest(((80, 80), (96, 96)), name="d80"),
            subtest(((80, 64), (96, 96)), name="qk80_v64"),
            subtest(((72, 128), (96, 128)), name="qk72_v128"),
            subtest(((160, 160), (192, 192)), name="d160"),
            subtest(((160, 128), (192, 128)), name="qk160_v128"),
            subtest(((112, 112), (128, 128)), name="d112"),
            subtest(((32, 32), (64, 64)), name="d32"),
            subtest(((64, 200), (64, 256)), name="qk64_v200"),
            subtest(((200, 200), (256, 256)), name="d200"),
            subtest(((512, 512), (512, 512)), name="d512"),
            subtest(((64, 384), (64, 384)), name="qk64_v384"),
            subtest(((320, 320), (384, 384)), name="d320"),
            subtest(((448, 448), (512, 512)), name="d448"),
            subtest(((300, 100), (384, 128)), name="qk300_v100"),
            subtest(((96, 300), (128, 384)), name="qk96_v300"),
            subtest(((448, 40), (512, 64)), name="qk448_v40"),
            subtest(((576, 512), None), name="qk576_v512"),
            subtest(((64, 513), None), name="qk64_v513"),
        ],
    )
    def test_kernel_head_dims(self, head_dims, expected):
        self.assertEqual(flex_flydsl_attention.kernel_head_dims(*head_dims), expected)

    def _check_packed_decode_union(self, metadata, num_kv_heads, num_kv_blocks):
        compiled = torch.compile(
            lambda *args: torch.ops.flydsl_flex_test.packed_decode_union(
                *args, num_kv_heads, num_kv_blocks
            ),
            fullgraph=True,
        )
        counts, indices, full_counts, full_indices = compiled(*metadata)
        partial_lists, full_lists = _reference_packed_decode_union(
            *metadata, num_kv_heads
        )
        batch = metadata[0].shape[0]
        self.assertEqual(counts.shape, (batch, num_kv_heads, 1))
        self.assertEqual(indices.shape, (batch, num_kv_heads, 1, num_kv_blocks))
        for actual_counts, actual_indices, expected_lists in (
            (counts, indices, partial_lists),
            (full_counts, full_indices, full_lists),
        ):
            # Tail slots past the count must still be valid block indices.
            self.assertTrue(
                ((actual_indices >= 0) & (actual_indices < num_kv_blocks)).all()
            )
            for (b, kv_head, q), expected in expected_lists.items():
                count = int(actual_counts[b, kv_head, q])
                self.assertEqual(count, len(expected))
                self.assertEqual(
                    actual_indices[b, kv_head, q, :count].tolist(), expected
                )

    @_subtests(
        "q_factor,kv_factor,seq_q,seq_kv",
        [
            subtest((2, 1, 1000, 1000), name="256x128_ragged"),
            subtest((1, 2, 1024, 1024), name="128x256"),
            subtest((2, 2, 1000, 1300), name="256x256_drops_tail"),
            subtest((1, 2, 256, 300), name="128x256_drops_tail"),
            subtest((1, 4, 513, 1100), name="128x512_drops_tail"),
            subtest((4, 1, 3, 640), name="512x128_decode"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_split_block_mask_metadata(self, q_factor, kv_factor, seq_q, seq_kv):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        user_q_blocks = -(-seq_q // (128 * q_factor))
        user_kv_blocks = -(-seq_kv // (128 * kv_factor))
        q_blocks, kv_blocks = -(-seq_q // 128), -(-seq_kv // 128)
        generator = torch.Generator().manual_seed(seq_q + seq_kv)
        metadata = _random_block_lists(generator, 2, 3, user_q_blocks, user_kv_blocks)
        expected_lists = []
        for counts, indices in (metadata[:2], metadata[2:]):
            lists = {}
            for b, h, q in torch.cartesian_prod(
                *(torch.arange(n) for n in (2, 3, q_blocks))
            ).tolist():
                row = indices[b, h, q // q_factor, : counts[b, h, q // q_factor]]
                lists[b, h, q] = [
                    block * kv_factor + tile
                    for block in row.tolist()
                    for tile in range(kv_factor)
                    if block * kv_factor + tile < kv_blocks
                ]
            expected_lists.append(lists)
        for device in devices:
            compiled = torch.compile(
                lambda *args: torch.ops.flydsl_flex_test.split_block_mask(
                    *args, q_factor, kv_factor, q_blocks, kv_blocks
                ),
                fullgraph=True,
            )
            outputs = compiled(*(t.to(device) for t in metadata))
            for (counts, indices), expected in zip(
                (outputs[:2], outputs[2:]), expected_lists
            ):
                self.assertEqual(counts.shape, (2, 3, q_blocks))
                self.assertTrue(((indices >= 0) & (indices < kv_blocks)).all())
                for (b, h, q), blocks in expected.items():
                    count = int(counts[b, h, q])
                    self.assertEqual(indices[b, h, q, :count].tolist(), blocks)

    @_subtests(
        "block_sizes,seq_q,seq_kv",
        [
            subtest(((64, 64), 1000, 1000), name="64x64_ragged"),
            subtest(((16, 16), 256, 384), name="16x16"),
            subtest(((32, 64), 300, 700), name="32x64_ragged"),
            subtest(((128, 64), 512, 1000), name="128x64"),
            subtest(((64, 128), 1000, 512), name="64x128"),
            subtest(((256, 32), 600, 500), name="256x32"),
            subtest(((16, 256), 300, 900), name="16x256"),
            subtest(((16, 64), 100, 1000), name="16x64_decode"),
            subtest(((96, 96), 1000, 1000), name="96x96_ragged"),
            subtest(((48, 24), 300, 700), name="48x24_ragged"),
            subtest(((8, 8), 256, 300), name="8x8"),
            subtest(((192, 192), 1000, 900), name="192x192_ragged"),
            subtest(((160, 96), 600, 1100), name="160x96"),
            subtest(((24, 256), 300, 900), name="24x256"),
            subtest(((256, 96), 600, 500), name="256x96"),
            subtest(((7, 300), 200, 1000), name="7x300"),
            subtest(((1000, 200), 1000, 700), name="1000x200"),
            subtest(((24, 48), 5, 1000), name="24x48_decode"),
            subtest(((112, 80), 700, 500), name="112x80"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_merge_block_mask_metadata(self, block_sizes, seq_q, seq_kv):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        user_q_blocks = -(-seq_q // block_sizes[0])
        user_kv_blocks = -(-seq_kv // block_sizes[1])
        generator = torch.Generator().manual_seed(seq_q + seq_kv + block_sizes[0])
        # Lists that disagree with any mask_mod, with one listed block past Sk,
        # the same lists with duplicate listings, and the full/partial
        # structure of a causal BlockMask.
        random_lists = _random_block_lists(
            generator, 2, 3, user_q_blocks, user_kv_blocks + 1
        )
        duplicated_lists = _add_duplicate_listings(generator, random_lists)
        causal = create_block_mask(
            lambda b, h, q, kv: q + seq_kv - seq_q >= kv,
            2,
            3,
            seq_q,
            seq_kv,
            device="cpu",
            BLOCK_SIZE=block_sizes,
        )
        causal_lists = (
            causal.kv_num_blocks,
            causal.kv_indices,
            causal.full_kv_num_blocks,
            causal.full_kv_indices,
        )
        cases = [
            (
                metadata,
                _reference_merged_block_mask(metadata, block_sizes, seq_q, seq_kv),
            )
            for metadata in (random_lists, duplicated_lists, causal_lists)
        ]
        self.assertTrue(
            seq_q < 128 or any(any(expected[1].values()) for _, expected in cases)
        )
        for device in devices:
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda *args: torch.ops.flydsl_flex_test.merge_block_mask(
                    *args, *block_sizes, seq_q, seq_kv
                ),
                fullgraph=True,
                dynamic=False,
            )
            for metadata, expected in cases:
                self._check_merged_block_mask(
                    compiled(*(t.to(device) for t in metadata)),
                    expected,
                    -(-seq_kv // 128),
                )

    def _check_merged_block_mask(self, outputs, expected, num_kv_blocks):
        *outputs, actual_states = outputs
        partial_lists, full_lists, states = expected
        self.assertEqual(actual_states.cpu(), states)
        for (counts, indices), lists in zip(
            (outputs[:2], outputs[2:]), (partial_lists, full_lists)
        ):
            self.assertEqual(indices.shape[-1], num_kv_blocks)
            self.assertTrue(((indices >= 0) & (indices < num_kv_blocks)).all())
            for (b, h, q), blocks in lists.items():
                count = int(counts[b, h, q])
                self.assertEqual(indices[b, h, q, :count].tolist(), blocks)

    @_subtests("block_sizes", [(64, 64), (64, 192)])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_merge_block_mask_metadata_wide(self, block_sizes):
        # More merged partial blocks per row than an int32 holds in 15 bits.
        seq_q = 100
        seq_kv = 128 * (flex_flydsl_attention.MAX_UNION_KV_BLOCKS + 200) - 50
        user_kv_blocks = -(-seq_kv // block_sizes[1])
        generator = torch.Generator().manual_seed(7)
        order = torch.randperm(user_kv_blocks, generator=generator).to(torch.int32)

        def row(blocks):
            padding = (0, user_kv_blocks - blocks.numel())
            return torch.nn.functional.pad(blocks, padding).view(1, 1, 1, -1)

        metadata = (
            torch.tensor([[[user_kv_blocks - 100]]], dtype=torch.int32),
            row(order[100:]),
            torch.tensor([[[100]]], dtype=torch.int32),
            row(order[:100]),
        )
        expected = _reference_merged_block_mask(metadata, block_sizes, seq_q, seq_kv)
        self.assertGreater(
            len(expected[0][0, 0, 0]), flex_flydsl_attention.MAX_UNION_KV_BLOCKS
        )
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda *args: torch.ops.flydsl_flex_test.merge_block_mask(
                    *args, *block_sizes, seq_q, seq_kv
                ),
                fullgraph=True,
            )
            self._check_merged_block_mask(
                compiled(*(t.to(device) for t in metadata)), expected, -(-seq_kv // 128)
            )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_packed_decode_union_metadata_handwritten(self):
        # Hq=4, Hkv=2, 6 KV blocks. KV head 0: block 0 full for both heads,
        # 3/4 full for one and partial for the other, 2 partial for one.
        # KV head 1: one head lists nothing (stale slots), so nothing is full.
        garbage = 99
        counts = torch.tensor([[[1], [2], [0], [1]]], dtype=torch.int32)
        indices = torch.tensor(
            [[[[2, garbage, garbage]], [[4, 3, garbage]], [[0, 0, 0]], [[5, 5, 5]]]],
            dtype=torch.int32,
        )
        full_counts = torch.tensor([[[3], [1], [0], [2]]], dtype=torch.int32)
        full_indices = torch.tensor(
            [[[[0, 3, 4]], [[0, garbage, garbage]], [[1, 1, 1]], [[1, 0, garbage]]]],
            dtype=torch.int32,
        )
        metadata = (counts, indices, full_counts, full_indices)
        self._check_packed_decode_union(metadata, 2, 6)
        partial_lists, full_lists = _reference_packed_decode_union(*metadata, 2)
        self.assertEqual(full_lists[0, 0, 0], [0])
        self.assertEqual(partial_lists[0, 0, 0], [2, 3, 4])
        self.assertEqual(full_lists[0, 1, 0], [])
        self.assertEqual(partial_lists[0, 1, 0], [0, 1, 5])

    @_subtests(
        "batch,q_heads,kv_heads,num_kv_blocks,capacity",
        [(1, 8, 2, 7, 7), (2, 8, 1, 16, 5), (3, 6, 3, 3, 2)],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_packed_decode_union_metadata_random(
        self, batch, q_heads, kv_heads, num_kv_blocks, capacity
    ):
        generator = torch.Generator().manual_seed(q_heads * 1000 + num_kv_blocks)
        metadata = _random_block_mask_metadata(
            generator, batch, q_heads, num_kv_blocks, capacity
        )
        self._check_packed_decode_union(metadata, kv_heads, num_kv_blocks)

    @_subtests(
        "q_size,k_size,mask_heads,block_size,expected",
        [
            subtest(((1, 8, 1, 128), (1, 2, 1024, 128), 8, 128, (2, 8)), name="union"),
            subtest(
                ((1, 8, 3, 128), (1, 2, 1000, 128), 8, 128, (2, 8)),
                name="union_ragged_kv",
            ),
            subtest(((1, 8, 1, 128), (1, 2, 1024, 128), 2, 128, None), name="kv_head"),
            subtest(((1, 8, 1, 128), (1, 2, 1024, 128), 1, 128, None), name="shared"),
            subtest(
                ((1, 8, 128, 128), (1, 2, 1024, 128), 8, 128, None), name="prefill"
            ),
            subtest(((1, 2, 1, 128), (1, 2, 1024, 128), 2, 128, None), name="mha"),
            subtest(((1, 8, 1, 128), (1, 2, 1024, 128), 8, 64, None), name="block64"),
            subtest(
                ((1, 8, 1, 128), (1, 2, 128 << 15, 128), 8, 128, None),
                name="too_many_kv_blocks",
            ),
        ],
    )
    def test_packed_decode_union_trigger(
        self, q_size, k_size, mask_heads, block_size, expected
    ):
        inputs = _supported_fake_forward_inputs(
            q_size=q_size, k_size=k_size, mask_heads=mask_heads, index_width=8
        )
        with V.set_graph_handler(_fake_graph()):
            shape = flex_flydsl_attention._packed_decode_union_shape(
                inputs["query"],
                inputs["key"],
                inputs["kv_num_blocks"],
                inputs["kv_indices"],
                inputs["full_kv_num_blocks"],
                inputs["full_kv_indices"],
                block_size,
            )
        self.assertEqual(shape, expected)

    def test_four_gib_kv_buffer_uses_rebased_head_slice(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(64, 64, 4, 128),
            k_size=(64, 4, 65536, 128),
            mask_heads=4,
            index_width=16,
            mask_fn=lambda b, h, q, kv: q + 65532 >= kv,
        )
        inputs["key"]._numel = 1 << 31
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNotNone(kwargs, reason)

    def test_identity_score_mod_passes_no_subgraph(self):
        kwargs, reason = _fake_choice_result(_supported_fake_forward_inputs())
        self.assertIsNotNone(kwargs, reason)
        self.assertFalse(kwargs["HAS_SCORE_MOD"])
        self.assertNotIn("subgraphs", kwargs)

    @_subtests(
        "dtype,value,prescaled,fused",
        [
            subtest((torch.bfloat16, 133120.0, 0.0, math.inf), name="bf16_133120"),
            subtest((torch.bfloat16, 135168.0, math.inf, math.inf), name="bf16_135168"),
            subtest((torch.float16, 32768.0, math.inf, 1.0), name="fp16_32768"),
            subtest((torch.float16, 49152.0, 0.0, math.inf), name="fp16_49152"),
        ],
    )
    def test_large_logit_exponent(self, dtype, value, prescaled, fused):
        for prescale_query, fused_backward, expected in (
            (True, True, prescaled),
            (False, True, fused),
            (False, False, 1.0),
        ):
            probability, grad_score = _large_logit_backward(
                value,
                dtype,
                prescale_query=prescale_query,
                fused_backward=fused_backward,
            )
            self.assertEqual(probability, expected)
            self.assertEqual(math.isnan(grad_score), math.isinf(expected))

    @_subtests("score_mod", [False, True])
    def test_fp16_accepts_large_scale(self, score_mod):
        inputs = _supported_fake_forward_inputs(
            q_size=(1, 8, 256, 128), k_size=(1, 8, 256, 128), dtype=torch.float16
        )
        inputs["scale"] = 0.7
        if score_mod:
            inputs["subgraph"] = SimpleNamespace(
                graph_module=torch.fx.symbolic_trace(
                    lambda score, b, h, q, kv: score * 2
                )
            )
            inputs["subgraph_buffer"] = object()
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNotNone(kwargs, reason)
        self.assertEqual(kwargs["DTYPE"], "fp16")
        self.assertEqual(kwargs["SM_SCALE"], 0.7)


_MOD_RENDER_TEMPLATE = """{{gen_defines()}}
def _score_mod(score, b, h, q_idx, kv_idx, aux):
    {{ modification(
        subgraph_number=0,
        output_name="score_out",
        score="score",
        b="b",
        h="h",
        m="q_idx",
        n="kv_idx",
    ) | indent_except_first(1) }}
    return score_out
{% if JOINT %}
def _joint_mod(score, b, h, q_idx, kv_idx, grad_score, aux):
    {{ modification(
        subgraph_number=1,
        output_name="grad_out",
        score="score",
        b="b",
        h="h",
        m="q_idx",
        n="kv_idx",
        grad_score_mod="grad_score",
    ) | indent_except_first(1) }}
    return grad_out
{% endif %}
MOD_KEY = "{{ mod_key() }}"
AUX_SLOTS = {{ aux_slots() }}
"""


def _trace_mod(fn, captures, *, joint=False):
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    example = [torch.zeros((), dtype=torch.bfloat16)]
    example += [torch.zeros((), dtype=torch.int32) for _ in range(4)]
    if joint:
        example.append(torch.zeros((), dtype=torch.bfloat16))
    captures = [
        torch.zeros(capture[0], dtype=_capture_dtype(capture)) for capture in captures
    ]

    def joint_fn(score, b, h, m, n, grad, *buffers):
        score = score.detach().requires_grad_()
        with torch.enable_grad():
            out = fn(score, b, h, m, n, *buffers)
        return torch.autograd.grad(out, score, grad)[0]

    from torch._inductor.decomposition import select_decomp_table

    with TransformGetItemToIndex():
        return make_fx(
            joint_fn if joint else fn, decomposition_table=select_decomp_table()
        )(*example, *captures)


def _capture_dtype(capture):
    return capture[2] if len(capture) > 2 else torch.float32


def _render_mod(fn, captures=(), *, joint=False):
    """Render a score_mod (and its joint) for ``(size, stride[, dtype])`` captures."""
    from torch._inductor.codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import Buffer, FixedLayout, InputBuffer, TensorBox
    from torch._inductor.kernel.flex.common import (
        build_subgraph_module_buffer,
        create_placeholder,
    )

    template = FlyDSLFlexTemplate(name="flydsl_mod_render", source=_MOD_RENDER_TEMPLATE)
    device = torch.device("cpu")
    graph = GraphLowering(make_fx(lambda: torch.zeros(2, 3))())
    with V.set_graph_handler(graph):
        placeholders = [
            create_placeholder(name, dtype, device)
            for name, dtype in (
                ("score", torch.bfloat16),
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            )
        ]
        buffers = [
            TensorBox.create(
                InputBuffer(
                    name=f"capture{index}",
                    layout=FixedLayout(
                        device,
                        _capture_dtype(capture),
                        list(capture[0]),
                        list(capture[1]),
                    ),
                )
            )
            for index, capture in enumerate(captures)
        ]
        subgraphs = [
            build_subgraph_module_buffer(
                placeholders + buffers, _trace_mod(fn, captures)
            )
        ]
        if joint:
            subgraphs.append(
                build_subgraph_module_buffer(
                    placeholders
                    + [create_placeholder("grad_score_mod", torch.bfloat16, device)]
                    + buffers,
                    _trace_mod(fn, captures, joint=True),
                )
            )
        output = Buffer(
            name="buf_out", layout=FixedLayout(device, torch.float32, [1], [1])
        )
        return template.render_kernel(
            "flydsl_mod_render",
            [],
            output,
            subgraphs=subgraphs,
            aux_buffers=[buffer.data.data for buffer in buffers],
            JOINT=joint,
        )


def _alibi_mod(score, b, h, q_idx, kv_idx):
    return score + (kv_idx - q_idx) * (h + 1) * 0.125


def _softcap_mod(score, b, h, q_idx, kv_idx):
    return 30.0 * torch.tanh(score / 30.0)


def _relative_position_mod(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx).abs() * -0.0625


def _relative_bias_table_mod(score, b, h, q_idx, kv_idx, table):
    return score + table[h, q_idx - kv_idx + 255]


def _captured_bias_mod(score, b, h, q_idx, kv_idx, bias):
    return score + bias[b, h, q_idx, kv_idx]


def _head_slope_mod(score, b, h, q_idx, kv_idx, slopes):
    return score + slopes[h] * (kv_idx - q_idx)


_DIRECTIONS = [subtest(False, name="forward"), subtest(True, name="backward")]
_RUNTIME_SCORE_MODS = (
    "alibi",
    "softcap",
    "relative_position",
    "relative_bias_table",
    "relative_bias_table_bf16",
    "captured_bias",
    "head_slopes",
)


def _runtime_score_mod(name, device, *, heads, seq, batch=1):
    torch.manual_seed(7)
    if name == "alibi":
        return _alibi_mod
    if name == "softcap":
        return _softcap_mod
    if name == "relative_position":
        return _relative_position_mod
    if name.startswith("relative_bias_table"):
        dtype = torch.bfloat16 if name.endswith("bf16") else torch.float32
        table = torch.randn(heads, 2 * seq - 1, device=device, dtype=dtype)
        return (
            lambda score, b, h, q_idx, kv_idx: score
            + table[h, q_idx - kv_idx + seq - 1]
        )
    if name == "captured_bias":
        bias = torch.randn(batch, heads, seq, seq, device=device)
        return lambda score, b, h, q_idx, kv_idx: score + bias[b, h, q_idx, kv_idx]
    if name == "head_slopes":
        slopes = torch.rand(heads, device=device)
        return lambda score, b, h, q_idx, kv_idx: score + slopes[h] * (kv_idx - q_idx)
    raise AssertionError(name)


_SCORE_MOD_SHAPES = {
    "gqa": {"q_heads": 8, "kv_heads": 2},
    "mqa": {"q_heads": 4, "kv_heads": 1},
    "cross": {"seq_kv": 512},
    "fp16": {"dtype": torch.float16},
    "d64": {"qk_dim": 64, "v_dim": 64},
    "v64": {"v_dim": 64},
}


# Paired with alternating forward and backward directions; only the forward
# checks the LSE of empty rows.
_RAGGED_CASES = [
    subtest((1000, 1000, 2, 2, "dense", None), name="1000_dense"),
    subtest((1000, 1000, 2, 2, "causal", None), name="1000_causal"),
    subtest((1000, 1000, 2, 2, "empty_rows", None), name="1000_empty_rows"),
    subtest((1000, 1000, 4, 2, "causal", _alibi_mod), name="1000_gqa_alibi"),
    subtest((1, 77, 2, 2, "dense", None), name="1_77_dense"),
    subtest((1, 77, 4, 1, "bottom_right", None), name="1_77_mqa_causal"),
    subtest((4, 300, 8, 2, "bottom_right", None), name="4_300_gqa_decode"),
    subtest((333, 4097, 8, 2, "dense", None), name="333_4097_gqa_dense"),
    subtest(
        (333, 4097, 8, 2, "window", _softcap_mod), name="333_4097_gqa_window_softcap"
    ),
    subtest((4097, 333, 2, 2, "causal", None), name="4097_333_causal"),
    subtest((130, 129, 2, 1, "causal", None), name="130_129_mqa_causal"),
]


_HEAD_DIM_CASES = [
    subtest(({}, "dense", None, False), name="dense"),
    subtest(({}, "causal", None, False), name="causal"),
    subtest(({"q_heads": 8, "kv_heads": 2}, "causal", None, False), name="gqa"),
    subtest(({"seq_kv": 640}, "bottom_right", None, False), name="cross"),
    subtest(({"seq_q": 1000, "seq_kv": 1300}, "window", None, False), name="ragged"),
    subtest(({"dtype": torch.float16}, "causal", None, False), name="fp16"),
    subtest(({}, "causal", _alibi_mod, False), name="alibi"),
    subtest(({"q_heads": 4, "kv_heads": 2}, "dense", _softcap_mod, True), name="dlse"),
    subtest(({"seq_q": 333, "seq_kv": 333}, "document", None, True), name="document"),
]


def _head_dim_cases(*names):
    return [case for case in _HEAD_DIM_CASES if case.name in names]


# Paired with alternating forward and backward directions.
_BLOCK_SIZE_CASES = [
    subtest(((256, 128), 1024, 1024, "causal"), name="256x128_causal"),
    subtest(((128, 256), 1000, 300, "bottom_right"), name="128x256_drops_tail"),
    subtest(((256, 256), 1000, 1300, "window"), name="256x256_ragged_window"),
    subtest(((256, 256), 1024, 1024, "block_sparse"), name="256x256_block_sparse"),
    subtest(((256, 128), 4, 1000, "bottom_right"), name="256x128_decode"),
    subtest(((64, 128), 1000, 1100, "block_sparse_mask"), name="64x128_sparse_mask"),
    subtest(((128, 512), 1000, 1100, "block_sparse"), name="128x512_block_sparse"),
    subtest(((32, 32), 1000, 1000, "document"), name="32x32_ragged_document"),
    subtest(((64, 16), 1, 4096, "block_sparse_mask"), name="64x16_sparse_decode"),
    subtest(((48, 24), 1000, 1300, "window"), name="48x24_ragged_window"),
    subtest(((192, 192), 1000, 1000, "causal"), name="192x192_ragged_causal"),
    subtest(((16, 32), 4, 1000, "bottom_right"), name="16x32_decode"),
]


def _ragged_mask_mod(kind, seq_q, seq_kv):
    offset = seq_kv - seq_q
    if kind == "dense":
        return lambda b, h, q, kv: q >= 0
    if kind == "causal":
        return lambda b, h, q, kv: q >= kv
    if kind == "empty_rows":
        return lambda b, h, q, kv: kv + 200 <= q
    if kind == "bottom_right":
        return lambda b, h, q, kv: q + offset >= kv
    if kind == "window":
        return lambda b, h, q, kv: (q + offset >= kv) & (q + offset - kv < 1000)
    if kind == "document":
        return lambda b, h, q, kv: (q // 300 == kv // 300) & (q >= kv)
    if kind == "block_sparse_mask":
        # Disagrees with the random lists of a block-sparse BlockMask.
        return lambda b, h, q, kv: (q + 2 * kv + h) % 3 != 0
    raise AssertionError(kind)


def _read_two_ways_mod(score, b, h, q_idx, kv_idx, positions):
    return score + positions[q_idx] - positions[kv_idx]


_BIAS_SIZE = (2, 4, 256, 256)
_BIAS_NUMEL = 2 * 4 * 256 * 256


class TestFlyDSLFlexModCodegen(TestCase):
    @_subtests(
        "mod,captures,expected,aux_slots",
        _cases(
            alibi=(_alibi_mod, (), ["fdu.sub(", "fdu.const_f32(float('0.125'))"], ()),
            softcap=(_softcap_mod, (), ["fdu.tanh(", "fdu.truediv("], ()),
            relative_position=(_relative_position_mod, (), ["fdu.abs("], ()),
            relative_bias_table=(
                _relative_bias_table_mod,
                [((4, 511), (511, 1))],
                ["fdu.where(fdu.lt(", "h, q_idx, kv_idx)"],
                ((0, (1, 511, 0, 0), 4 * 511),),
            ),
            relative_bias_table_bf16=(
                _relative_bias_table_mod,
                [((4, 511), (511, 1), torch.bfloat16)],
                ["h, q_idx, kv_idx)"],
                ((0, (1, 511, 0, 0), 4 * 511),),
            ),
            captured_bias=(
                _captured_bias_mod,
                [(_BIAS_SIZE, (262144, 65536, 256, 1))],
                ["aux[0](b, h, q_idx, kv_idx)"],
                ((0, (262144, 65536, 256, 1), _BIAS_NUMEL),),
            ),
            captured_bias_transposed_fp16=(
                _captured_bias_mod,
                [(_BIAS_SIZE, (262144, 65536, 1, 256), torch.float16)],
                ["aux[0](b, h, q_idx, kv_idx)"],
                ((0, (262144, 65536, 1, 256), _BIAS_NUMEL),),
            ),
            head_slopes=(
                _head_slope_mod,
                [((4,), (1,))],
                ["aux[0](b, h, q_idx, kv_idx)"],
                ((0, (0, 1, 0, 0), 4),),
            ),
            read_two_ways=(
                _read_two_ways_mod,
                [((256,), (1,))],
                ["aux[0](", "aux[1]("],
                ((0, (0, 0, 1, 0), 256), (0, (0, 0, 0, 1), 256)),
            ),
            **{
                f"{count}_tables": (
                    lambda score, b, h, q_idx, kv_idx, *tables: score
                    + sum(table[h] for table in tables),
                    [((4,), (1,))] * count,
                    [f"aux[{count - 1}](b, h, q_idx, kv_idx)"],
                    tuple((index, (0, 1, 0, 0), 4) for index in range(count)),
                )
                for count in (8, 16)
            },
        ),
    )
    def test_render(self, mod, captures, expected, aux_slots):
        kernel, source = _render_mod(mod, captures)
        self.assertIn("def _score_mod(score, b, h, q_idx, kv_idx, aux):", source)
        self.assertIn(
            "import torch._inductor.codegen.flydsl.flydsl_mod_runtime", source
        )
        for snippet in expected:
            self.assertIn(snippet, source)
        self.assertEqual(kernel.aux_slots(), aux_slots)
        compile(source, "<flydsl_mod>", "exec")

    def test_softcap_joint(self):
        _, source = _render_mod(_softcap_mod, joint=True)
        joint = source.split("def _joint_mod", 1)[1]
        self.assertIn("grad_score", joint)
        self.assertIn("fdu.tanh(", joint)
        compile(source, "<flydsl_mod>", "exec")

    @_subtests(
        "mod,captures,error",
        _cases(
            erfinv=(
                lambda score, b, h, q_idx, kv_idx: torch.special.erfinv(score),
                (),
                "erfinv",
            )
        ),
    )
    def test_rejects(self, mod, captures, error):
        with self.assertRaisesRegex(NotImplementedError, error):
            _render_mod(mod, captures)

    def test_mod_key(self):
        def cap50(score, b, h, q_idx, kv_idx):
            return 50.0 * torch.tanh(score / 50.0)

        key = _render_mod(_softcap_mod)[0].mod_key()
        self.assertEqual(key, _render_mod(_softcap_mod)[0].mod_key())
        self.assertNotEqual(key, _render_mod(cap50)[0].mod_key())

    def test_codegen_does_not_import_flydsl(self):
        import subprocess
        import sys

        script = (
            "import sys\n"
            "import torch._inductor.codegen.flydsl.flydsl_flex_kernel\n"
            "assert not any(m.split('.')[0] == 'flydsl' for m in sys.modules)\n"
        )
        subprocess.check_call([sys.executable, "-c", script])


def _document_mask_graph(count):
    """A causal mask graph over ``count`` int64 document captures and their nodes."""
    docs = tuple(torch.zeros(1024, dtype=torch.int64) for _ in range(count))
    graph = _mask_graph(
        lambda b, h, q, kv, *docs: functools.reduce(
            operator.and_, (doc[q] == doc[kv] for doc in docs), q >= kv
        ),
        *docs,
    )
    return SimpleNamespace(graph_module=graph), [_capture_node(doc) for doc in docs]


def _window_capture_mask_graph():
    """A causal window over a 0-d int64 capture of its width, and its node."""
    window = torch.tensor(256, dtype=torch.int64)
    graph = _mask_graph(lambda b, h, q, kv, w: (q >= kv) & (q - kv < w), window)
    return SimpleNamespace(graph_module=graph), [_capture_node(window)]


# (QK, V) head dims of the 4-wave dual-wave forward; 160 and 224 also run on K/V
# read in place, at the padded Q and O widths of the main forward.
_WIDE_HEAD_DIMS = (
    (64, 256),
    (128, 256),
    (192, 128),
    (192, 192),
    (256, 128),
    (256, 256),
)
_NATIVE_WIDE_HEAD_DIMS = {(160, 160): (192, 192), (224, 224): (256, 256)}
_DUALWAVE_INPUT_NAMES = (
    "query",
    "key",
    "value",
    "logsumexp",
    "max_scores",
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
)
_SCALED_SCORE_MOD = SimpleNamespace(
    graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score * 2)
)
_SOFTCAP_SCORE_MOD = SimpleNamespace(
    graph_module=torch.fx.symbolic_trace(
        lambda score, b, h, q, kv: 30.0 * torch.tanh(score / 30.0)
    )
)


class TestFlyDSLDualWaveForwardConfig(TestCase):
    @_subtests(
        "seq_kv,head_dims",
        [
            subtest((1024, (128, 128)), name="aligned"),
            subtest((1000, (128, 128)), name="tail_104"),
            subtest((1050, (128, 128)), name="tail_26"),
            subtest((1025, (128, 128)), name="tail_1"),
            subtest((1050, (64, 128)), name="tail_26_qk64_v128"),
            subtest((1000, (96, 96)), name="tail_104_d96"),
            subtest((1050, (192, 192)), name="tail_26_d192"),
            subtest((1000, (256, 128)), name="tail_104_qk256_v128"),
        ],
    )
    def test_symbolic_sizes_reach_the_dualwave_at_runtime(self, seq_kv, head_dims):
        graph, (batch, seq_q, length_kv) = _symbolic_graph(3, 1000, seq_kv)

        def bshd(heads, sequence, dim=head_dims[0]):
            size = [batch, heads, sequence, dim]
            stride = [sequence * heads * dim, dim, heads * dim, 1]
            return _FakeNode(size, stride, numel=sympy.Mul(*size))

        inputs = _dualwave_fake_inputs(
            block_mask_is_dense=True, mask_fn=lambda b, h, q, kv: q >= 0
        )
        inputs.update(
            query=bshd(8, seq_q),
            key=bshd(8, length_kv),
            value=bshd(8, length_kv, head_dims[1]),
        )
        config, reason = _dualwave_choice_result(
            inputs, graph=graph, wide=_dualwave_wide(head_dims)
        )
        self.assertIsNotNone(config, reason)
        self.assertEqual(
            [config[name] for name in ("BATCH_SIZE", "SEQ_LEN", "KV_SEQ_LEN")],
            [None, None, None],
        )
        qk, v = head_dims
        self.assertEqual(config["Q_STRIDE"], (("seq", 8 * qk), qk, 8 * qk, 1))
        self.assertEqual(config["O_STRIDE"], (("seq", 8 * v), v, 8 * v, 1))
        self.assertTrue(config["RUNTIME_SIZES"])
        self.assertEqual(
            config["PLAN"],
            {
                "mask_last_tile": seq_kv % 128 != 0,
                "mask_second_last_tile": 0 < seq_kv % 128 < 64,
            },
        )

    @_subtests(
        "input_kwargs,overrides,expected_reason",
        [
            subtest(
                (
                    {},
                    {
                        "query": _FakeNode(
                            [1, 8, 1024, 128], [8 * 1024 * 132, 1024 * 132, 132, 1]
                        )
                    },
                    "multiples of 8 elements",
                ),
                name="unaligned_rows",
            ),
            *(
                subtest(
                    ({"qk_dim": qk, "v_dim": v}, {}, "(QK, V) head dims"),
                    name=f"qk{qk}_v{v}",
                )
                for qk, v in (
                    (64, 64),
                    (192, 128),
                    (192, 192),
                    (256, 128),
                    (256, 256),
                    (128, 256),
                    (128, 64),
                    (96, 128),
                )
            ),
            subtest(
                (
                    {"qk_dim": 96},
                    {"key": _FakeNode([1, 8, 1024, 80], _bshd_stride(1, 8, 1024, 80))},
                    "(QK, V) head dims",
                ),
                name="key_narrower_than_query",
            ),
            subtest(({"seq_q": 256}, {}, "Sk >= 384"), name="too_short"),
            subtest(
                ({"seq_q": 1024, "seq_kv": 256}, {}, "Sk >= 384"),
                name="cross_kv_too_short",
            ),
            subtest(
                ({"seq_q": 1000, "seq_kv": 383}, {}, "Sk >= 384"),
                name="ragged_kv_too_short",
            ),
            subtest(
                ({"seq_q": 4, "seq_kv": 4096}, {}, "Sq < 128 to the decode forward"),
                name="decode",
            ),
            subtest(
                ({}, {"sparse_kv_block_size": 64}, "sparse Q/KV block sizes of 128"),
                name="block_size_64",
            ),
            subtest(
                ({}, {"subgraph": _SCALED_SCORE_MOD}, "only without a BlockMask"),
                name="score_mod_with_causal_block_mask",
            ),
            subtest(
                (
                    {"mask_fn": lambda b, h, q, kv: (q >= kv) & (q - kv < 256)},
                    {"subgraph": _SOFTCAP_SCORE_MOD},
                    "only without a BlockMask",
                ),
                name="score_mod_with_window",
            ),
            subtest(
                ({"mask_heads": 3}, {}, "BlockMask rows per batch and head"),
                name="block_mask_heads_3",
            ),
            subtest(
                (
                    {"seq_q": 256, "seq_kv": 128 * 8193, "partial_width": 1},
                    {},
                    "at most 8192 128-key blocks",
                ),
                name="too_many_kv_blocks",
            ),
            subtest(
                (
                    {"block_mask_is_dense": True},
                    {
                        "subgraph": _SCALED_SCORE_MOD,
                        "score_mod_other_buffers": [
                            _FakeNode([1024], [1], torch.float16) for _ in range(5)
                        ],
                    },
                    "at most 4 captures",
                ),
                name="score_capture_table",
            ),
            subtest(
                (
                    {},
                    dict(
                        zip(
                            ("mask_graph", "mask_mod_other_buffers"),
                            _document_mask_graph(5),
                        )
                    ),
                    "at most 4 captures",
                ),
                name="mask_capture_table",
            ),
            subtest(
                ({"q_heads": 8, "kv_heads": 3}, {}, "Hq divisible by Hkv"),
                name="gqa_indivisible",
            ),
        ],
    )
    def test_unsupported_inputs_are_not_offered(
        self, input_kwargs, overrides, expected_reason
    ):
        inputs = _dualwave_fake_inputs(**input_kwargs)
        inputs.update(overrides)
        kwargs, reason = _dualwave_choice_result(inputs, allow_causal_block_mask=True)
        self.assertIsNone(kwargs)
        self.assertIn(expected_reason, reason)

    @_subtests(
        "input_kwargs,overrides,expected_reason",
        _cases(
            fp32=({"dtype": torch.float32}, {}, "BF16 or FP16"),
            mixed_dtypes=(
                {"dtype": torch.float16},
                {"value": _FakeNode([1, 8, 1024, 128], _bshd_stride(1, 8, 1024, 128))},
                "of one dtype",
            ),
            head_dim_64=({"qk_dim": 64}, {}, "(QK, V) head dims"),
            score_mod_with_block_mask=(
                {},
                {"subgraph": _SCALED_SCORE_MOD},
                "only without a BlockMask",
            ),
        ),
    )
    def test_dualwave_limits_hold_if_shared_gate_widens(
        self, input_kwargs, overrides, expected_reason
    ):
        inputs = _dualwave_fake_inputs(**input_kwargs)
        inputs.update(overrides)
        permissive = {"Q_STRIDE": None, "K_STRIDE": None, "V_STRIDE": None}
        with mock.patch.object(
            flex_flydsl_attention,
            "_get_flydsl_common_config",
            return_value=(permissive, ""),
        ):
            kwargs, reason = _dualwave_choice_result(
                inputs, allow_causal_block_mask=True
            )
        self.assertIsNone(kwargs)
        self.assertIn("dual-wave forward", reason)
        self.assertIn(expected_reason, reason)

    @_subtests(
        "option,name",
        [
            ("dualwave_causal", "FLYDSL_DUALWAVE_CAUSAL"),
            ("head_slow_order", "FLYDSL_HEAD_SLOW_ORDER"),
            ("dualwave_wide", "FLYDSL_DUALWAVE_WIDE"),
            ("single_pass", "FLYDSL_WIDE_SINGLE_PASS"),
        ],
    )
    def test_invalid_bool_option(self, option, name):
        with self.assertRaisesRegex(ValueError, f"{name} must be a bool"):
            flex_flydsl_attention.create_flydsl_flex_attention_kernel(
                **dict.fromkeys(
                    (
                        "query",
                        "key",
                        "value",
                        "kv_num_blocks",
                        "kv_indices",
                        "full_kv_num_blocks",
                        "full_kv_indices",
                        "subgraph",
                        "mask_graph",
                        "score_mod_other_buffers",
                        "mask_mod_other_buffers",
                        "scale",
                        "sparse_q_block_size",
                        "sparse_kv_block_size",
                        "subgraph_buffer",
                        "mask_graph_buffer",
                    )
                ),
                **{option: 1},
            )

    def _assert_dualwave_builder_rejects(self, overrides, expected):
        if importlib.util.find_spec("flydsl") is None:
            self.skipTest("requires the FlyDSL package")
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            build_flex_attn_fwd_dualwave_module,
        )

        kwargs = {
            "batch_size": 1,
            "num_q_heads": 8,
            "num_kv_heads": 8,
            "seq_len": 1024,
            "scale": 128**-0.5,
            "causal": True,
            **overrides,
        }
        with self.assertRaisesRegex(ValueError, expected):
            build_flex_attn_fwd_dualwave_module(**kwargs)

    @_subtests(
        "overrides,expected",
        _cases(
            head_dim=({"head_dim": 64}, r"\(QK, V\) head dims"),
            v_head_dim=({"head_dim": 128, "v_head_dim": 64}, r"\(QK, V\) head dims"),
            unaligned_row_stride=(
                {"v_stride": (1024 * 8 * 132, 132, 8 * 132, 1)},
                "value strides .* multiples of 8",
            ),
            short=({"seq_len": 256}, "Sk >= 384"),
            short_kv=({"kv_seq_len": 256}, "Sk >= 384"),
            empty_q=({"seq_len": 0}, "Sq > 0"),
            dtype=({"dtype_str": "fp32"}, "bf16/f16"),
            gqa_ratio=({"num_kv_heads": 3}, "Hq % Hkv"),
            strided_head_dim=(
                {"q_stride": (1024 * 8 * 256, 256, 8 * 256, 2)},
                r"query strides \(B, H, S, 1\)",
            ),
        ),
    )
    def test_dualwave_builder_validates_contract(self, overrides, expected):
        self._assert_dualwave_builder_rejects(overrides, expected)

    @_subtests(
        "input_kwargs,overrides,expected_reason",
        _cases(
            d64_d64=({"qk_dim": 64, "v_dim": 64}, {}, "(QK, V) head dims"),
            d128_d64=({"qk_dim": 128, "v_dim": 64}, {}, "(QK, V) head dims"),
            qk192_v128=({"qk_dim": 192, "v_dim": 128}, {}, "(QK, V) head dims"),
            softcap_score_mod=(
                {},
                {"subgraph": _SOFTCAP_SCORE_MOD},
                "only without a BlockMask",
            ),
            bf16_score_capture=(
                {},
                {
                    "subgraph": _SCALED_SCORE_MOD,
                    "score_mod_other_buffers": [_FakeNode([8], [1], torch.bfloat16)],
                },
                "only without a BlockMask",
            ),
            fp16_score_capture=(
                {},
                {
                    "subgraph": _SCALED_SCORE_MOD,
                    "score_mod_other_buffers": [_FakeNode([8], [1], torch.float16)],
                },
                "only without a BlockMask",
            ),
            scalar_score_capture=(
                {},
                {
                    "subgraph": _SCALED_SCORE_MOD,
                    "score_mod_other_buffers": [
                        _FakeNode([], [], torch.float32, numel=1)
                    ],
                },
                "only without a BlockMask",
            ),
        ),
    )
    def test_dualwave_rejects_inputs_the_upstream_forward_accepts(
        self, input_kwargs, overrides, expected_reason
    ):
        inputs = _dualwave_fake_inputs(**input_kwargs)
        inputs.update(overrides)
        # Upstream registration realizes 0-d captures, which the fake graph can't run.
        if all(buf.get_size() for buf in inputs["score_mod_other_buffers"]):
            upstream_inputs = dict(inputs, subgraph_buffer=object())
            del upstream_inputs["block_mask_is_dense"]
            upstream_kwargs, upstream_reason = _fake_choice_result(upstream_inputs)
            self.assertIsNotNone(upstream_kwargs, upstream_reason)
        kwargs, reason = _dualwave_choice_result(inputs, allow_causal_block_mask=True)
        self.assertIsNone(kwargs)
        self.assertIn("dual-wave forward", reason)
        self.assertIn(expected_reason, reason)

    @_subtests(
        "input_kwargs,expected",
        _cases(
            causal_longer_q=({"seq_q": 2048, "seq_kv": 1024}, True),
            causal_longer_kv=({"seq_q": 128, "seq_kv": 2048}, True),
            dense_longer_kv=(
                {"seq_q": 1024, "seq_kv": 2048, "block_mask_is_dense": True},
                False,
            ),
            causal_gqa=(
                {"q_heads": 16, "kv_heads": 4, "seq_q": 512, "seq_kv": 4096},
                True,
            ),
            causal_ragged=({"seq_q": 1000}, True),
            dense_ragged=({"seq_q": 1000, "block_mask_is_dense": True}, False),
            causal_ragged_kv=({"seq_q": 1024, "seq_kv": 1050}, True),
            causal_gqa_ragged_q=(
                {
                    "batch": 2,
                    "q_heads": 16,
                    "kv_heads": 4,
                    "seq_q": 333,
                    "seq_kv": 1000,
                },
                True,
            ),
            dense_gqa_ragged_cross=(
                {
                    "q_heads": 16,
                    "kv_heads": 4,
                    "seq_q": 333,
                    "seq_kv": 4097,
                    "block_mask_is_dense": True,
                },
                False,
            ),
            causal_short_ragged=({"seq_q": 129, "seq_kv": 384}, True),
            causal_fp16=({"dtype": torch.float16}, True),
            dense_fp16=({"dtype": torch.float16, "block_mask_is_dense": True}, False),
            causal_fp16_cross=(
                {"dtype": torch.float16, "seq_q": 768, "seq_kv": 1280},
                True,
            ),
            causal_fp16_ragged_cross=(
                {"dtype": torch.float16, "seq_q": 777, "seq_kv": 1300},
                True,
            ),
        ),
    )
    def test_dualwave_sequence_lengths_and_dtypes(self, input_kwargs, expected):
        inputs = _dualwave_fake_inputs(**input_kwargs)
        kwargs, reason = _dualwave_choice_result(inputs, allow_causal_block_mask=True)
        self.assertIsNotNone(kwargs, reason)
        b, hq, sq, d = inputs["query"].get_size()
        _, hkv, sk, _ = inputs["key"].get_size()
        self.assertEqual(
            (kwargs["SEQ_LEN"], kwargs["KV_SEQ_LEN"], kwargs["CAUSAL"]),
            (sq, sk, expected),
        )
        fp16 = inputs["query"].get_dtype() == torch.float16
        self.assertEqual(kwargs["DTYPE_STR"], "f16" if fp16 else "bf16")
        self.assertEqual(kwargs["Q_STRIDE"], tuple(_bshd_stride(b, hq, sq, d)))
        self.assertEqual(kwargs["K_STRIDE"], tuple(_bshd_stride(b, hkv, sk, d)))

    @_subtests("seq_kv,kv_blocks", [(1000, 32), (5000, 64)])
    def test_symbolic_block_lists_bound_the_kv_blocks(self, seq_kv, kv_blocks):
        graph, (batch, seq_q, length_kv) = _symbolic_graph(3, 1000, seq_kv)
        inputs = _dualwave_fake_inputs(
            mask_fn=lambda b, h, q, kv: (q >= kv) & (q - kv < 256)
        )
        symbolic = _symbolic_forward_inputs(batch, seq_q, length_kv, q_heads=8)

        def bshd(heads, sequence):
            size = [batch, heads, sequence, 128]
            stride = [sequence * heads * 128, 128, heads * 128, 1]
            return _FakeNode(size, stride, numel=sympy.Mul(*size))

        inputs.update(
            {
                name: symbolic[name]
                for name in (
                    "kv_num_blocks",
                    "kv_indices",
                    "full_kv_num_blocks",
                    "full_kv_indices",
                )
            },
            query=bshd(8, seq_q),
            key=bshd(8, length_kv),
            value=bshd(8, length_kv),
        )
        config, reason = _dualwave_choice_result(inputs, graph=graph)
        self.assertIsNotNone(config, reason)
        self.assertTrue(config["BLOCK_LISTS"])
        self.assertTrue(config["RUNTIME_SIZES"])
        self.assertEqual(config["PLAN"]["kv_blocks"], kv_blocks)
        self.assertIsNone(config["MAX_PARTIAL_BLOCKS"])

    @_subtests(
        "overrides,expected",
        _cases(
            mask_without_block_lists=(
                {"causal": False, "mask_program": (("ge", 2, 3),)},
                "mask programs on block lists",
            ),
            causal=({"block_lists": True}, "causal or block lists"),
            too_many_kv_blocks=(
                {"causal": False, "block_lists": True, "kv_seq_len": 128 * 8193},
                "at most 8192 KV blocks",
            ),
            runtime_sk_without_bound=(
                {
                    "causal": False,
                    "block_lists": True,
                    "batch_size": None,
                    "seq_len": None,
                    "plan": {"mask_last_tile": False, "mask_second_last_tile": False},
                },
                "kv_blocks plan bound",
            ),
            block_mask_heads=(
                {"causal": False, "block_lists": True, "block_mask_heads": 3},
                "heads 1, Hkv or Hq",
            ),
        ),
    )
    def test_dualwave_builder_validates_mods(self, overrides, expected):
        self._assert_dualwave_builder_rejects(overrides, expected)

    @_subtests(
        "head_dims,wide,input_kwargs,overrides,expected_reason",
        [
            *(
                subtest(
                    (dims, False, {}, {}, "and ((64, 256)"),
                    name=f"qk{dims[0]}_v{dims[1]}_without_option",
                )
                for dims in _WIDE_HEAD_DIMS
            ),
            subtest(
                (
                    (192, 192),
                    True,
                    {},
                    dict(
                        zip(
                            ("mask_graph", "mask_mod_other_buffers"),
                            _window_capture_mask_graph(),
                        )
                    ),
                    "widens to int64",
                ),
                name="window_capture",
            ),
            subtest(
                (
                    (256, 256),
                    True,
                    {},
                    dict(
                        zip(
                            ("mask_graph", "mask_mod_other_buffers"),
                            _document_mask_graph(4),
                        )
                    ),
                    "at most 3 captures",
                ),
                name="four_documents_d256",
            ),
            subtest(
                (
                    (256, 256),
                    True,
                    {"block_mask_is_dense": True},
                    {
                        "subgraph": _SCALED_SCORE_MOD,
                        "score_mod_other_buffers": [
                            _FakeNode([1024], [1], torch.float16) for _ in range(4)
                        ],
                    },
                    "at most 3 captures",
                ),
                name="four_score_captures_d256",
            ),
            subtest(
                (
                    (256, 256),
                    True,
                    {"seq_q": 256, "seq_kv": 128 * 6909, "partial_width": 1},
                    {},
                    "at most 6908 128-key blocks",
                ),
                name="too_many_kv_blocks_d256",
            ),
        ],
    )
    def test_wide_inputs_are_not_offered(
        self, head_dims, wide, input_kwargs, overrides, expected_reason
    ):
        qk, v = head_dims
        inputs = _dualwave_fake_inputs(qk_dim=qk, v_dim=v, **input_kwargs)
        inputs.update(overrides)
        kwargs, reason = _dualwave_choice_result(
            inputs, allow_causal_block_mask=True, wide=wide
        )
        self.assertIsNone(kwargs)
        self.assertIn(expected_reason, reason)

    @_subtests(
        "overrides,expected",
        _cases(
            four_waves_d128=({"head_dim": 128, "num_waves": 4}, "4-wave dual-wave"),
            six_waves=({"head_dim": 128, "num_waves": 6}, "8 or 4 waves"),
            eight_waves_d192=({"head_dim": 192}, r"\(QK, V\) head dims"),
            kv_head_dims_8_waves=(
                {"head_dim": 128, "kv_head_dims": (96, 96)},
                "4 waves only",
            ),
            kv_head_dims_wider=(
                {"head_dim": 192, "num_waves": 4, "kv_head_dims": (224, 224)},
                "up to the head dims",
            ),
            too_many_kv_blocks_d256=(
                {
                    "head_dim": 256,
                    "num_waves": 4,
                    "causal": False,
                    "block_lists": True,
                    "kv_seq_len": 128 * 6909,
                },
                "at most 6908 KV blocks",
            ),
        ),
    )
    def test_dualwave_builder_validates_wide_geometry(self, overrides, expected):
        self._assert_dualwave_builder_rejects(overrides, expected)


class TestFlyDSLFlexAttention(_FlyDSLFlexAttentionRuntimeMixin, TestCase):
    def _dualwave_inputs(
        self,
        device,
        *,
        batch,
        q_heads,
        kv_heads,
        seq,
        causal,
        seq_kv=None,
        dtype=torch.bfloat16,
        head_dims=(128, 128),
        layout="bshd",
    ):
        """``layout``: ``bshd`` and ``bhsd`` memory, or ``bshd_gap`` and
        ``bhsd_gap`` with every row read from a twice-as-wide one."""
        seq_kv = seq if seq_kv is None else seq_kv
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            seq_kv=seq_kv,
            qk_dim=head_dims[0],
            v_dim=head_dims[1],
            seed=seq + q_heads + kv_heads,
            transposed=layout.startswith("bshd"),
            dtype=dtype,
        )
        if layout.endswith("_gap"):

            def with_gap(tensor):
                b, h, s, d = tensor.shape
                bshd = layout.startswith("bshd")
                wide = tensor.new_full(
                    (b, s, h, 2 * d) if bshd else (b, h, s, 2 * d), float("nan")
                )
                view = wide[..., :d].transpose(1, 2) if bshd else wide[..., :d]
                return view.copy_(tensor)

            query, key, value = (with_gap(t) for t in (query, key, value))
        block_mask = None
        if causal:
            block_mask = create_block_mask(
                lambda b, h, q, kv: q >= kv, None, None, seq, seq_kv, device=device
            )
        return query, key, value, block_mask

    def _run_flydsl_forward(
        self,
        query,
        key,
        value,
        block_mask,
        *,
        force_dualwave,
        requires_grad=False,
        scale=None,
        score_mod=None,
        dualwave_causal=True,
        head_slow_order=True,
    ):
        """Compile with FlyDSL autotuning; ``force_dualwave`` drops the upstream choice."""
        kernel_options = {"BACKEND": "FLYDSL"}
        if block_mask is not None and dualwave_causal:
            kernel_options["FLYDSL_DUALWAVE_CAUSAL"] = True
        if not head_slow_order:
            kernel_options["FLYDSL_HEAD_SLOW_ORDER"] = False
        if _dualwave_wide((query.size(-1), value.size(-1))):
            kernel_options["FLYDSL_DUALWAVE_WIDE"] = True

        def run(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                score_mod=score_mod,
                scale=scale,
                enable_gqa=q.size(1) != k.size(1),
                kernel_options=kernel_options,
                return_aux=AuxRequest(lse=True, max_scores=True),
            )

        selected = []
        select = flex_flydsl_attention.autotune_select_algorithm

        def spy_select(name, choices, *args, **kwargs):
            selected.append([choice.name for choice in choices])
            return select(name, choices, *args, **kwargs)

        patches = [
            torch._inductor.config.patch(
                {"flydsl_enable_autotuning": True, "fx_graph_cache": False}
            ),
            mock.patch.object(
                flex_flydsl_attention, "autotune_select_algorithm", spy_select
            ),
        ]
        if force_dualwave:
            patches.append(
                mock.patch.object(
                    flex_flydsl_attention,
                    "maybe_append_flydsl_flex_attention_choice",
                    return_value=(True, ""),
                )
            )
        inputs = (query, key, value)
        if requires_grad:
            inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
        torch._dynamo.reset()
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            (output, aux), code = run_and_get_code(
                torch.compile(run, fullgraph=True), *inputs
            )
        torch.cuda.synchronize()
        return inputs, output, aux, "\n".join(code), selected

    def _assert_matches_fp64(self, query, key, value, output, aux, causal):
        group = query.size(1) // key.size(1)
        key64 = key.double().repeat_interleave(group, dim=1)
        value64 = value.double().repeat_interleave(group, dim=1)
        scores = (query.double() @ key64.transpose(-2, -1)) * query.size(-1) ** -0.5
        if causal:
            keep = torch.ones(
                query.size(2), key.size(2), device=query.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~keep, -float("inf"))
        reference = scores.softmax(-1) @ value64
        self.assertEqual(output.double(), reference, atol=0.025, rtol=0.025)
        self.assertEqual(aux.lse.double(), scores.logsumexp(-1), atol=0.025, rtol=0.01)
        self.assertEqual(
            aux.max_scores.double(), scores.amax(-1), atol=0.025, rtol=0.01
        )

    def _assert_matches_triton(self, query, key, value, output, aux, **kwargs):
        """``output`` and its LSE match Triton's and eager fp64 flex attention
        of ``query``, ``key`` and ``value`` with ``kwargs``."""
        torch._dynamo.reset()
        reference, reference_aux = torch.compile(
            lambda q, k, v: flex_attention(
                q,
                k,
                v,
                kernel_options={"BACKEND": "TRITON"},
                return_aux=AuxRequest(lse=True),
                **kwargs,
            ),
            fullgraph=True,
        )(query, key, value)
        golden, golden_aux = flex_attention(
            *(t.double() for t in (query, key, value)),
            return_aux=AuxRequest(lse=True),
            **{**kwargs, "block_mask": _with_listed_blocks(kwargs.get("block_mask"))},
        )
        self._assert_close_to_fp64(output, reference, golden, atol=0.02, rtol=0.02)
        self._assert_close_to_fp64(
            aux.lse, reference_aux.lse, golden_aux.lse, atol=0.02, rtol=0.01
        )

    def _check_dualwave_against_triton(self, device, case, head_dims=(128, 128)):
        """Run a (B, Hq, Hkv, Sq, Sk, causal, dtype) ``case`` on the dual-wave
        forward; return its code."""
        self._require_runtime()
        batch, q_heads, kv_heads, seq_q, seq_kv, causal, dtype = case
        query, key, value, block_mask = self._dualwave_inputs(
            device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq=seq_q,
            seq_kv=seq_kv,
            causal=causal,
            dtype=dtype,
            head_dims=head_dims,
        )
        _, output, aux, code, selected = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=True
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        self.assertIn("dualwave", selected[0][0])
        self._assert_matches_fp64(query, key, value, output, aux, causal)
        self._assert_matches_triton(
            query,
            key,
            value,
            output,
            aux,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
        )
        return code

    @parametrize("case", list(_DYNAMIC_SHAPE_CASES))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    @torch.fx.experimental._config.patch(use_duck_shape=False)
    def test_gfx950_dynamic_shapes_match_triton(self, device, case):
        self._require_runtime()
        results = _dynamic_shape_runs(case, device, "FLYDSL", True)
        references = _dynamic_shape_runs(case, device, "TRITON", False)
        goldens = _dynamic_shape_runs(case, device, None, False)
        for result, reference, golden in zip(results, references, goldens, strict=True):
            for actual, expected, exact in zip(result, reference, golden):
                self._assert_close_to_fp64(
                    actual, expected, exact, atol=0.04, rtol=0.03
                )

    @parametrize(
        "batch,q_heads,kv_heads,seq,causal",
        [
            subtest((2, 16, 4, 640, False), name="dense_gqa_ragged"),
            subtest((1, 8, 1, 384, True), name="causal_mqa_min_length"),
            subtest((1, 16, 16, 4096, True), name="causal_mha_long"),
        ],
    )
    def test_gfx950_dualwave_forward_matches_fp64(
        self, device, batch, q_heads, kv_heads, seq, causal
    ):
        self._require_runtime()
        query, key, value, block_mask = self._dualwave_inputs(
            device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq=seq,
            causal=causal,
        )
        _, output, aux, code, selected = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=True
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        self.assertNotIn("build_flex_attn_fwd_module(", code)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]), 1)
        self.assertIn("dualwave", selected[0][0])
        self._assert_matches_fp64(query, key, value, output, aux, causal)

    @parametrize(
        *_paired(
            (
                "batch,q_heads,kv_heads,seq,head_dims",
                [
                    subtest((1, 8, 8, 16384, (128, 128)), name="mha"),
                    subtest((2, 16, 4, 16500, (96, 96)), name="gqa_ragged_d96"),
                    subtest((1, 8, 2, 16384, (64, 128)), name="gqa_qk64_v128"),
                ],
            ),
            ("kernel", ["forward", "dualwave"]),
        )
    )
    def test_gfx950_head_slow_order_is_bitwise(
        self, device, kernel, batch, q_heads, kv_heads, seq, head_dims
    ):
        self._require_runtime()
        query, key, value, _ = self._dualwave_inputs(
            device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq=seq,
            causal=False,
            head_dims=head_dims,
        )
        runs = []
        for head_slow_order in (True, False):
            with contextlib.ExitStack() as stack:
                if kernel == "forward":
                    stack.enter_context(
                        mock.patch.object(
                            flex_flydsl_attention,
                            "maybe_append_flydsl_dualwave_forward_choice",
                            return_value=(False, ""),
                        )
                    )
                _, output, aux, code, _ = self._run_flydsl_forward(
                    query,
                    key,
                    value,
                    None,
                    force_dualwave=kernel == "dualwave",
                    head_slow_order=head_slow_order,
                )
            self.assertEqual("head_slow_order=HEAD_SLOW_ORDER" in code, head_slow_order)
            runs.append((output, aux.lse, aux.max_scores))
        for ordered, unordered in zip(*runs, strict=True):
            self.assertTrue(torch.equal(ordered, unordered))
        output, lse, max_scores = runs[0]
        group = q_heads // kv_heads
        # One head at a time keeps the fp64 scores to a few GiB.
        for b, h in itertools.product(range(batch), range(q_heads)):
            rows = (slice(b, b + 1), slice(h, h + 1))
            kv_rows = (slice(b, b + 1), slice(h // group, h // group + 1))
            self._assert_matches_fp64(
                query[rows],
                key[kv_rows],
                value[kv_rows],
                output[rows],
                SimpleNamespace(lse=lse[rows], max_scores=max_scores[rows]),
                False,
            )

    @parametrize(
        "case",
        [
            subtest(
                (2, 16, 4, 512, 2048, True, torch.float16), name="causal_gqa_cross_fp16"
            ),
            subtest(
                (1, 8, 8, 1000, 1000, False, torch.float16), name="dense_ragged_fp16"
            ),
            subtest(
                (1, 8, 8, 2048, 1050, True, torch.bfloat16),
                name="causal_longer_q_ragged_kv",
            ),
            subtest((1, 8, 1, 129, 384, True, torch.bfloat16), name="causal_mqa_short"),
        ],
    )
    def test_gfx950_dualwave_against_triton(self, device, case):
        self._check_dualwave_against_triton(device, case)

    @parametrize(
        *_paired(
            (
                "head_dims",
                [
                    subtest(
                        dims,
                        name=f"qk{dims[0]}_v{dims[1]}"
                        if dims[0] != dims[1]
                        else f"d{dims[0]}",
                    )
                    for dims in (
                        (64, 128),
                        (96, 96),
                        *_WIDE_HEAD_DIMS,
                        *_NATIVE_WIDE_HEAD_DIMS,
                    )
                ],
            ),
            (
                "case",
                [
                    subtest((1, 8, 8, 1024, 1024, False, torch.bfloat16), name="dense"),
                    subtest(
                        (1, 8, 8, 1024, 1024, True, torch.float16), name="causal_fp16"
                    ),
                    subtest(
                        (2, 16, 4, 1000, 1000, True, torch.bfloat16),
                        name="causal_gqa_ragged",
                    ),
                    subtest(
                        (2, 16, 4, 333, 4097, False, torch.float16),
                        name="dense_gqa_cross_ragged_fp16",
                    ),
                    subtest(
                        (1, 8, 2, 777, 1050, True, torch.bfloat16),
                        name="causal_gqa_cross_ragged",
                    ),
                ],
            ),
        )
    )
    def test_gfx950_dualwave_head_dims_against_triton(self, device, head_dims, case):
        code = self._check_dualwave_against_triton(device, case, head_dims)
        if head_dims in _NATIVE_WIDE_HEAD_DIMS:
            self.assertIn(f"KV_HEAD_DIMS: fx.Constexpr = {head_dims}", code)
        else:
            self.assertIn(f"V_HEAD_DIM: fx.Constexpr = {head_dims[1]}", code)
        if _dualwave_wide(head_dims):
            self.assertIn("NUM_WAVES: fx.Constexpr = 4", code)

    @parametrize("case", list(_DUALWAVE_MOD_CASES))
    def test_gfx950_dualwave_mods_against_triton(self, device, case):
        self._require_runtime()
        query, key, value, block_mask, reference_mask, score_mod = _dualwave_mod_inputs(
            case, device
        )
        _, output, aux, code, selected = self._run_flydsl_forward(
            query,
            key,
            value,
            block_mask,
            force_dualwave=True,
            score_mod=score_mod,
            dualwave_causal=False,
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        self.assertNotIn("build_flex_attn_fwd_module(", code)
        self.assertEqual(selected, [[selected[0][0]]])
        self.assertIn("dualwave", selected[0][0])
        self._assert_matches_triton(
            query,
            key,
            value,
            output,
            aux,
            block_mask=reference_mask,
            score_mod=score_mod,
            enable_gqa=True,
        )

    @parametrize("case", ["dense", "causal", "key_only"])
    def test_gfx950_dualwave_broadcast_kv_batch(self, device, case):
        self._require_runtime()
        batch, seq = 3, 1024
        query, key, value, _ = self._dualwave_inputs(
            device, batch=batch, q_heads=16, kv_heads=4, seq=seq, causal=False
        )
        causal = case == "causal"
        block_mask = None
        if case == "key_only":
            # V keeps a batch per query batch, so only K is read from batch 0.
            key = key[:1].expand(batch, -1, -1, -1)
        elif causal:
            key, value = key[:1], value[:1]
            block_mask = create_block_mask(
                lambda b, h, q, kv: q >= kv, batch, None, seq, seq, device=device
            )
        else:
            # Without a BlockMask the K/V batch must match; expand it in place.
            key, value = (t[:1].expand(batch, -1, -1, -1) for t in (key, value))
        _, output, aux, code, _ = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=True
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        self.assertIn("K_STRIDE: fx.Constexpr = (0, ", code)
        self._assert_matches_fp64(query, key, value, output, aux, causal)

    @parametrize(
        "q_heads,kv_heads,causal,head_dims",
        [
            subtest((16, 4, True, (128, 128)), name="causal_gqa_flydsl_bwd"),
            subtest((8, 8, False, (96, 96)), name="dense_mha_flydsl_bwd_d96"),
        ],
    )
    def test_gfx950_dualwave_lse_feeds_backward(
        self, device, q_heads, kv_heads, causal, head_dims
    ):
        self._require_runtime()
        seq = 640
        query, key, value, block_mask = self._dualwave_inputs(
            device,
            batch=1,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq=seq,
            causal=causal,
            head_dims=head_dims,
        )
        torch.manual_seed(24)
        grad_output = torch.randn(
            *query.shape[:-1], value.size(-1), device=device, dtype=query.dtype
        )
        inputs, output, _, code, _ = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=True, requires_grad=True
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        _, backward_code = run_and_get_code(lambda: output.backward(grad_output))
        backward_code = "\n".join(backward_code)
        self.assertIn("flydsl_fused_flex_attention_backward_", backward_code)
        grads = [t.grad for t in inputs]

        reference_inputs = tuple(
            t.detach().clone().requires_grad_() for t in (query, key, value)
        )
        torch._dynamo.reset()
        reference = torch.compile(
            lambda q, k, v: flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=q_heads != kv_heads,
                kernel_options={"BACKEND": "TRITON"},
            ),
            fullgraph=True,
        )(*reference_inputs)
        reference.backward(grad_output)
        golden_inputs = tuple(
            t.detach().double().requires_grad_() for t in (query, key, value)
        )
        golden = flex_attention(
            *golden_inputs, block_mask=block_mask, enable_gqa=q_heads != kv_heads
        )
        golden.backward(grad_output.double())
        self._assert_close_to_fp64(output, reference, golden, atol=0.03, rtol=0.02)
        # A natural-log LSE from the forward would skew every gradient by ~ln 2.
        for grad, reference_input, golden_input in zip(
            grads, reference_inputs, golden_inputs
        ):
            self._assert_close_to_fp64(
                grad, reference_input.grad, golden_input.grad, atol=0.04, rtol=0.03
            )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_dualwave_autotunes_against_upstream(self, device):
        self._require_runtime()
        query, key, value, block_mask = self._dualwave_inputs(
            device, batch=1, q_heads=16, kv_heads=16, seq=2048, causal=True
        )
        _, output, aux, code, selected = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=False
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]), 2)
        self.assertTrue(any("dualwave" in name for name in selected[0]))
        self.assertTrue(
            "build_flex_attn_fwd_dualwave_module" in code
            or "build_flex_attn_fwd_module(" in code
        )
        self._assert_matches_fp64(query, key, value, output, aux, True)

    @parametrize(
        *_paired(
            (
                "seq,kv_heads,causal,head_dims",
                [
                    subtest((1024, 8, False, (128, 128)), name="dense"),
                    subtest((1000, 2, False, (128, 128)), name="ragged_gqa"),
                    subtest((1024, 8, True, (128, 128)), name="causal"),
                    subtest((777, 4, False, (256, 256)), name="ragged_gqa_d256"),
                ],
            ),
            ("layout", ("bhsd", "bshd_gap", "bhsd_gap")),
        )
    )
    def test_gfx950_dualwave_strided_layouts(
        self, device, layout, seq, kv_heads, causal, head_dims
    ):
        # Each (b, h) slab is bounded on its own, so ragged rows read zeros and
        # drop their stores whichever batch or head follows in memory.
        self._require_runtime()
        query, key, value, block_mask = self._dualwave_inputs(
            device,
            batch=2,
            q_heads=8,
            kv_heads=kv_heads,
            seq=seq,
            causal=causal,
            head_dims=head_dims,
            layout=layout,
        )
        _, output, aux, code, _ = self._run_flydsl_forward(
            query, key, value, block_mask, force_dualwave=True
        )
        self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        self.assertIn(f"CAUSAL: fx.Constexpr = {causal}", code)
        self._assert_matches_fp64(query, key, value, output, aux, causal)

    @parametrize(
        "head_dims,dualwave",
        [((128, 128), True), ((256, 256), True), ((192, 128), False)],
    )
    @torch._inductor.config.patch(
        {"fx_graph_cache": False, "flydsl_enable_autotuning": False}
    )
    def test_gfx950_dense_bshd_prefill_default(self, device, head_dims, dualwave):
        # Without autotuning, dense BSHD prefill with more Q blocks than CUs
        # runs only the dual-wave forward at the head dims it is faster at.
        self._require_runtime()
        num_cus = torch.cuda.get_device_properties(device).multi_processor_count
        heads = num_cus // 8 + 8
        query, key, value, _ = self._dualwave_inputs(
            device,
            batch=1,
            q_heads=heads,
            kv_heads=heads,
            seq=1024,
            causal=False,
            head_dims=head_dims,
        )
        selected = []
        select = flex_flydsl_attention.autotune_select_algorithm

        def spy_select(name, choices, *args, **kwargs):
            selected.append([choice.name for choice in choices])
            return select(name, choices, *args, **kwargs)

        def run(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                kernel_options={"BACKEND": "FLYDSL"},
                return_aux=AuxRequest(lse=True, max_scores=True),
            )

        torch._dynamo.reset()
        with mock.patch.object(
            flex_flydsl_attention, "autotune_select_algorithm", spy_select
        ):
            output, aux = torch.compile(run, fullgraph=True)(query, key, value)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]), 1)
        self.assertEqual("dualwave" in selected[0][0], dualwave)
        self._assert_matches_fp64(query, key, value, output, aux, False)

    # Past 2**26 32-bit elements, a byte bound counted in i32 bits wraps.
    @parametrize("elements", [(1 << 26) + 64, (1 << 27) + 64])
    def test_runtime_extent_buffer_views_bound_large_tensors(self, device, elements):
        # Generic buffer copies through a runtime-extent view run on any ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl._mlir.dialects import fly
        from flydsl.expr.typing import Vector

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            layout_extents,
            make_global_view,
        )
        from torch._inductor.runtime.flydsl_cache import (
            flydsl_runtime_layout_arg,
            temporary_env,
        )

        threads = 64

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def copy_kernel(source: fx.Tensor, target: fx.Tensor, start: fx.Int32):
            tid = fx.Int32(fx.thread_idx.x)
            source_view = make_global_view(
                source, None, layout_extents(source)[0][0], 1
            )
            target_view = make_global_view(target, None, threads, 1)
            copy_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            value = fly.copy_atom_call_ssa(
                [Vector.make_type(1, fx.Int32)],
                copy_32,
                fx.slice(
                    fx.logical_divide(source_view, fx.make_layout(1, 1)),
                    (None, start + tid),
                ),
            )
            register = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.memref_store_vec(value, register)
            fx.copy(
                copy_32,
                register,
                fx.slice(
                    fx.logical_divide(target_view, fx.make_layout(1, 1)), (None, tid)
                ),
            )

        @flyc.jit
        def launch(
            source: fx.Tensor,
            target: fx.Tensor,
            start: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            copy_kernel(source, target, start).launch(
                grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
            )

        # Nonzero memory follows the source.
        backing = torch.arange(
            1, elements + threads + 1, device=device, dtype=torch.int32
        )
        for start in (elements - threads // 2, elements - threads):
            target = torch.full((threads,), -1, device=device, dtype=torch.int32)
            # FlyDSL's disk cache key leaves out make_global_view's source.
            with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
                launch(
                    flydsl_runtime_layout_arg(backing[:elements]),
                    flyc.from_torch_tensor(target),
                    start,
                    torch.cuda.current_stream(),
                )
            torch.cuda.synchronize()
            expected = backing[start : start + threads].clone()
            expected[expected > elements] = 0
            self.assertEqual(target, expected, atol=0, rtol=0)

    # A tile past the end of the source, as the last K/V tile of the first
    # K/V batch past Sk, must read zeros too.
    @parametrize(*_paired(("offset", [0, 640, 2048 + 64]), ("dim", [16, 24])))
    def test_dualwave_flat_buffer_views_address_memory(self, device, offset, dim):
        # Generic buffer copies through the dual-wave's addressing run on any ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl._mlir.dialects import fly
        from flydsl.expr.typing import Vector

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_fwd_dualwave_gfx950 import (
            flat_buffer_view,
            tile_buffer_view,
        )
        from torch._inductor.runtime.flydsl_cache import temporary_env

        batch, seq, heads = 2, 16, 4
        elements = batch * seq * heads * dim
        threads = elements // 8
        stats_size = batch * heads * seq

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def copy_kernel(
            source: fx.Tensor, target: fx.Tensor, stats: fx.Tensor, offset: fx.Int32
        ):
            tid = fx.Index(fx.thread_idx.x)
            # Like a ragged K/V tail: the bound ends where the source does.
            source_view = tile_buffer_view(source, fx.Int32(elements), offset)
            target_view = flat_buffer_view(target, fx.Int32(elements * 2))
            stats_view = flat_buffer_view(stats)
            copy_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
            pack = fly.copy_atom_call_ssa(
                [Vector.make_type(4, fx.Int32)],
                copy_128,
                fx.slice(source_view, (None, fx.Int32(tid * 8))),
            )
            register = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.memref_store_vec(pack, register)
            fx.copy(
                copy_128, register, fx.slice(target_view, (None, fx.Int32(tid * 8)))
            )
            if tid < stats_size:
                bits = Vector.from_elements(
                    [fx.Float32(fx.Int32(tid))], fx.Float32
                ).bitcast(fx.Int32)
                stats_register = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                fx.memref_store_vec(bits, stats_register)
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32),
                    stats_register,
                    fx.slice(stats_view, (None, fx.Int32(tid))),
                )

        @flyc.jit
        def launch(
            source: fx.Tensor,
            target: fx.Tensor,
            stats: fx.Tensor,
            offset: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            copy_kernel(source, target, stats, offset).launch(
                grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
            )

        # BHSD views of [B, S, H, D] memory, as the dual-wave kernel receives
        # them; nonzero memory follows the source.
        backing = torch.randn(2, batch, seq, heads, dim, device=device).bfloat16()
        source = backing[0]
        target = torch.zeros_like(source)
        stats = torch.full((batch, heads, seq), -1.0, device=device)
        # FlyDSL's disk cache key leaves out the dual-wave module's source.
        with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
            launch(
                *(
                    flyc.from_torch_tensor(tensor)
                    for tensor in (
                        source.transpose(1, 2),
                        target.transpose(1, 2),
                        stats,
                    )
                ),
                offset,
                torch.cuda.current_stream(),
            )
        torch.cuda.synchronize()
        expected = torch.zeros_like(source).flatten()
        expected[: max(elements - offset, 0)] = source.flatten()[offset:]
        self.assertEqual(target.flatten(), expected, atol=0, rtol=0)
        self.assertEqual(
            stats.flatten(), torch.arange(stats_size, device=device).float()
        )

    @parametrize(
        "q_blocks,kv_blocks,width,rows",
        [
            # Rows with no block, one block at 0, one past it, and a full row.
            subtest((4, 5, 5, ((), (0,), (3,), (4, 3, 2, 1, 0))), name="edges"),
            subtest((7, 200, 200, None), name="random_multi_chunk"),
        ],
    )
    def test_dualwave_block_union_matches_lists(
        self, device, q_blocks, kv_blocks, width, rows
    ):
        # The dual-wave's BlockMask union runs on any ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            flex_attn_fwd_dualwave_gfx950 as dualwave,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            make_global_view,
        )
        from torch._inductor.runtime.flydsl_cache import temporary_env

        generator = torch.Generator().manual_seed(q_blocks * kv_blocks)
        shape = (1, 1, q_blocks)
        lists = []
        for full in (False, True):
            # Unsorted, duplicated and out-of-range blocks, garbage past each count.
            counts = torch.randint(0, width + 1, shape, generator=generator)
            indices = torch.randint(
                -3, kv_blocks + 3, (*shape, width), generator=generator
            )
            if rows is not None:
                counts.zero_()
                for row, blocks in enumerate(rows):
                    if not full:
                        counts[0, 0, row] = len(blocks)
                        indices[0, 0, row, : len(blocks)] = torch.tensor(blocks)
            lists += [counts.int(), indices.int()]
        pairs = -(-q_blocks // 2)
        slots = kv_blocks + 4

        @fx.struct
        class Storage:
            states: fx.Array[fx.Int32, 2 * kv_blocks + 1, 16]
            union: fx.Array[fx.Int32, slots, 16]

        @flyc.kernel(known_block_size=[512, 1, 1])
        def union_kernel(
            counts: fx.Tensor,
            indices: fx.Tensor,
            full_counts: fx.Tensor,
            full_indices: fx.Tensor,
            union: fx.Tensor,
        ):
            lds = fx.SharedAllocator().allocate(Storage).peek()
            state_ptr, union_ptr = (
                dualwave._lds_ptr_from_index(fx.Index(fx.ptrtoint(array.ptr)))
                for array in (lds.states, lds.union)
            )
            pair = fx.Int32(fx.block_idx.x)
            views = tuple(
                (
                    make_global_view(c, None, q_blocks, 1),
                    make_global_view(i, None, q_blocks * width, 1),
                    fx.Int32(width),
                )
                for c, i in ((counts, indices), (full_counts, full_indices))
            )
            dualwave.build_block_union(
                views,
                pair * fx.Int32(2),
                pair * fx.Int32(2) + fx.Int32(1) < fx.Int32(q_blocks),
                fx.Int32(kv_blocks),
                state_ptr,
                union_ptr,
                kv_blocks,
                512,
            )
            thread = fx.Int32(fx.thread_idx.x)
            if thread < fx.Int32(slots):
                fx.ptr_store(
                    dualwave._lds_load_i32(union_ptr, thread),
                    fx.add_offset(fx.get_iter(union), pair * fx.Int32(slots) + thread),
                )

        @flyc.jit
        def launch(
            counts: fx.Tensor,
            indices: fx.Tensor,
            full_counts: fx.Tensor,
            full_indices: fx.Tensor,
            union: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            union_kernel(counts, indices, full_counts, full_indices, union).launch(
                grid=(pairs, 1, 1), block=(512, 1, 1), stream=stream
            )

        union = torch.full((pairs, slots), -1, dtype=torch.int32, device=device)
        # FlyDSL's disk cache key leaves out the dual-wave module's source.
        with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
            launch(
                *(flyc.from_torch_tensor(t.to(device)) for t in (*lists, union)),
                torch.cuda.current_stream(),
            )
        torch.cuda.synchronize()
        states = _dense_block_states(*lists, kv_blocks)[0, 0]
        for pair in range(pairs):
            rows_states = states[2 * pair : 2 * pair + 2]
            if rows_states.size(0) == 1:
                rows_states = torch.cat((rows_states, torch.zeros_like(rows_states)))
            listed = (rows_states != 0).any(0).nonzero().flatten().tolist()
            expected = [
                block
                | int(rows_states[0, block]) << 24
                | int(rows_states[1, block]) << 26
                for block in listed
            ]
            if not expected:
                expected = [0, 1]
            elif len(expected) == 1:
                first = listed[0]
                expected = [expected[0], 1] if first == 0 else [first - 1, expected[0]]
            count = int(union[pair, kv_blocks + 2])
            self.assertEqual(union[pair, :count].tolist(), expected)

    def test_tile_score_mod_readers_match_captures(self, device):
        # Capture-table reads are generic buffer loads and run on any ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr.typing import Vector

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            CaptureTable,
            make_global_view,
            make_score_mod_readers,
            make_tile_score_mod_readers,
        )
        from torch._inductor.runtime.flydsl_cache import temporary_env

        # The last two 32-key tiles read past Sk, partly and wholly.
        seq_q, seq_kv, tiles = 3, 72, 4
        captures = [
            torch.randn(seq_kv, device=device),
            torch.randn(seq_kv, device=device).bfloat16(),
            torch.randn(seq_q, device=device),
            torch.randint(-9, 9, (seq_q, seq_kv), device=device, dtype=torch.int32),
        ]
        aux = (
            (0, (0, 0, 0, 1), seq_kv),
            (1, (0, 0, 0, 1), seq_kv),
            (2, (0, 0, 1, 0), seq_q),
            (3, (0, 0, seq_kv, 1), seq_q * seq_kv),
        )
        sizes = tuple(capture.numel() for capture in captures)
        dtypes = tuple(
            str(capture.dtype).removeprefix("torch.") for capture in captures
        )
        values = tiles * 64 * 16 * len(aux)

        @flyc.kernel(known_block_size=[64, 1, 1])
        def read_kernel(table: fx.Tensor, tiled: fx.Tensor, scalar: fx.Tensor):
            lane = fx.Int32(fx.thread_idx.x)
            tile = fx.Int32(fx.block_idx.x)
            capture_table = CaptureTable(table, sizes, dtypes)
            # Keys as in a 32x32 MFMA accumulator: runs of four per lane half.
            keys = [
                tile * fx.Int32(32)
                + fx.Int32(element % 4 + 8 * (element // 4))
                + lane // fx.Int32(32) * fx.Int32(4)
                for element in range(16)
            ]
            readers = (
                make_tile_score_mod_readers(aux, capture_table, keys),
                make_score_mod_readers(None, aux, capture_table),
            )
            query = lane % fx.Int32(seq_q)
            store = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            for element in fx.range_constexpr(16):
                for slot in fx.range_constexpr(len(aux)):
                    index = (
                        (tile * fx.Int32(64) + lane) * fx.Int32(16) + fx.Int32(element)
                    ) * fx.Int32(len(aux)) + fx.Int32(slot)
                    for output, read in zip((tiled, scalar), readers):
                        register = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                        fx.memref_store_vec(
                            Vector.from_elements(
                                [
                                    read[slot](
                                        fx.Int32(0), fx.Int32(0), query, keys[element]
                                    )
                                ],
                                fx.Float32,
                            ),
                            register,
                        )
                        fx.copy(
                            store,
                            register,
                            fx.slice(
                                fx.logical_divide(
                                    make_global_view(output, None, values, 1),
                                    fx.make_layout(1, 1),
                                ),
                                (None, index),
                            ),
                        )

        @flyc.jit
        def launch(
            table: fx.Tensor,
            tiled: fx.Tensor,
            scalar: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            read_kernel(table, tiled, scalar).launch(
                grid=(tiles, 1, 1), block=(64, 1, 1), stream=stream
            )

        table = torch.tensor(
            [capture.data_ptr() for capture in captures],
            dtype=torch.int64,
            device=device,
        )
        tiled, scalar = (
            torch.full((values,), float("nan"), device=device) for _ in range(2)
        )
        # FlyDSL's disk cache key leaves out the capture table's source.
        with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
            launch(
                *(flyc.from_torch_tensor(t) for t in (table, tiled, scalar)),
                torch.cuda.current_stream(),
            )
        torch.cuda.synchronize()
        lane = torch.arange(64, device=device)
        element = torch.arange(16, device=device)
        key = (
            torch.arange(tiles, device=device)[:, None, None] * 32
            + (element % 4 + 8 * (element // 4))[None, None, :]
            + (lane // 32 * 4)[None, :, None]
        )
        query = (lane % seq_q)[None, :, None].expand_as(key)

        def gather(flat, offset):
            padded = torch.cat((flat.float(), flat.new_zeros(256).float()))
            return padded[offset]

        expected = torch.stack(
            [
                gather(captures[0], key),
                gather(captures[1], key),
                captures[2][query],
                gather(captures[3].flatten(), query * seq_kv + key),
            ],
            dim=-1,
        ).flatten()
        self.assertEqual(scalar, expected, atol=0, rtol=0)
        self.assertEqual(tiled, expected, atol=0, rtol=0)

    def _combine_kv_splits(self, partials, split_lse, split_max, output, **kwargs):
        """``output`` and the merged log2 LSE and max the combine kernel writes."""
        import flydsl.compiler as flyc

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_split_kv_combine import (
            build_flex_attn_split_kv_combine_module,
        )
        from torch._inductor.runtime.flydsl_cache import temporary_env

        splits, batch, heads, seq_q, head_dim = partials.shape
        stats = [
            torch.full((batch, heads + 1, seq_q), 7.0, device=partials.device)[:, 1:]
            for _ in range(2)
        ]
        launch = build_flex_attn_split_kv_combine_module(
            kv_splits=splits,
            batch_size=batch,
            num_heads=heads,
            seq_q=seq_q,
            head_dim=head_dim,
            partial_stride=partials.stride(),
            stats_stride=split_lse.stride(),
            o_stride=output.stride(),
            lse_stride=stats[0].stride(),
            dtype="bf16" if output.dtype == torch.bfloat16 else "fp16",
            **kwargs,
        )
        with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
            launch(
                *(
                    flyc.from_torch_tensor(t)
                    for t in (partials, split_lse, split_max, *stats, output)
                ),
                torch.cuda.current_stream(),
            )
        torch.cuda.synchronize()
        return output, *stats

    @parametrize(
        "splits,batch,heads,seq_q,head_dim,layout",
        [
            subtest((4, 2, 3, 1, 128, "contiguous"), name="contiguous"),
            subtest((7, 2, 8, 3, 128, "bshd"), name="bshd_output"),
            subtest((5, 3, 4, 2, 96, "odd_strides"), name="odd_strides"),
            subtest((33, 2, 2, 4, 256, "fp16"), name="fp16_many_splits"),
        ],
    )
    def test_split_kv_combine_matches_merge(
        self, device, splits, batch, heads, seq_q, head_dim, layout
    ):
        # The combine kernel uses no gfx950-only instruction and runs on any
        # ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        generator = torch.Generator(device=device).manual_seed(splits)
        rows = (batch, heads, seq_q)
        if layout == "odd_strides":
            # Odd row strides and a head stride above the batch's.
            storage = torch.randn(
                splits, heads + 1, batch, seq_q, head_dim + 3, device=device
            )
            partials = storage[:, 1:, :, :, 1 : head_dim + 1].transpose(1, 2)
            split_lse, split_max = (
                torch.randn(
                    splits, heads, batch, seq_q + 1, device=device, generator=generator
                )[..., 1:].transpose(1, 2)
                for _ in range(2)
            )
        else:
            partials = torch.randn(
                splits, *rows, head_dim, device=device, generator=generator
            )
            split_lse, split_max = (
                torch.randn(splits, *rows, device=device, generator=generator)
                for _ in range(2)
            )
        split_lse.mul_(8)
        split_max.copy_(split_lse - split_max.abs())
        # Splits without keys leave -inf statistics; their outputs are unread.
        empty = torch.rand(split_lse.shape, device=device, generator=generator) < 0.3
        empty[:, 0, 0, 0] = True
        split_lse.masked_fill_(empty, float("-inf"))
        split_max.masked_fill_(empty, float("-inf"))
        partials.masked_fill_(empty[..., None], float("nan"))
        dtype = torch.float16 if layout == "fp16" else torch.bfloat16
        if layout == "bshd":
            output = torch.empty(batch, seq_q, heads, head_dim, device=device)
            output = output.to(dtype).transpose(1, 2)
        elif layout == "odd_strides":
            output = torch.empty(batch, heads, seq_q, head_dim + 1, device=device)
            output = output.to(dtype)[..., :head_dim]
        else:
            output = torch.empty(*rows, head_dim, device=device, dtype=dtype)
        output, lse, max_scores = self._combine_kv_splits(
            partials, split_lse, split_max, output
        )

        lse64 = split_lse.double()
        weights = torch.exp2(lse64 - lse64.amax(0).nan_to_num(neginf=0.0))
        total = weights.sum(0)
        expected = (
            torch.where(weights[..., None] > 0, partials.double(), 0.0)
            * weights[..., None]
        ).sum(0) / total.clamp_min(1e-300)[..., None]
        expected_lse = lse64.amax(0).nan_to_num(neginf=0.0) + torch.log2(total)
        self.assertFalse(output.isnan().any())
        self.assertEqual(output, expected.to(dtype), atol=0.02, rtol=0.01)
        self.assertEqual(lse, expected_lse.float(), atol=1e-5, rtol=1e-6)
        self.assertEqual(max_scores, split_max.amax(0), atol=0, rtol=0)
        self.assertTrue(torch.isneginf(lse[0, 0, 0]))
        self.assertEqual(output[0, 0, 0], torch.zeros_like(output[0, 0, 0]))

    @parametrize("write_max_scores", [False, True])
    def test_split_kv_combine_matches_gqa_attention(self, device, write_max_scores):
        # Partials of split key ranges of a GQA attention merge to the full
        # attention; runs on any ROCm GPU.
        if not flydsl_utils.runtime_available():
            self.skipTest("requires ROCm and a built FlyDSL runtime")
        batch, q_heads, kv_heads, seq_q, seq_kv, splits = 3, 8, 2, 2, 1000, 6
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=9,
        )
        positions = torch.arange(seq_kv, device=device)
        # Rows of different lengths per head, so the last split has no keys
        # for most heads, and one head of one batch with no keys at all.
        heads = torch.arange(q_heads, device=device)[:, None, None]
        keep = positions <= (
            700 + 37 * heads + torch.arange(seq_q, device=device)[:, None]
        )
        keep = keep.expand(batch, -1, -1, -1).clone()
        keep[2, 5] = False
        scores, v = _masked_scores(query, key, value, keep, 0.1)
        bounds = [0, 128, 256, 400, 640, 900, seq_kv]
        partials, split_lse, split_max = [], [], []
        for start, end in itertools.pairwise(bounds):
            part = torch.full_like(scores, float("-inf"))
            part[..., start:end] = scores[..., start:end]
            partials.append(_normalized_attention(part, v))
            split_lse.append(part.logsumexp(-1) / math.log(2))
            split_max.append(part.amax(-1) / math.log(2))
        self.assertEqual(len(partials), splits)
        output, lse, max_scores = self._combine_kv_splits(
            *(torch.stack(t) for t in (partials, split_lse, split_max)),
            torch.empty_like(query),
            write_max_scores=write_max_scores,
        )
        scores64 = scores.double()
        expected = _normalized_attention(scores64, v.double())
        self.assertEqual(output, expected.to(output.dtype), atol=0.01, rtol=0.01)
        self.assertEqual(
            lse, (scores64.logsumexp(-1) / math.log(2)).float(), atol=1e-5, rtol=1e-6
        )
        if write_max_scores:
            self.assertEqual(max_scores, (scores64.amax(-1) / math.log(2)).float())
        else:
            self.assertEqual(max_scores, torch.full_like(max_scores, 7.0))
        self.assertTrue(torch.isneginf(lse[2, 5]).all())
        self.assertEqual(output[2, 5], torch.zeros_like(output[2, 5]))

    @parametrize(
        "qk_dim,mask_kind,dq_accum_fp32",
        [
            subtest((128, "dense", True), name="d128_dense_fp32"),
            subtest((192, "causal", False), name="d192_causal_bf16"),
            subtest((192, "document", True), name="d192_document_fp32"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward(self, device, qk_dim, mask_kind, dq_accum_fp32):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, qk_dim=qk_dim, seed=23)
        block_mask = None
        if mask_kind != "dense":
            if mask_kind == "causal":

                def mask_mod(b, h, q, kv):
                    return q >= kv

            else:
                document_ids = (
                    torch.arange(256, device=device, dtype=torch.int32) // 128
                )
                document_starts = torch.tensor(
                    [0, 128], device=device, dtype=torch.int32
                )

                def mask_mod(b, h, q, kv):
                    return (q >= kv) & (kv >= document_starts[document_ids[q]])

            block_mask = create_block_mask(
                mask_mod, 1, 1, 256, 256, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=dq_accum_fp32,
        )

    @parametrize("mask_kind", ["dense", "causal", "empty_heads"])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_direct_dq(self, device, mask_kind):
        # One Q block and one KV owner: the compute kernel stores dQ itself,
        # and zeroes it for heads whose block list is empty.
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, q_heads=4, seq_q=100, seq_kv=128, qk_dim=64, v_dim=64
        )
        block_mask = None
        if mask_kind == "causal":
            block_mask = create_block_mask(
                lambda b, h, q, kv: q >= kv, 1, 1, 100, 128, device=device
            )
        elif mask_kind == "empty_heads":
            block_mask = create_block_mask(
                lambda b, h, q, kv: (h < 2) & (q >= kv), 1, 4, 100, 128, device=device
            )
        # Freed small-pool NaNs make an unwritten dQ row fail the comparison.
        poison = [torch.full((1 << 14,), math.nan, device=device) for _ in range(64)]
        del poison
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @parametrize(
        "shape,dims,causal,dtype",
        [
            subtest(
                ((2, 4, 4, 512, 512), (128, 128), False, torch.bfloat16), name="d128"
            ),
            subtest(
                ((2, 4, 4, 1000, 1000), (64, 64), True, torch.bfloat16),
                name="d64_causal_ragged",
            ),
            subtest(
                ((1, 8, 2, 700, 1300), (96, 96), True, torch.bfloat16),
                name="d96_gqa_causal_long_kv",
            ),
            subtest(
                ((1, 4, 1, 1300, 700), (128, 128), True, torch.float16),
                name="mqa_causal_long_q_fp16",
            ),
            subtest(
                ((1, 4, 2, 640, 384), (192, 128), False, torch.bfloat16),
                name="qk192_v128",
            ),
            subtest(
                ((1, 2, 2, 300, 300), (512, 512), False, torch.bfloat16), name="d512"
            ),
            subtest(
                ((1, 4, 2, 1000, 1000), (128, 128), 300, torch.bfloat16),
                name="d128_gqa_window_ragged",
            ),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_backward(self, device, shape, dims, causal, dtype):
        # An int ``causal`` is a sliding window's lookback.
        self._require_runtime()
        batch, q_heads, kv_heads, seq_q, seq_kv = shape
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=dims[0],
            v_dim=dims[1],
            seed=53,
            dtype=dtype,
        )
        block_mask = mask_mod = None
        if causal is True:

            def mask_mod(b, h, q, kv):
                return q >= kv

        elif causal:

            def mask_mod(b, h, q, kv):
                return (q >= kv) & (q - kv < causal)

        if causal:
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            fp64_mask_mod=mask_mod,
            split=True,
        )

    @parametrize("layout", ("transposed", "kv_expanded", "grad_lse"))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_backward_layouts(self, device, layout):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            batch=1 if layout == "kv_expanded" else 2,
            q_heads=4,
            kv_heads=2,
            seq_q=640,
            seq_kv=640,
            seed=59,
            transposed=layout == "transposed",
        )
        if layout == "kv_expanded":
            query = query.expand(2, -1, -1, -1).contiguous()
        self._compare_backward(
            query,
            key,
            value,
            block_mask=None,
            dq_accum_fp32=True,
            grad_lse=layout == "grad_lse",
            expand_kv=layout == "kv_expanded",
            split=True,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_backward_dynamic_shapes(self, device):
        self._require_runtime()
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, enable_gqa=True, kernel_options={"BACKEND": "FLYDSL"}
            ),
            dynamic=True,
            fullgraph=True,
        )
        codes = []
        # 1001 query rows leave a partial 4-row group in the last Q block.
        for batch, seq_q, seq_kv in ((1, 384, 512), (2, 1001, 777)):
            query, key, value = (
                t.requires_grad_()
                for t in _make_qkv(
                    device=device,
                    batch=batch,
                    q_heads=4,
                    kv_heads=2,
                    seq_q=seq_q,
                    seq_kv=seq_kv,
                    seed=61,
                )
            )
            grad_output = torch.randn_like(query)
            # Under dynamic shapes the backward may compile with the forward.
            output, code = run_and_get_code(compiled, query, key, value)
            grads, backward_code = run_and_get_code(
                lambda: torch.autograd.grad(output, (query, key, value), grad_output)
            )
            codes += code + backward_code
            expected = _attention_grads_fp64(
                query, key, value, grad_output, lambda b, h, q, kv: q >= 0
            )
            for grad, expected_grad in zip(grads, expected):
                error = (grad.double() - expected_grad).norm() / expected_grad.norm()
                self.assertLess(error.item(), 0.005)
        codes = "\n".join(codes)
        self.assertIn("build_split_bwd_dq", codes)
        self.assertNotIn("build_flex_attn_bwd_module", codes)

    @parametrize(
        "case",
        ["score_mod", "causal_block_256", "document", "opted_out", "one_kv_block"],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_backward_falls_back_to_fused(self, device, case):
        self._require_runtime()
        if case == "one_kv_block":
            query, key, value = _make_qkv(
                device=device, batch=128, q_heads=8, seq_q=128, qk_dim=64, v_dim=64
            )
        else:
            query, key, value = _make_qkv(device=device, seed=67)
        block_mask = score_mod = None
        options = {}

        def causal(b, h, q, kv):
            return q >= kv

        def two_documents(b, h, q, kv):
            return (q >= kv) & ((q < 128) | (kv >= 128))

        if case == "score_mod":

            def score_mod(score, b, h, q, kv):
                return score * 0.5

        elif case in ("causal_block_256", "document"):
            # Only an explicit opt-in reaches the BlockMask checks.
            options["FLYDSL_SPLIT_BACKWARD"] = True
            block_mask = create_block_mask(
                two_documents if case == "document" else causal,
                1,
                1,
                256,
                256,
                device=device,
                BLOCK_SIZE=128 if case == "document" else 256,
            )
        elif case == "opted_out":
            options["FLYDSL_SPLIT_BACKWARD"] = False
        torch._dynamo.reset()
        query.requires_grad_()
        output, _ = run_and_get_code(
            torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    kernel_options={"BACKEND": "FLYDSL", **options},
                )
            ),
            query,
            key,
            value,
        )
        _, backward_code = run_and_get_code(lambda: output.sum().backward())
        backward_code = "\n".join(backward_code)
        self.assertIn("build_flex_attn_bwd_module", backward_code)
        self.assertNotIn("build_split_bwd_dq", backward_code)

    @parametrize("qk_dim", (128, 192))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_long_causal_dq_matches_fp64(self, device, qk_dim):
        self._require_runtime()
        sequence_length = 16384
        # Multiple KV owners contribute to each private dQ partition.
        query, key, value = _make_qkv(
            device=device, q_heads=16, seq_q=sequence_length, qk_dim=qk_dim, seed=41
        )
        query.requires_grad_()
        torch.manual_seed(42)
        grad_output = torch.randn_like(value)
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: query_index >= key_index,
            1,
            1,
            sequence_length,
            sequence_length,
            device=device,
            BLOCK_SIZE=128,
        )
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLYDSL", "FLYDSL_SPLIT_BACKWARD": True},
            ),
            fullgraph=True,
        )
        output, forward_code = run_and_get_code(compiled, query, key, value)
        gradients, backward_code = run_and_get_code(
            lambda: torch.autograd.grad(output, query, grad_output)
        )
        self.assertIn("build_flex_attn_fwd_module", "\n".join(forward_code))
        self.assertIn("build_split_bwd_dq", "\n".join(backward_code))
        grad_query = gradients[0]
        reference = torch.empty_like(query, dtype=torch.float64)
        scale = qk_dim**-0.5
        # Chunk rows and heads to avoid materializing the full S x S matrix.
        with torch.no_grad():
            key_positions = torch.arange(sequence_length, device=device)
            for head in range(query.size(1)):
                key64 = key[:, head].double()
                value64 = value[:, head].double()
                for start in range(0, sequence_length, 256):
                    end = start + 256
                    scores = (
                        query[:, head, start:end].double() @ key64.transpose(-2, -1)
                    ) * scale
                    rows = torch.arange(start, end, device=device)
                    scores.masked_fill_(
                        rows[:, None] < key_positions[None, :], -float("inf")
                    )
                    probability = scores.softmax(-1)
                    grad_probability = grad_output[
                        :, head, start:end
                    ].double() @ value64.transpose(-2, -1)
                    grad_score = probability * (
                        grad_probability
                        - (probability * grad_probability).sum(-1, keepdim=True)
                    )
                    reference[:, head, start:end] = (grad_score @ key64) * scale
        error = grad_query.double() - reference
        self.assertEqual(grad_query.double(), reference, atol=0.01, rtol=0.02)
        self.assertLess((error.norm() / reference.norm()).item(), 0.003)
        late_rows = sequence_length // 2
        self.assertLess(
            (
                error[:, :, late_rows:].norm() / reference[:, :, late_rows:].norm()
            ).item(),
            0.003,
        )

    @parametrize(
        "q_heads,kv_heads,seq_q,seq_kv,qk_dim,mask_kind,dq_accum_fp32",
        [
            subtest((8, 2, 512, 512, 192, "causal", False), name="gqa4_causal_d192"),
            subtest((16, 2, 512, 512, 128, "document", True), name="gqa8_document"),
            subtest((8, 2, 512, 512, 128, "per_q_head", True), name="gqa4_hq_mask"),
            subtest((4, 1, 256, 256, 128, "dense", False), name="mqa_dense"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_gqa_backward(
        self, device, q_heads, kv_heads, seq_q, seq_kv, qk_dim, mask_kind, dq_accum_fp32
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=qk_dim,
            seed=31,
        )
        mask_heads = 1
        if mask_kind == "dense":

            def mask_mod(b, h, q, kv):
                return q >= 0

        elif mask_kind == "causal":

            def mask_mod(b, h, q, kv):
                return q >= kv

        elif mask_kind == "document":
            document_ids = torch.arange(seq_q, device=device, dtype=torch.int32) // 192
            document_starts = document_ids * 192

            def mask_mod(b, h, q, kv):
                return (q >= kv) & (kv >= document_starts[q])

        else:
            # Head-dependent windows give every Q head its own block lists, so a
            # KV owner visits the union of its group's lists.
            mask_heads = q_heads

            def mask_mod(b, h, q, kv):
                return (q >= kv) & (q - kv < (h % 2 + 1) * 128)

        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, mask_heads, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=dq_accum_fp32,
            fp64_mask_mod=mask_mod,
        )

    @parametrize(
        "seq_q,seq_kv,q_heads,kv_heads,mask_kind",
        [
            subtest((512, 256, 2, 2, "causal"), name="512_256_causal"),
            subtest((384, 640, 4, 2, "dense"), name="384_640_gqa_dense"),
            subtest((640, 384, 4, 1, "causal"), name="640_384_mqa_causal"),
            subtest((256, 1024, 2, 2, "window"), name="256_1024_window"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_cross_attention_backward(
        self, device, seq_q, seq_kv, q_heads, kv_heads, mask_kind
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=37,
        )
        offset = seq_kv - seq_q

        if mask_kind == "dense":

            def mask_mod(b, h, q, kv):
                return q >= 0

        elif mask_kind == "causal":

            def mask_mod(b, h, q, kv):
                return q >= kv

        else:
            # Bottom-right banded window: every query row keeps some keys.
            def mask_mod(b, h, q, kv):
                return (q + offset >= kv) & (q + offset - kv < 256)

        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            fp64_mask_mod=mask_mod,
        )

    @parametrize(
        *_paired(
            ("seq_q,seq_kv,q_heads,kv_heads,mask_kind,score_mod", _RAGGED_CASES),
            ("backward", _DIRECTIONS),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_ragged(
        self, device, backward, seq_q, seq_kv, q_heads, kv_heads, mask_kind, score_mod
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=46 if backward else 45,
        )
        mask_mod = _ragged_mask_mod(mask_kind, seq_q, seq_kv)
        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        if backward:
            self._compare_backward(
                query,
                key,
                value,
                block_mask=block_mask,
                dq_accum_fp32=True,
                fp64_mask_mod=mask_mod if score_mod is None else None,
                score_mod=score_mod,
            )
            return
        _, aux = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
            score_mod=score_mod,
        )
        if mask_kind == "empty_rows":
            self.assertTrue(torch.isneginf(aux.lse[:, :, :200]).all())

    @parametrize(
        *_paired(
            ("block_size,seq_q,seq_kv,mask_kind", _BLOCK_SIZE_CASES),
            ("backward", _DIRECTIONS),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_block_sizes(
        self, device, backward, block_size, seq_q, seq_kv, mask_kind
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=4,
            kv_heads=2,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=48 if backward else 47,
        )
        reference = {"reference_options": _triton_tile_options(block_size)}
        if mask_kind.startswith("block_sparse"):
            # Over a no-op mask_mod only the lists select keys.
            generator = torch.Generator().manual_seed(sum(block_size) + seq_q)
            metadata = _random_block_lists(
                generator, 1, 1, -(-seq_q // block_size[0]), -(-seq_kv // block_size[1])
            )
            mask_mod = (
                None
                if mask_kind == "block_sparse"
                else _ragged_mask_mod(mask_kind, seq_q, seq_kv)
            )

            def from_lists(lists):
                return BlockMask.from_kv_blocks(
                    *(t.to(device) for t in lists),
                    BLOCK_SIZE=block_size,
                    mask_mod=mask_mod,
                    seq_lengths=(seq_q, seq_kv),
                )

            block_mask = from_lists(metadata)
            reference["reference_block_mask"] = from_lists(
                _sorted_block_lists(metadata)
            )
        else:
            block_mask = create_block_mask(
                _ragged_mask_mod(mask_kind, seq_q, seq_kv),
                1,
                1,
                seq_q,
                seq_kv,
                device=device,
                BLOCK_SIZE=block_size,
            )
        if backward:
            self._compare_backward(
                query,
                key,
                value,
                block_mask=block_mask,
                dq_accum_fp32=True,
                fp64_mask_mod=(
                    None
                    if mask_kind.startswith("block_sparse")
                    else _ragged_mask_mod(mask_kind, seq_q, seq_kv)
                ),
                **reference,
            )
            return
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=True,
            return_aux=True,
            **reference,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_lse_aux_backward(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seed=44)
        grad_output = torch.randn_like(value)
        grad_lse = torch.randn(query.shape[:3], device=device, dtype=torch.float32)
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: query_index >= key_index,
            1,
            1,
            256,
            256,
            device=device,
            BLOCK_SIZE=128,
        )

        def run(backend):
            dtype = torch.float64 if backend is None else query.dtype
            inputs = tuple(
                tensor.detach().to(dtype, copy=True).requires_grad_()
                for tensor in (query, key, value)
            )
            compiled = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    return_aux=AuxRequest(lse=True),
                    kernel_options=None
                    if backend is None
                    else {
                        "BACKEND": backend,
                        **(
                            {"FLYDSL_SPLIT_BACKWARD": True}
                            if backend == "FLYDSL"
                            else {}
                        ),
                    },
                ),
                fullgraph=True,
                backend="eager" if backend is None else "inductor",
            )
            (output, aux), forward_code = run_and_get_code(compiled, *inputs)
            gradients, backward_code = run_and_get_code(
                lambda: torch.autograd.grad(
                    (output, aux.lse),
                    inputs,
                    (grad_output.to(dtype), grad_lse.to(aux.lse.dtype)),
                )
            )
            return gradients, "\n".join(forward_code), "\n".join(backward_code)

        gradients, forward_code, backward_code = run("FLYDSL")
        reference, _, _ = run("TRITON")
        golden, _, _ = run(None)
        self.assertIn("build_flex_attn_fwd_module", forward_code)
        self.assertIn("build_split_bwd_dq", backward_code)
        for gradient, expected, exact in zip(gradients, reference, golden):
            self._assert_close_to_fp64(gradient, expected, exact, atol=0.03, rtol=0.02)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_strided_sliding_window_backward(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            batch=2,
            q_heads=4,
            seq_q=512,
            qk_dim=192,
            seed=45,
            transposed=True,
        )
        for tensor in (query, key, value):
            self.assertFalse(tensor.is_contiguous())
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: (
                (query_index >= key_index) & (query_index - key_index < 256)
            ),
            2,
            1,
            512,
            512,
            device=device,
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @parametrize(
        "window_device,window_dtype",
        [
            subtest(("cpu", torch.int64), name="cpu_int64"),
            subtest(("cuda", torch.int32), name="device_int32"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_scalar_tensor_capture(self, device, window_device, window_dtype):
        self._require_runtime()

        def sliding_window_mask(window):
            def sliding_window(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

            return create_block_mask(
                sliding_window, 1, 1, 512, 512, device=device, BLOCK_SIZE=128
            )

        window = torch.tensor(96, device=window_device, dtype=window_dtype)
        query, key, value = _make_qkv(device=device, seq_q=512, seed=46)
        block_mask = sliding_window_mask(window)
        # Triton cannot load a CPU capture, so its reference reads a device copy.
        reference = {"reference_block_mask": sliding_window_mask(window.to(device))}
        self._compare_forward(query, key, value, block_mask=block_mask, **reference)
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True, **reference
        )

    @parametrize(
        "name",
        ["document_int64", "nested_offsets_int64", "float_threshold", "strided_modulo"],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_mask_mod_parity(self, device, name):
        self._require_runtime()
        seq = 512
        positions = torch.arange(seq, device=device)
        documents = (positions // 160).to(torch.int64) * (1 << 33) - (1 << 34)
        offsets = torch.tensor([0], device=device)
        slopes = torch.tensor([0.05, 0.4], device=device, dtype=torch.bfloat16)
        mask_mods = {
            "document_int64": lambda b, h, q, kv: documents[q] == documents[kv],
            "nested_offsets_int64": lambda b, h, q, kv: documents[offsets[b] + q]
            == documents[kv],
            "float_threshold": lambda b, h, q, kv: (q - kv).abs() * slopes[h] < 6.0,
            "strided_modulo": lambda b, h, q, kv: (q >= kv)
            & (((q - kv) % 64 < 16) | (kv // 128 == q // 128)),
        }
        query, key, value = _make_qkv(device=device, seq_q=seq, seed=61)
        block_mask = create_block_mask(
            mask_mods[name], 1, 1, seq, seq, device=device, BLOCK_SIZE=128
        )
        self._compare_forward(query, key, value, block_mask=block_mask, return_aux=True)
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_element_fully_masked_rows(self, device):
        self._require_runtime()

        def causal_skip_rows(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx % 64 != 7)

        query, key, value = _make_qkv(device=device, seq_q=512, seed=47)
        block_mask = create_block_mask(
            causal_skip_rows, 1, 1, 512, 512, device=device, BLOCK_SIZE=128
        )
        output, aux = self._compare_forward(
            query, key, value, block_mask=block_mask, return_aux=True
        )
        masked_rows = torch.arange(7, 512, 64, device=device)
        self.assertEqual(output[:, :, masked_rows].abs().max().item(), 0.0)
        self.assertTrue(torch.isneginf(aux.lse[:, :, masked_rows]).all())
        self.assertTrue(torch.isfinite(output).all())

        inputs = tuple(t.detach().clone().requires_grad_() for t in (query, key, value))
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )
        with DeterministicGuard(True):
            compiled(*inputs).backward(torch.randn_like(value))
        for tensor in inputs:
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertEqual(inputs[0].grad[:, :, masked_rows].abs().max().item(), 0.0)
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @parametrize("case", ["sparse", "edited_causal", "edited_window"])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_respects_block_mask_metadata(self, device, case):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seed=25)
        kv_num_blocks = torch.tensor([[[0, 1]]], device=device, dtype=torch.int32)
        kv_indices = torch.tensor([[[[0], [1]]]], device=device, dtype=torch.int32)
        full_kv_num_blocks = torch.zeros_like(kv_num_blocks)
        full_kv_indices = torch.zeros_like(kv_indices)

        def causal(b, h, q, kv):
            return q >= kv

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(256, 256),
            compute_q_blocks=True,
        )
        if case != "sparse":

            def selected_mask(b, h, q, kv):
                return (
                    (q >= kv) & (q - kv < 512) if case == "edited_window" else q >= kv
                )

            block_mask = create_block_mask(selected_mask, 1, 1, 256, 256, device=device)
            block_mask.full_kv_num_blocks.zero_()
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            split=None,
        )

    @parametrize(
        "q_heads,kv_heads,seq,captured",
        [(4, 2, 512, False), (8, 2, 4096, True)],
    )
    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(
        self, device, q_heads, kv_heads, seq, captured
    ):
        self._require_runtime()

        head_dim = 192 if captured else 128
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=head_dim,
        )
        kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 0] * (seq // 512)]], device=device, dtype=torch.int32
        )
        kv_indices = torch.tensor(
            [[[[0], [1], [2], [0]] * (seq // 512)]], device=device, dtype=torch.int32
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1, 1, 0] * (seq // 512)]], device=device, dtype=torch.int32
        )
        full_kv_indices = torch.tensor(
            [[[[0]] * (seq // 128)]], device=device, dtype=torch.int32
        )

        if captured:
            offsets = torch.zeros(seq, device=device, dtype=torch.int32)

            def causal(b, h, q_idx, kv_idx):
                del b, h
                return q_idx >= kv_idx + offsets[q_idx]

        else:

            def causal(b, h, q_idx, kv_idx):
                del b, h
                return q_idx >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(seq, seq),
            compute_q_blocks=False,
        )
        output, aux = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
            return_aux=True,
        )
        for start in range(384, seq, 512):
            self.assertEqual(output[:, :, start : start + 128].abs().max().item(), 0.0)
            self.assertTrue(torch.isneginf(aux.lse[:, :, start : start + 128]).all())
            self.assertTrue(
                torch.isneginf(aux.max_scores[:, :, start : start + 128]).all()
            )

    @parametrize("seq,document_starts", [(128, [0, 29, 83]), (8192, [0, 1701, 5387])])
    def test_gfx950_forward_unequal_packed_documents(
        self, device, seq, document_starts
    ):
        self._require_runtime()
        starts = torch.tensor(document_starts, device=device, dtype=torch.int32)
        document_ids = torch.bucketize(
            torch.arange(seq, device=device), starts[1:], right=True
        ).to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (kv_idx >= starts[document_ids[q_idx]])

        self._compare_created_mask(
            mask_mod,
            device=device,
            q_heads=16,
            kv_heads=4,
            seq=seq,
            qk_dim=192,
            transposed=True,
            return_aux=True,
        )

    @parametrize("seq_q", [3, 16])
    def test_gfx950_public_api_per_kv_head_decode(self, device, seq_q):
        self._require_runtime()
        batch, q_heads, kv_heads = 1, 64, 4
        seq_kv, head_dim = 8192, 128
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=head_dim,
            seed=9,
        )
        kv_num_blocks = torch.ones(
            1,
            kv_heads,
            1,
            device=device,
            dtype=torch.int32,
        )
        kv_indices = torch.zeros(
            1,
            kv_heads,
            1,
            16,
            device=device,
            dtype=torch.int32,
        )
        kv_indices[..., 0] = 63
        full_kv_num_blocks = torch.full(
            (1, kv_heads, 1),
            15,
            device=device,
            dtype=torch.int32,
        )
        full_kv_indices = torch.arange(
            15,
            device=device,
            dtype=torch.int32,
        ).view(1, 1, 1, 15) + (
            torch.arange(kv_heads, device=device, dtype=torch.int32).view(
                1, kv_heads, 1, 1
            )
            * 8
        )

        q_offset = seq_kv - seq_q

        def bottom_right_causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx + q_offset >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=bottom_right_causal,
            seq_lengths=(seq_q, seq_kv),
            compute_q_blocks=False,
        )
        group_size = q_heads // kv_heads
        reference_block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks.repeat_interleave(group_size, dim=1),
            kv_indices.repeat_interleave(group_size, dim=1),
            full_kv_num_blocks.repeat_interleave(group_size, dim=1),
            full_kv_indices.repeat_interleave(group_size, dim=1),
            BLOCK_SIZE=128,
            mask_mod=bottom_right_causal,
            seq_lengths=(seq_q, seq_kv),
            compute_q_blocks=False,
        )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            reference_block_mask=reference_block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
            return_aux=True,
        )

    @parametrize(
        "block_size,seq_q",
        [
            subtest(((16, 32), 4), name="16x32"),
            subtest(((32, 128), 8), name="32x128"),
        ],
    )
    def test_gfx950_small_block_per_kv_head_decode(self, device, block_size, seq_q):
        self._require_runtime()
        batch, q_heads, kv_heads, seq_kv = 2, 16, 4, 4096
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=19,
        )
        generator = torch.Generator().manual_seed(sum(block_size) + seq_q)
        lists = _random_block_lists(
            generator,
            batch,
            kv_heads,
            -(-seq_q // block_size[0]),
            seq_kv // block_size[1],
            garbage=0,
        )

        def block_mask(group):
            return BlockMask.from_kv_blocks(
                *(t.to(device).repeat_interleave(group, dim=1) for t in lists),
                BLOCK_SIZE=block_size,
                mask_mod=_ragged_mask_mod("block_sparse_mask", seq_q, seq_kv),
                seq_lengths=(seq_q, seq_kv),
            )

        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask(1),
            reference_block_mask=block_mask(q_heads // kv_heads),
            enable_gqa=True,
            return_aux=True,
        )

    @parametrize(
        "qk_dim,v_dim,kv_heads,dtype,dq_accum_fp32",
        [
            subtest((128, 128, 2, torch.float16, True), name="d128_fp16"),
            subtest((192, 128, 2, torch.float16, False), name="d192_fp16_dq16"),
            subtest((64, 64, 1, torch.float16, True), name="d64_fp16_gqa"),
            subtest((128, 64, 2, torch.bfloat16, True), name="qk128_v64_bf16"),
            subtest((256, 128, 2, torch.float16, False), name="qk256_v128_dq16"),
            subtest((64, 256, 1, torch.bfloat16, True), name="qk64_v256_gqa"),
            subtest((512, 512, 1, torch.float16, False), name="d512_fp16_gqa_dq16"),
            subtest((448, 64, 1, torch.float16, True), name="qk448_v64_fp16_gqa"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_dtype_head_dim_backward(
        self, device, qk_dim, v_dim, kv_heads, dtype, dq_accum_fp32
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=2,
            kv_heads=kv_heads,
            seq_q=512,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=23,
            dtype=dtype,
        )
        causal = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, 1, 1, 512, 512, device=device
        )
        for block_mask in (None, causal):
            self._compare_backward(
                query, key, value, block_mask=block_mask, dq_accum_fp32=dq_accum_fp32
            )

    @parametrize(
        *_paired(
            (
                "qk_dim,v_dim,dtype",
                [
                    subtest((128, 128, torch.float16), name="d128_fp16"),
                    subtest((64, 64, torch.bfloat16), name="d64_bf16"),
                    subtest((128, 64, torch.bfloat16), name="qk128_v64_bf16"),
                    subtest((192, 192, torch.bfloat16), name="d192_v192_bf16"),
                    subtest((256, 128, torch.float16), name="qk256_v128_fp16"),
                    subtest((96, 128, torch.float16), name="qk96_v128_fp16"),
                    subtest((80, 80, torch.bfloat16), name="d80_bf16"),
                    subtest((512, 512, torch.bfloat16), name="d512_bf16"),
                    subtest((384, 128, torch.bfloat16), name="qk384_v128_bf16"),
                ],
            ),
            ("seq_q", [1, 4, 1024]),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_dtype_head_dim_gqa_forward(
        self, device, seq_q, qk_dim, v_dim, dtype
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=16,
            kv_heads=4,
            seq_q=seq_q,
            seq_kv=2048,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=24,
            dtype=dtype,
        )
        block_mask = _bottom_right_mask(seq_q, 2048, device)
        self._compare_forward(
            query, key, value, block_mask=block_mask, enable_gqa=True, return_aux=True
        )

    @parametrize(*_paired(("seq_q", [1, 300]), ("head_dim", [32, 80])))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_padded_head_dim_ignores_row_gaps(self, device, seq_q, head_dim):
        # Q, K and V slice a fused projection whose unused columns hold NaNs;
        # the forward reads all three at their stored widths.
        self._require_runtime()
        batch, q_heads, kv_heads, seq_kv = 2, 8, 2, 384
        widths = (q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
        fused = torch.full(
            (batch, seq_kv, sum(widths) + 16),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        )
        torch.manual_seed(25)
        fused[..., : sum(widths)] = torch.randn(
            batch, seq_kv, sum(widths), device=device, dtype=torch.bfloat16
        )
        query, key, value = (
            part.unflatten(-1, (-1, head_dim)).transpose(1, 2)
            for part in fused[..., : sum(widths)].split(widths, dim=-1)
        )
        self._compare_forward(
            query[:, :, :seq_q],
            key,
            value,
            block_mask=_bottom_right_mask(seq_q, seq_kv, device),
            enable_gqa=True,
            return_aux=True,
        )

    @parametrize(
        *_paired(
            (
                "head_dims",
                [
                    subtest((96, 96), name="d96"),
                    subtest((256, 256), name="d256"),
                    subtest((64, 128), name="qk64_v128"),
                    subtest((128, 256), name="qk128_v256"),
                    subtest((80, 80), name="d80"),
                    subtest((160, 160), name="d160"),
                    subtest((512, 512), name="d512"),
                    subtest((320, 320), name="d320"),
                    subtest((96, 384), name="qk96_v384"),
                ],
            ),
            ("shape,mask_kind,score_mod,grad_lse", _HEAD_DIM_CASES),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_head_dim_forward_backward(
        self, device, head_dims, shape, mask_kind, score_mod, grad_lse
    ):
        self._require_runtime()
        qk_dim, v_dim = head_dims
        query, key, value = _make_qkv(
            device=device,
            **{"seq_q": 512, "qk_dim": qk_dim, "v_dim": v_dim, **shape},
            seed=71,
        )
        seq_q, seq_kv = query.size(2), key.size(2)
        if mask_kind == "document":
            document = (torch.arange(seq_q, device=device) // 120).to(torch.int32)

            def mask_mod(b, h, q, kv):
                return document[q] == document[kv]

        else:
            mask_mod = _ragged_mask_mod(mask_kind, seq_q, seq_kv)
        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=query.size(1) != key.size(1),
            return_aux=True,
            score_mod=score_mod,
        )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            fp64_mask_mod=None if score_mod or grad_lse else mask_mod,
            score_mod=score_mod,
            grad_lse=grad_lse,
        )

    def _run_single_pass_and_sliced(self, query, key, value, **kwargs):
        """The FlyDSL forward with and without ``FLYDSL_WIDE_SINGLE_PASS``."""
        results = []
        for single_pass in (True, False):
            torch._dynamo.reset()
            (output, aux), code = run_and_get_code(
                torch.compile(
                    lambda q, k, v: flex_attention(
                        q,
                        k,
                        v,
                        enable_gqa=q.size(1) != k.size(1),
                        kernel_options={
                            "BACKEND": "FLYDSL",
                            "FLYDSL_WIDE_SINGLE_PASS": single_pass,
                        },
                        return_aux=AuxRequest(lse=True, max_scores=True),
                        **kwargs,
                    ),
                    fullgraph=True,
                ),
                query,
                key,
                value,
            )
            source = "\n".join(code)
            self.assertEqual("single_pass=SINGLE_PASS" in source, single_pass)
            self.assertEqual("VALUE_SLICES" in source, not single_pass)
            results.append((output, aux.lse, aux.max_scores))
        torch.cuda.synchronize()
        for single, sliced in zip(*results, strict=True):
            torch.testing.assert_close(single, sliced, atol=2e-2, rtol=2e-2)

    @parametrize(
        *_paired(
            (
                "head_dims",
                [
                    subtest((384, 384), name="d384"),
                    subtest((512, 512), name="d512"),
                    subtest((512, 384), name="qk512_v384"),
                    subtest((512, 256), name="qk512_v256"),
                    subtest((128, 384), name="qk128_v384"),
                    subtest((448, 448), name="d448"),
                ],
            ),
            (
                "shape,mask_kind,score_mod,grad_lse",
                _head_dim_cases("gqa", "cross", "ragged", "fp16", "dlse", "document"),
            ),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_single_pass_forward(
        self, device, head_dims, shape, mask_kind, score_mod, grad_lse
    ):
        self._require_runtime()
        qk_dim, v_dim = head_dims
        query, key, value = _make_qkv(
            device=device,
            **{"seq_q": 512, "qk_dim": qk_dim, "v_dim": v_dim, **shape},
            seed=73,
        )
        seq_q, seq_kv = query.size(2), key.size(2)
        if mask_kind == "document":
            document = (torch.arange(seq_q, device=device) // 120).to(torch.int32)

            def mask_mod(b, h, q, kv):
                return document[q] == document[kv]

        else:
            mask_mod = _ragged_mask_mod(mask_kind, seq_q, seq_kv)
        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=query.size(1) != key.size(1),
            return_aux=True,
            score_mod=score_mod,
            flydsl_options={"FLYDSL_WIDE_SINGLE_PASS": True},
        )
        self._run_single_pass_and_sliced(
            query, key, value, block_mask=block_mask, score_mod=score_mod
        )

    @parametrize(
        *_paired(
            (
                "seq_q,q_heads,kv_heads,seq_kv",
                [
                    subtest((1, 4, 4, 4096), name="one_row_long_kv"),
                    subtest((4, 32, 4, 4096), name="gqa_32_rows"),
                    subtest((16, 32, 2, 2048), name="gqa_256_rows"),
                    subtest((3, 8, 8, 100), name="short_kv"),
                ],
            ),
            (
                "head_dims",
                [
                    subtest((384, 384), name="d384"),
                    subtest((512, 512), name="d512"),
                    subtest((256, 384), name="qk256_v384"),
                ],
            ),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_single_pass_decode(
        self, device, head_dims, seq_q, q_heads, kv_heads, seq_kv
    ):
        self._require_runtime()
        qk_dim, v_dim = head_dims
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=74,
            dtype=torch.float16,
        )
        block_mask = _bottom_right_mask(seq_q, seq_kv, device)
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
            flydsl_options={"FLYDSL_WIDE_SINGLE_PASS": True},
        )
        self._run_single_pass_and_sliced(query, key, value, block_mask=block_mask)

    @parametrize(
        *_paired(
            (
                "head_dims",
                [
                    subtest((384, 384), name="d384"),
                    subtest((512, 256), name="qk512_v256"),
                    subtest((256, 512), name="qk256_v512"),
                    subtest((128, 384), name="qk128_v384"),
                    subtest((64, 512), name="qk64_v512"),
                    subtest((320, 320), name="d320"),
                ],
            ),
            (
                "shape,mask_kind,score_mod,grad_lse",
                _head_dim_cases("gqa", "cross", "ragged", "alibi", "dlse", "document"),
            ),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_single_pass_backward(
        self, device, head_dims, shape, mask_kind, score_mod, grad_lse
    ):
        self._require_runtime()
        qk_dim, v_dim = head_dims
        query, key, value = _make_qkv(
            device=device,
            **{"seq_q": 512, "qk_dim": qk_dim, "v_dim": v_dim, **shape},
            seed=76,
        )
        seq_q, seq_kv = query.size(2), key.size(2)
        if mask_kind == "document":
            document = (torch.arange(seq_q, device=device) // 120).to(torch.int32)

            def mask_mod(b, h, q, kv):
                return document[q] == document[kv]

        else:
            mask_mod = _ragged_mask_mod(mask_kind, seq_q, seq_kv)
        block_mask = None
        if mask_kind != "dense":
            block_mask = create_block_mask(
                mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            fp64_mask_mod=None if score_mod or grad_lse else mask_mod,
            score_mod=score_mod,
            grad_lse=grad_lse,
            flydsl_options={"bwd_FLYDSL_WIDE_SINGLE_PASS": True},
        )
        torch.manual_seed(77)
        grad_output = torch.randn(
            *query.shape[:3], v_dim, device=device, dtype=query.dtype
        )
        grads = []
        for single_pass in (True, False):
            torch._dynamo.reset()
            inputs = [t.detach().clone().requires_grad_() for t in (query, key, value)]
            with DeterministicGuard(True):
                output = torch.compile(
                    lambda q, k, v, single_pass=single_pass: flex_attention(
                        q,
                        k,
                        v,
                        block_mask=block_mask,
                        score_mod=score_mod,
                        enable_gqa=q.size(1) != k.size(1),
                        kernel_options={
                            "BACKEND": "FLYDSL",
                            "bwd_FLYDSL_WIDE_SINGLE_PASS": single_pass,
                        },
                    ),
                    fullgraph=True,
                )(*inputs)
                output.backward(grad_output)
            grads.append([t.grad for t in inputs])
        torch.cuda.synchronize()
        for single, split in zip(*grads, strict=True):
            torch.testing.assert_close(single, split, atol=3e-2, rtol=3e-2)

    @parametrize(
        *_paired(
            (
                "head_dims",
                [
                    subtest((96, 96), name="d96"),
                    subtest((256, 256), name="d256"),
                    subtest((64, 256), name="qk64_v256"),
                    subtest((128, 256), name="qk128_v256"),
                    subtest((80, 80), name="d80"),
                    subtest((160, 160), name="d160"),
                    subtest((512, 512), name="d512"),
                    subtest((384, 256), name="qk384_v256"),
                    subtest((320, 128), name="qk320_v128"),
                ],
            ),
            (
                "seq_q,q_heads,kv_heads,seq_kv",
                [
                    subtest((1, 4, 4, 4096), name="one_row_long_kv"),
                    subtest((4, 32, 4, 4096), name="gqa_32_rows"),
                    subtest((16, 32, 2, 2048), name="gqa_256_rows"),
                    subtest((3, 8, 8, 100), name="short_kv"),
                ],
            ),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_head_dim_decode(
        self, device, head_dims, seq_q, q_heads, kv_heads, seq_kv
    ):
        self._require_runtime()
        qk_dim, v_dim = head_dims
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=72,
            dtype=torch.float16,
        )
        block_mask = _bottom_right_mask(seq_q, seq_kv, device)
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_per_q_head_topk_decode(self, device):
        self._require_runtime()
        batch, q_heads, kv_heads, seq_q = 2, 16, 4, 4
        seq_kv, head_dim, top_k = 4096, 128, 5
        num_kv_blocks = seq_kv // 128
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=head_dim,
            seed=11,
        )
        generator = torch.Generator().manual_seed(5)
        selected = torch.zeros(q_heads, num_kv_blocks, dtype=torch.int32)
        for head in range(q_heads):
            # Shared blocks stay full in the merged mask; the rest become partial.
            shared = torch.tensor([0, head // (q_heads // kv_heads) + 1])
            picked = torch.randperm(num_kv_blocks, generator=generator)[:top_k]
            selected[head, torch.cat([shared, picked])] = 1
        # Mask bytecode does not lower tensor floor division; index per token.
        selected = selected.repeat_interleave(128, dim=1).to(device)
        q_offset = seq_kv - seq_q

        def topk_mask(b, h, q_idx, kv_idx):
            # Thin a few heads inside selected blocks so they are partial.
            thinned = (h % 3 == 0) & ((kv_idx + q_idx) % 5 == 0)
            return (selected[h, kv_idx] == 1) & (q_idx + q_offset >= kv_idx) & ~thinned

        block_mask = create_block_mask(
            topk_mask, batch, q_heads, seq_q, seq_kv, device=device
        )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
            return_aux=True,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_per_q_head_decode_lists_without_mask_mod(self, device):
        # The heads of each GQA group list different blocks and the mask_mod
        # keeps everything, so only the lists keep a head off its neighbours'.
        self._require_runtime()
        q_heads, seq_kv = 8, 2048
        query, key, value = _make_qkv(
            device=device, q_heads=q_heads, kv_heads=2, seq_q=1, seq_kv=seq_kv, seed=23
        )
        heads = torch.arange(q_heads, device=device, dtype=torch.int32)
        counts = torch.ones(1, q_heads, 1, device=device, dtype=torch.int32)
        block_mask = BlockMask.from_kv_blocks(
            counts * 2,
            torch.stack([heads, heads + 8], -1).view(1, q_heads, 1, 2),
            counts,
            ((heads + 3) % 8).view(1, q_heads, 1, 1),
            BLOCK_SIZE=128,
            seq_lengths=(1, seq_kv),
            compute_q_blocks=False,
        )
        self._compare_forward(
            query, key, value, block_mask=block_mask, enable_gqa=True, return_aux=True
        )

    def test_gfx950_public_api_transposed_document_qk192_v128(self, device):
        self._require_runtime()

        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        document_end = torch.tensor(
            [
                [127] * 128 + [255] * 128,
                [63] * 64 + [191] * 128 + [255] * 64,
            ],
            device=device,
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del h
            return (q_idx >= kv_idx) & (q_idx <= document_end[b, kv_idx])

        output, _ = self._compare_created_mask(
            document_causal,
            device=device,
            batch=batch,
            q_heads=heads,
            seq=seq,
            qk_dim=qk_head_dim,
            v_dim=v_head_dim,
            seed=2,
            transposed=True,
            scale=0.07,
            return_aux=True,
        )
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))
        self.assertEqual(output.stride()[-1], 1)

    @parametrize(
        "mask_mod,kwargs",
        [
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    {
                        "q_heads": 4,
                        "kv_heads": 2,
                        "return_aux": True,
                        "seed": 3,
                    },
                ),
                name="sliding_window",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q + 2044 >= kv) & (q + 2044 - kv < 96),
                    {
                        "seq": 4,
                        "seq_kv": 2048,
                        "q_heads": 4,
                        "kv_heads": 2,
                        "return_aux": True,
                    },
                ),
                name="window_decode",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (torch.add(q, 64, alpha=2) >= kv)
                    & ((torch.add(q, 64, alpha=2) - kv) % (2 + h) == 0),
                    {
                        "batch": 2,
                        "q_heads": 3,
                        "seq": 512,
                        "seed": 8,
                        "mask_heads": 3,
                    },
                ),
                name="head_dependent",
            ),
        ],
    )
    def test_gfx950_public_api_supported_masks(self, device, mask_mod, kwargs):
        self._require_runtime()
        self._compare_created_mask(mask_mod, device=device, **kwargs)

    def test_gfx950_public_api_composed_document_mask(self, device):
        self._require_runtime()
        seq = 512
        document_ids = torch.arange(seq, device=device, dtype=torch.int32) // (seq // 2)
        document_starts = torch.tensor([0, seq // 2], device=device, dtype=torch.int32)
        document_causal = and_masks(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            lambda b, h, q_idx, kv_idx: kv_idx >= document_starts[document_ids[q_idx]],
        )
        self._compare_created_mask(
            document_causal, device=device, seq=seq, seed=4, return_aux=True
        )

    def test_gfx950_public_api_sliding_window_mask_lowering(self, device):
        self._require_runtime()

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            is_sliding_window_mask_program,
        )

        def check_match(mask_mod, expected):
            matched = []

            def check_window_lowering(*args, **kwargs):
                program, reason = lower_flydsl_mask_graph(*args, **kwargs)
                self.assertIsNotNone(program, reason)
                matched.append(
                    is_sliding_window_mask_program(
                        program.instructions, program.output, program.buffer_strides
                    )
                )
                return program, reason

            torch._dynamo.reset()
            with (
                torch._inductor.config.patch({"fx_graph_cache": False}),
                mock.patch(
                    "torch._inductor.kernel.flex.flex_flydsl_attention.lower_flydsl_mask_graph",
                    side_effect=check_window_lowering,
                ) as lower,
            ):
                self._compare_created_mask(
                    mask_mod, device=device, seed=3, return_aux=True
                )
                lower.assert_called_once()
            self.assertEqual(matched, [expected])

        # The two spellings lower to different programs, so both need covering:
        # and_masks() seeds a const_bool True that the bare lambda never emits.
        check_match(lambda b, h, q, kv: (q >= kv) & (q - kv < 96), True)
        check_match(
            and_masks(
                lambda b, h, q, kv: q >= kv,
                lambda b, h, q, kv: q - kv < 96,
            ),
            True,
        )
        # Unbounded causal must not match: it is the ragged-tail case the reversed
        # query-block dispatch exists for, and matching here would silently disable it.
        check_match(lambda b, h, q, kv: q >= kv, False)

    def test_gfx950_auto_keeps_flydsl_opt_in(self, device):
        self._require_runtime()

        query, key, value = _make_qkv(device=device, seed=6)

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            256,
            256,
            device=device,
            BLOCK_SIZE=128,
        )

        def run(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "AUTO"},
            )

        torch._dynamo.reset()
        output, code = run_and_get_code(
            torch.compile(run, fullgraph=True),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertFalse(torch.isnan(output).any())
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(code))

    @parametrize(
        *_paired(
            ("mod_name", _RUNTIME_SCORE_MODS),
            (
                "backward,causal",
                [
                    subtest((False, False), name="forward"),
                    subtest((True, True), name="backward_causal"),
                    subtest((False, True), name="forward_causal"),
                    subtest((True, False), name="backward"),
                ],
            ),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod(self, device, backward, mod_name, causal):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, seq_q=512, seed=55 if backward else 51
        )
        block_mask = None
        if causal:
            block_mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                1,
                1,
                512,
                512,
                device=device,
            )
        score_mod = _runtime_score_mod(mod_name, device, heads=2, seq=512)
        if backward:
            self._compare_backward(
                query,
                key,
                value,
                block_mask=block_mask,
                dq_accum_fp32=True,
                score_mod=score_mod,
            )
        else:
            self._compare_forward(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=score_mod,
                return_aux=True,
            )

    @parametrize(
        *_paired(
            ("shape", ["gqa", "mqa", "cross", "fp16", "d64", "v64"]),
            ("mod_name", ["alibi", "softcap", "head_slopes"]),
            ("backward", _DIRECTIONS),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod_shapes(self, device, backward, shape, mod_name):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            seq_q=256,
            seed=60 if backward else 61,
            **_SCORE_MOD_SHAPES[shape],
        )
        score_mod = _runtime_score_mod(
            mod_name, device, heads=query.size(1), seq=key.size(2)
        )
        if backward:
            self._compare_backward(
                query,
                key,
                value,
                block_mask=None,
                dq_accum_fp32=True,
                score_mod=score_mod,
            )
        else:
            self._compare_forward(
                query,
                key,
                value,
                enable_gqa=query.size(1) != key.size(1),
                score_mod=score_mod,
                return_aux=True,
            )

    @parametrize("mod_name", ["softcap", "head_slopes"])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_decode_score_mod(self, device, mod_name):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, q_heads=8, kv_heads=2, seq_q=4, seq_kv=1024, seed=52
        )
        self._compare_forward(
            query,
            key,
            value,
            enable_gqa=True,
            score_mod=_runtime_score_mod(mod_name, device, heads=8, seq=1024),
            return_aux=True,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod_atan2_edges(self, device):
        self._require_runtime()
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        from torch._inductor.codegen.flydsl import flydsl_mod_runtime as runtime
        from torch._inductor.runtime.flydsl_cache import temporary_env

        @flyc.kernel
        def kernel(y: fx.Tensor, x: fx.Tensor, out: fx.Tensor):
            i = fx.Int32(fx.thread_idx.x)
            fx.get_iter(out)[i] = runtime.atan2(fx.get_iter(y)[i], fx.get_iter(x)[i])

        @flyc.jit
        def launch(y: fx.Tensor, x: fx.Tensor, out: fx.Tensor, stream: fx.Stream):
            kernel(y, x, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

        values = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 0.25],
            device=device,
        )
        y, x = (
            t.flatten().contiguous()
            for t in torch.meshgrid(values, values, indexing="ij")
        )
        out = torch.empty_like(x)
        with temporary_env({"FLYDSL_RUNTIME_ENABLE_CACHE": "0"}):
            launch(
                *map(flyc.from_torch_tensor, (y, x, out)), torch.cuda.current_stream()
            )
        torch.cuda.synchronize()
        expected = torch.atan2(y, x)
        self.assertEqual(out.isnan(), expected.isnan())
        finite = expected.isfinite()
        self.assertEqual(out[finite], expected[finite], atol=1e-5, rtol=1e-5)
        self.assertEqual(
            torch.signbit(out[expected == 0]), torch.signbit(expected[expected == 0])
        )

    def test_gfx950_score_mod_scalar_capture(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=256, seed=59)

        def softcap(cap):
            return lambda score, b, h, q_idx, kv_idx: cap * torch.tanh(score / cap)

        expected = self._compare_forward(
            query, key, value, score_mod=softcap(torch.tensor(20.0, device=device))
        )
        # Triton cannot read a CPU scalar capture; compare against the device one.
        actual = torch.compile(
            lambda q, k, v: flex_attention(
                q,
                k,
                v,
                score_mod=softcap(torch.tensor(20.0)),
                kernel_options={"BACKEND": "FLYDSL"},
            ),
            fullgraph=True,
        )(query, key, value)
        self.assertEqual(actual, expected)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod_masks_whole_rows(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=256, seed=53)

        def drop_late_rows(score, b, h, q_idx, kv_idx):
            return torch.where(q_idx < 128, score, -float("inf"))

        output, aux = self._compare_forward(
            query, key, value, score_mod=drop_late_rows, return_aux=True
        )
        self.assertEqual(output[:, :, 128:], torch.zeros_like(output[:, :, 128:]))
        self.assertTrue(torch.isneginf(aux.lse[:, :, 128:]).all())

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod_with_captured_mask_buffer(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=512, seed=54)
        document = (torch.arange(512, device=device) // 192).to(torch.int32)
        slopes = torch.rand(2, device=device)

        def same_document(b, h, q_idx, kv_idx):
            return document[q_idx] == document[kv_idx]

        block_mask = create_block_mask(
            same_document, 1, 1, 512, 512, device=device, BLOCK_SIZE=128
        )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=lambda score, b, h, q_idx, kv_idx: score
            + slopes[h] * (kv_idx - q_idx),
            return_aux=True,
        )

    @parametrize(
        *_paired(("case", ["forward", "decode", "backward"]), ("count", [8, 16]))
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_many_captures(self, device, count, case):
        self._require_runtime()
        seq_q = 4 if case == "decode" else 512
        query, key, value = _make_qkv(
            device=device, q_heads=8, kv_heads=2, seq_q=seq_q, seq_kv=512, seed=56
        )
        generator = torch.Generator().manual_seed(count)
        dtypes = (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool)
        documents = [
            torch.randint(0, 4, (512,), generator=generator)
            .sort()
            .values.to(device, dtypes[index % 5])
            for index in range(count // 2)
        ]
        biases = [
            torch.randn(512, generator=generator).to(
                device, (torch.float32, torch.bfloat16)[index % 2]
            )
            for index in range(count - count // 2)
        ]

        def mask_mod(b, h, q_idx, kv_idx):
            allowed = q_idx + (512 - seq_q) >= kv_idx
            for index, document in enumerate(documents):
                same = document[q_idx] == document[kv_idx]
                allowed = (allowed | same) if index % 3 == 2 else (allowed & same)
            return allowed | (q_idx + (512 - seq_q) == kv_idx)

        def score_mod(score, b, h, q_idx, kv_idx):
            for index, bias in enumerate(biases):
                score = score + 0.1 * bias[kv_idx if index % 2 else q_idx]
            return score

        block_mask = create_block_mask(
            mask_mod, 1, 1, seq_q, 512, device=device, BLOCK_SIZE=128
        )
        if case == "backward":
            self._compare_backward(
                query,
                key,
                value,
                block_mask=block_mask,
                dq_accum_fp32=True,
                score_mod=score_mod,
                grad_lse=True,
            )
            return
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=True,
            return_aux=True,
        )

    @parametrize(
        *_paired(
            (
                "mod,table",
                [
                    subtest(("score_mod", False), name="score_mod"),
                    subtest(("score_mod", True), name="score_mod_table"),
                    subtest(("mask_mod", False), name="mask_mod"),
                    subtest(("mask_mod", True), name="mask_mod_table"),
                ],
            ),
            ("seq_kv", [777, 776]),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_one_hot_capture(self, device, mod, table, seq_kv):
        # A capture that is zero but for one large entry at [Sq - 1, Sk - 1]:
        # a load that misaligns or overruns a row moves it. Rows of an odd Sk
        # fail the 4-alignment of four-key vector loads; Sk = 776 takes them.
        self._require_runtime()
        seq_q = 256
        query, key, value = _make_qkv(
            device=device, seq_q=seq_q, seq_kv=seq_kv, seed=57
        )
        dtype = torch.float32 if mod == "score_mod" else torch.int32
        one_hot = torch.zeros(seq_q, seq_kv, device=device, dtype=dtype)
        one_hot[-1, -1] = 30 if mod == "score_mod" else 1 << 20
        # Four more captures put the build on the capture table.
        zeros = [
            torch.zeros(seq_kv, device=device, dtype=dtype) for _ in range(4 * table)
        ]
        if mod == "score_mod":

            def score_mod(score, b, h, q_idx, kv_idx):
                for zero in zeros:
                    score = score + zero[kv_idx]
                return score + one_hot[q_idx, kv_idx]

            self._compare_forward(
                query, key, value, score_mod=score_mod, return_aux=True
            )
            return

        def mask_mod(b, h, q_idx, kv_idx):
            keep = one_hot[q_idx, kv_idx] == 0
            for zero in zeros:
                keep = keep & (zero[kv_idx] == 0)
            return keep

        block_mask = create_block_mask(
            mask_mod, 1, 1, seq_q, seq_kv, device=device, BLOCK_SIZE=128
        )
        self._compare_forward(query, key, value, block_mask=block_mask, return_aux=True)

    @parametrize(
        "batch,q_heads,kv_heads,seq_q,seq_kv,captures,expand_kv",
        [
            subtest((2, 8, 2, 333, 1000, False, False), name="gqa_ragged_cross"),
            subtest((2, 8, 2, 4, 1024, True, False), name="gqa_decode_captures"),
            subtest((2, 8, 2, 512, 512, False, True), name="expanded_gqa"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_broadcast_kv_batch(
        self, device, batch, q_heads, kv_heads, seq_q, seq_kv, captures, expand_kv
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=61,
        )
        key, value = key[:1].contiguous(), value[:1].contiguous()
        offset = seq_kv - seq_q
        document = (torch.arange(seq_kv, device=device) // 160).to(torch.int32)
        bias = torch.randn(batch, seq_kv, device=device)

        def mask_mod(b, h, q_idx, kv_idx):
            allowed = q_idx + offset >= kv_idx
            if captures:
                allowed = allowed & (
                    (document[q_idx + offset] == document[kv_idx])
                    | (q_idx + offset == kv_idx)
                )
            return allowed

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + 0.5 * bias[b, kv_idx]

        block_mask = create_block_mask(
            mask_mod, 1 if expand_kv else batch, 1, seq_q, seq_kv, device=device
        )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            score_mod=score_mod if captures else None,
            expand_kv=expand_kv,
        )

    @parametrize(
        "q_heads,kv_heads,seq_q,expanded",
        [
            subtest((32, 8, 4, False), name="gqa_decode"),
            subtest((4, 4, 1, True), name="split_kv_expanded"),
            subtest((16, 2, 512, True), name="prefill_expanded"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_broadcast_kv_batch_forward(
        self, device, q_heads, kv_heads, seq_q, expanded
    ):
        self._require_runtime()
        batch, seq_kv = 4, 4096
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=62,
        )
        key, value = key[:1], value[:1]
        if expanded:
            key, value = (t.expand(batch, -1, -1, -1) for t in (key, value))
        block_mask = create_block_mask(
            lambda b, h, q, kv: (q + seq_kv - seq_q >= kv) & (kv % 5 != 3),
            1 if expanded else batch,
            1,
            seq_q,
            seq_kv,
            device=device,
        )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
        )

    @parametrize(
        "batch,q_heads,kv_heads,seq_q,seq_kv,head_dims,kv_batch",
        [
            subtest((1, 4, 2, 1000, 1000, (192, 128), 1), name="gqa_ragged_d192"),
            subtest((2, 8, 2, 4, 1024, (128, 128), 2), name="gqa_decode"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_non_contiguous_head_dim(
        self, device, batch, q_heads, kv_heads, seq_q, seq_kv, head_dims, kv_batch
    ):
        self._require_runtime()
        query, key, value = (
            t.transpose(2, 3).contiguous().transpose(2, 3)
            for t in _make_qkv(
                device=device,
                batch=batch,
                q_heads=q_heads,
                kv_heads=kv_heads,
                seq_q=seq_q,
                seq_kv=seq_kv,
                qk_dim=head_dims[0],
                v_dim=head_dims[1],
                seed=63,
            )
        )
        key, value = key[:kv_batch], value[:kv_batch]
        self.assertNotEqual(query.stride(-1), 1)
        block_mask = _bottom_right_mask(
            seq_q, seq_kv, device, batch=batch if kv_batch != batch else 1
        )
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @parametrize(
        "q_heads,kv_heads,seq_q,head_dims,backward",
        [
            subtest((32, 8, 4, (128, 128), True), name="gqa_d128"),
            subtest((4, 4, 1, (128, 128), False), name="split_kv"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_strided_decode(
        self, device, q_heads, kv_heads, seq_q, head_dims, backward
    ):
        self._require_runtime()
        seq_kv = 4096
        query, key, value = _make_qkv(
            device=device,
            batch=2,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=head_dims[0],
            v_dim=head_dims[1],
            seed=64,
            transposed=True,
        )
        block_mask = _bottom_right_mask(seq_q, seq_kv, device)
        if backward:
            self._compare_backward(
                query, key, value, block_mask=block_mask, dq_accum_fp32=True
            )
            return
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
        )

    @parametrize(
        "shape,case,split_kv",
        [
            subtest(((4, 32, 8, 1, 65613), "causal", True), name="b4_gqa_ragged"),
            subtest(((3, 8, 2, 2, 10000), "lists", 7), name="b3_lists_empty_splits"),
            subtest(((2, 4, 4, 1, 20000), "document_bias", 5), name="b2_document_bias"),
            subtest(((2, 8, 4, 3, 8192), "bshd", 6), name="b2_bshd"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_kv_decode(self, device, shape, case, split_kv):
        self._require_runtime()
        batch, q_heads, kv_heads, seq_q, seq_kv = shape
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=65,
            transposed=case == "bshd",
        )
        block_mask = reference_block_mask = score_mod = None
        if case in ("causal", "bshd"):
            block_mask = _bottom_right_mask(seq_q, seq_kv, device)
        elif case == "lists":
            generator = torch.Generator().manual_seed(66)
            lists = list(
                _random_block_lists(
                    generator, batch, kv_heads, 1, -(-seq_kv // 128), garbage=0
                )
            )
            lists[0][1, 1] = lists[2][1, 1] = 0

            def from_lists(lists):
                return BlockMask.from_kv_blocks(
                    *(t.to(device) for t in lists),
                    BLOCK_SIZE=128,
                    mask_mod=lambda b, h, q, kv: (q + 2 * kv + h) % 3 != 0,
                    seq_lengths=(seq_q, seq_kv),
                )

            block_mask = from_lists(lists)
            # A per-KV-head BlockMask serves every query head of its group.
            reference_block_mask = from_lists(
                _sorted_block_lists(
                    [t.repeat_interleave(q_heads // kv_heads, dim=1) for t in lists]
                )
            )
        elif case == "document_bias":
            documents = (torch.arange(seq_kv, device=device) * 7 // seq_kv).to(
                torch.int32
            )
            bias = torch.randn(q_heads, seq_kv, device=device)

            def mask_mod(b, h, q_idx, kv_idx):
                return documents[kv_idx] != (b + h) % 7

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[h, kv_idx]

            block_mask = create_block_mask(
                mask_mod, batch, q_heads, seq_q, seq_kv, device=device, BLOCK_SIZE=128
            )
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            reference_block_mask=reference_block_mask,
            enable_gqa=q_heads != kv_heads,
            return_aux=True,
            score_mod=score_mod,
            flydsl_options={"FLYDSL_SPLIT_KV": split_kv},
        )

    @parametrize("grad_lse", [False, True])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_split_kv_backward(self, device, grad_lse):
        # The backward reads the merged LSE of the split forward.
        self._require_runtime()
        seq_q, seq_kv = 4, 8192
        query, key, value = _make_qkv(
            device=device,
            batch=2,
            q_heads=8,
            kv_heads=2,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=67,
        )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=_bottom_right_mask(seq_q, seq_kv, device),
            dq_accum_fp32=True,
            grad_lse=grad_lse,
            flydsl_options={"FLYDSL_SPLIT_KV": True},
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_softcap_backward_matches_fp64(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=256, seed=56)
        inputs = [t.detach().clone().requires_grad_() for t in (query, key, value)]
        torch.manual_seed(57)
        grad_output = torch.randn_like(value)
        output = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, score_mod=_softcap_mod, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )(*inputs)
        grads = torch.autograd.grad(output, inputs, grad_output)
        reference_inputs = [
            t.detach().double().requires_grad_() for t in (query, key, value)
        ]
        q64, k64, v64 = reference_inputs
        scores = 30.0 * torch.tanh((q64 @ k64.transpose(-2, -1)) * 128**-0.5 / 30.0)
        reference = scores.softmax(-1) @ v64
        reference_grads = torch.autograd.grad(
            reference, reference_inputs, grad_output.double()
        )
        for grad, reference_grad in zip(grads, reference_grads):
            self.assertEqual(grad.double(), reference_grad, atol=0.04, rtol=0.03)

    @parametrize(
        "dtype,value,seq,dualwave",
        [
            subtest((torch.bfloat16, 133120.0, 1024, False), name="bf16_seq1024"),
            subtest((torch.float16, 32768.0, 1024, False), name="fp16_seq1024"),
            subtest((torch.bfloat16, 133120.0, 1, False), name="bf16_seq1"),
            subtest((torch.bfloat16, 176.0, 1024, False), name="bf16_moderate_seq1024"),
            subtest((torch.float16, 176.0, 1, False), name="fp16_moderate_seq1"),
            subtest(
                (torch.bfloat16, 133120.0, 1024, True), name="bf16_seq1024_dualwave"
            ),
            subtest((torch.float16, 32768.0, 1000, True), name="fp16_seq1000_dualwave"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_large_logits_match_fp64(self, device, dtype, value, seq, dualwave):
        # Q = K = V filled with one value gives equal, large logits.
        self._require_runtime()
        query, key, value_tensor = (
            torch.full((1, seq, 2, 128), value, device=device, dtype=dtype).transpose(
                1, 2
            )
            for _ in range(3)
        )
        if dualwave:
            inputs, output, _, code, _ = self._run_flydsl_forward(
                query,
                key,
                value_tensor,
                None,
                force_dualwave=True,
                requires_grad=True,
                scale=0.125,
            )
            self.assertIn("build_flex_attn_fwd_dualwave_module", code)
        else:
            inputs = [
                t.detach().clone().requires_grad_() for t in (query, key, value_tensor)
            ]
            output, code = run_and_get_code(
                torch.compile(
                    lambda q, k, v: flex_attention(
                        q, k, v, scale=0.125, kernel_options={"BACKEND": "FLYDSL"}
                    ),
                    fullgraph=True,
                ),
                *inputs,
            )
            self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        # Only the fused backward repeats the forward's rounding of logits
        # near 1e11, which keeps P <= 1; the split one runs moderate logits.
        # Single-row queries take the fused backward regardless.
        extreme = value >= 1024
        with _fused_backward() if extreme else contextlib.nullcontext():
            _, backward_code = run_and_get_code(lambda: output.sum().backward())
        self.assertIn(
            "build_flex_attn_bwd_module"
            if extreme or seq == 1
            else "build_split_bwd_dq",
            "\n".join(backward_code),
        )

        reference_inputs = [
            t.detach().double().requires_grad_() for t in (query, key, value_tensor)
        ]
        q64, k64, v64 = reference_inputs
        reference = ((q64 @ k64.transpose(-2, -1)) * 0.125).softmax(-1) @ v64
        reference.sum().backward()
        self.assertEqual(output.double(), reference, atol=0, rtol=0.01)
        grads = [t.grad.double() for t in inputs]
        for grad in grads:
            self.assertTrue(grad.isfinite().all())
        for grad, reference_input in zip(grads[:2], reference_inputs[:2]):
            self.assertEqual(grad, reference_input.grad, atol=1.0, rtol=0)
        # A float32 log2 LSE near 1e11 cannot hold log2(Sk), so P is only exact
        # at one key or at moderate logits, and there within half an LSE ulp.
        if seq == 1 or value < 1024:
            log2_lse = value * value * query.size(-1) * 0.125 * math.log2(math.e)
            ulp = 2.0 ** (math.floor(math.log2(log2_lse)) - 23)
            rtol = 0.01 if seq == 1 else 0.01 + 2 ** (ulp / 2) - 1
            self.assertEqual(grads[2], reference_inputs[2].grad, atol=0, rtol=rtol)

    @parametrize(
        *_paired(
            ("shape", ["mha", "gqa", "mqa", "cross", "fp16", "d64"]),
            ("mod_name", ["none", "alibi", "softcap"]),
            ("causal", [False, True]),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_grad_logsumexp(self, device, shape, causal, mod_name):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, seq_q=256, seed=61, **_SCORE_MOD_SHAPES.get(shape, {})
        )
        seq_q, seq_kv = query.size(2), key.size(2)
        block_mask = _bottom_right_mask(seq_q, seq_kv, device) if causal else None
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            score_mod=None
            if mod_name == "none"
            else _runtime_score_mod(mod_name, device, heads=query.size(1), seq=seq_kv),
            grad_lse=True,
        )

    @parametrize(
        *_paired(
            (
                "seq_q,seq_kv,q_heads,kv_heads,block_size",
                [
                    subtest((1000, 1000, 2, 2, 128), name="1000"),
                    subtest((77, 77, 4, 2, 128), name="gqa_77"),
                    subtest((256, 1000, 2, 2, 128), name="cross_ragged_kv"),
                    subtest((333, 200, 8, 2, 128), name="gqa_cross_ragged"),
                    subtest((1000, 1000, 4, 2, (256, 128)), name="256x128"),
                ],
            ),
            ("causal", [False, True]),
        )
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_grad_logsumexp_ragged(
        self, device, seq_q, seq_kv, q_heads, kv_heads, block_size, causal
    ):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            seed=63,
        )
        block_mask = (
            create_block_mask(
                (lambda b, h, q_idx, kv_idx: q_idx + seq_kv - seq_q >= kv_idx)
                if causal
                else (lambda b, h, q_idx, kv_idx: kv_idx >= 0),
                1,
                1,
                seq_q,
                seq_kv,
                device=device,
                BLOCK_SIZE=block_size,
            )
            if causal or block_size != 128
            else None
        )
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True, grad_lse=True
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_grad_logsumexp_empty_rows(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=256, seed=62)
        grad_logsumexp = torch.randn(1, 2, 256, device=device)
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: (q_idx >= 64) & (q_idx >= kv_idx),
            1,
            1,
            256,
            256,
            device=device,
        )

        def run(backend):
            dtype = torch.float64 if backend is None else query.dtype
            inputs = [
                t.detach().to(dtype, copy=True).requires_grad_()
                for t in (query, key, value)
            ]
            out, aux = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    return_aux=AuxRequest(lse=True),
                    kernel_options=None if backend is None else {"BACKEND": backend},
                ),
                fullgraph=True,
                backend="eager" if backend is None else "inductor",
            )(*inputs)
            # Empty rows have a -inf LSE, so the loss is -inf but its gradient is finite.
            (out.float().sum() + (aux.lse * grad_logsumexp).sum()).backward()
            return [t.grad for t in inputs]

        grads = run("FLYDSL")
        for grad, reference_grad, golden_grad in zip(grads, run("TRITON"), run(None)):
            self.assertTrue(grad.isfinite().all())
            self._assert_close_to_fp64(
                grad, reference_grad, golden_grad, atol=0.04, rtol=0.03
            )
        self.assertEqual(grads[0][:, :, :64], torch.zeros_like(grads[0][:, :, :64]))

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_captured_tensor_gradient_falls_back_to_triton(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seq_q=256, seed=58)
        query.requires_grad_()
        bias = torch.randn(1, 2, 256, 256, device=device, requires_grad=True)
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q,
                k,
                v,
                score_mod=lambda score, b, h, q_idx, kv_idx: score
                + bias[b, h, q_idx, kv_idx],
                kernel_options={"BACKEND": "FLYDSL"},
            ),
            fullgraph=True,
        )
        output, forward_code = run_and_get_code(compiled, query, key, value)
        _, backward_code = run_and_get_code(lambda: output.sum().backward())
        self.assertIn("build_flex_attn_fwd_module", "\n".join(forward_code))
        self.assertNotIn("build_flex_attn_bwd_module", "\n".join(backward_code))
        self.assertIsNotNone(bias.grad)


instantiate_device_type_tests(TestFlyDSLFlexAttention, globals(), only_for=("cuda",))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
