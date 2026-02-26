# Owner(s): ["module: inductor"]

import functools
import unittest
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch._dynamo.testing import CompileCounterWithBackend, EagerAndRecordGraphs
from torch._inductor.kernel.flex.flex_flash_attention import (
    _hierarchical_indexer_cute,
    ensure_flash_available,
    HierarchicalIndex,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    flex_attention,
)
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_cuda import (
    IS_SM90,
    PLATFORM_SUPPORTS_FP8,
    xfailIfSM90,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    e4m3_type,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    decorateIf,
    DeterministicGuard,
    parametrize,
)


def _times_two(score, _b, _h, _m, _n):
    return score * 2


def _causal(score, _b, _h, token_q, token_kv):
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _causal_mask(_b, _h, token_q, token_kv):
    return token_q >= token_kv


def _rel_bias(score, _b, _h, token_q, token_kv):
    return score + (token_q - token_kv)


def _score_squared(score, _b, _h, _q, _k):
    return score * score


def _softcap(score, _b, _h, _q, _k):
    cap = 50.0
    return torch.tanh(score / cap) * cap


def _head_offset(score, _b, h, _q, _k):
    return score + h * 0.1


def _distance_decay(score, _b, _h, q_idx, kv_idx):
    distance_sq = (q_idx - kv_idx) * (q_idx - kv_idx)
    decay = 1.0 / (1.0 + distance_sq * 0.0001)
    return score * decay


def create_alibi_learned(num_heads=4, dtype=torch.float16):
    """ALiBi with learned per-head slopes (tests tensor loading)."""
    slopes = torch.exp2(-torch.linspace(1, 8, num_heads, device="cuda", dtype=dtype))

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        bias = (kv_idx - q_idx) * slopes[h]
        return score + bias

    return alibi_score_mod


def create_pos_bias_table(seq_len=512, dtype=torch.float16):
    """Relative position bias table (tests computed indexing)."""
    max_len = seq_len
    table = torch.randn(2 * max_len - 1, device="cuda", dtype=dtype) * 0.1

    def pos_bias_mod(score, b, h, q_idx, kv_idx):
        rel_pos = kv_idx - q_idx + max_len - 1
        bias = table[rel_pos]
        return score + bias

    return pos_bias_mod


def create_head_scale(num_heads=4, dtype=torch.float16):
    """Per-head scaling factors (tests multiplication with tensor loading)."""
    scales = torch.rand(num_heads, device="cuda", dtype=dtype) + 0.5

    def head_scale_mod(score, b, h, q_idx, kv_idx):
        return score * scales[h]

    return head_scale_mod


def create_batch_bias(batch_size=2, dtype=torch.float16):
    """Per-batch bias (tests batch indexing)."""
    bias = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1

    def batch_bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b]

    return batch_bias_mod


def create_batch_head_bias(batch_size=2, num_heads=4, dtype=torch.float16):
    """Per-batch-head bias matrix (tests 2D indexing with batch + head)."""
    bias_matrix = torch.randn(batch_size, num_heads, device="cuda", dtype=dtype) * 0.5

    def batch_head_mod(score, b, h, q_idx, kv_idx):
        bias = bias_matrix[b, h]
        return score + bias

    return batch_head_mod


def create_dual_buffer_bias(num_heads=4, seq_len=512, dtype=torch.float16):
    """Dual buffer loading (tests loading from 2 separate tensors)."""
    head_bias = torch.randn(num_heads, device="cuda", dtype=dtype) * 0.2
    pos_scale = torch.arange(seq_len, device="cuda", dtype=dtype)

    def dual_buffer_mod(score, b, h, q_idx, kv_idx):
        head_component = head_bias[h]
        pos_component = pos_scale[q_idx] * 0.01
        return score + head_component + pos_component

    return dual_buffer_mod


def create_score_view_mod(num_heads=4, dtype=torch.float16, device="cuda"):
    base_scales = torch.rand(num_heads, 2, device=device, dtype=dtype) + 0.5
    scales_view = base_scales[:, 0]
    if scales_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def score_view_mod(score, _b, h, _q_idx, _kv_idx):
        return score + scales_view[h]

    return score_view_mod


def create_complex_score_mod(
    batch_size=2, num_heads=4, seq_len=512, dtype=torch.float16, device="cuda"
):
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.15
    query_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    kv_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.1

    def complex_score(score, b, h, q_idx, kv_idx):
        head_term = head_bias[h]
        query_term = query_scale[q_idx] - kv_scale[kv_idx]
        batch_term = batch_bias[b]
        return score + head_term + query_term + batch_term

    return complex_score


def create_score_mod_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    score_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2

    def score_with_buffer(score, _b, h, _q_idx, _kv_idx):
        return score + score_bias[h]

    return score_with_buffer


def create_mask_mod_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    mask_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.1

    def custom_mask(_b, h, q_idx, kv_idx):
        bias_value = mask_bias[h]
        return (q_idx >= kv_idx) | (bias_value > 0)

    return custom_mask


def create_document_mask(lengths_per_batch, device):
    document_ids = []
    for lengths in lengths_per_batch:
        doc_tokens = []
        for doc_id, length in enumerate(lengths):
            doc_tokens.extend([doc_id] * length)
        document_ids.append(doc_tokens)
    document_ids = torch.tensor(document_ids, device=device, dtype=torch.long)

    def document_mask(b, _h, q_idx, kv_idx):
        doc_id_q = document_ids[b, q_idx // 2]
        doc_id_kv = document_ids[b, kv_idx]
        return doc_id_q == doc_id_kv

    return document_mask


def create_mask_mod_view_buffer(num_heads=4, dtype=torch.float16, device="cuda"):
    base_bias = torch.randn(num_heads, 3, device=device, dtype=dtype)
    mask_bias_view = base_bias[:, 1]
    if mask_bias_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def mask_with_view_buffer(_b, h, q_idx, kv_idx):
        bias_value = mask_bias_view[h]
        double_bias = bias_value * 2
        return (q_idx >= kv_idx) | (double_bias > 0)

    return mask_with_view_buffer


def create_mask_mod_dual_buffers(
    batch_size=2, num_heads=4, dtype=torch.float16, device="cuda"
):
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.2

    def dual_buffer_mask(b, h, q_idx, kv_idx):
        head_term = head_bias[h]
        batch_term = batch_bias[b]
        causal = q_idx >= kv_idx
        bias_cond = (head_term + batch_term).to(torch.float32) > 0
        return causal | bias_cond

    return dual_buffer_mask


def create_test_tensors(
    batch_size=2,
    num_heads=4,
    seq_len=512,
    dim=64,
    dtype=torch.float16,
    device="cuda",
    requires_grad=False,
    num_heads_kv=None,
):
    num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
    q_shape = (batch_size, num_heads, seq_len, dim)
    kv_shape = (batch_size, num_heads_kv, seq_len, dim)
    q = torch.randn(q_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def _create_block_mask_for_device(
    mask_mod, batch_size, num_heads, q_len, kv_len, *, device
):
    """Match FlexAttention's block-height expectations per compute capability."""
    q_block = _DEFAULT_SPARSE_BLOCK_SIZE
    kv_block = _DEFAULT_SPARSE_BLOCK_SIZE
    dev = torch.device(device)
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        if major >= 10:
            q_block *= 2
    return create_block_mask(
        mask_mod,
        batch_size,
        num_heads,
        q_len,
        kv_len,
        device=device,
        BLOCK_SIZE=(q_block, kv_block),
    )


@contextmanager
def cuda_kernel_profiler(kernel_pattern="flash_attncute"):
    """Context manager for profiling CUDA kernels."""
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


def flash_vs_triton(q, k, v, score_mod=None, block_mask=None, rtol=2, *, dynamic=False):
    compiled_fn = torch.compile(flex_attention, dynamic=dynamic)
    enable_gqa = q.shape[1] != k.shape[1]

    out_ref_fp32 = flex_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    ).to(q.dtype)

    out_flash = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options={"BACKEND": "FLASH"},
    )
    out_triton = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options={"BACKEND": "TRITON"},
    )

    if not (out_flash.shape == out_ref_fp32.shape == out_triton.shape):
        raise AssertionError(
            f"Shape mismatch: flash={out_flash.shape}, ref={out_ref_fp32.shape}, triton={out_triton.shape}"
        )
    if torch.isnan(out_flash).any():
        raise AssertionError("out_flash contains NaN")
    if torch.isnan(out_triton).any():
        raise AssertionError("out_triton contains NaN")
    if torch.isnan(out_ref_fp32).any():
        raise AssertionError("out_ref_fp32 contains NaN")
    if not torch.isfinite(out_flash).all():
        raise AssertionError("out_flash contains non-finite values")
    if not torch.isfinite(out_triton).all():
        raise AssertionError("out_triton contains non-finite values")
    if not torch.isfinite(out_ref_fp32).all():
        raise AssertionError("out_ref_fp32 contains non-finite values")

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()

    triton_error = (out_triton - out_ref_fp32).abs().max().item()
    flash_error = (out_flash - out_ref_fp32).abs().max().item()

    if flash_error > rtol * triton_error + fwd_atol:
        raise AssertionError(
            f"Flash error {flash_error:.2e} exceeds {rtol}x Triton error {triton_error:.2e} + {fwd_atol:.2e}"
        )

    needs_backward = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v)
    )
    if needs_backward:
        grad = torch.randn_like(out_flash)
        inputs = (q, k, v)
        grads_ref = torch.autograd.grad(out_ref_fp32, inputs, grad)
        grads_triton = torch.autograd.grad(out_triton, inputs, grad)
        grads_flash = torch.autograd.grad(out_flash, inputs, grad)

        dq_atol = 2 * (grads_ref[0] + 0.3 - 0.3 - grads_ref[0]).abs().max().item()
        dk_atol = 2 * (grads_ref[1] + 0.3 - 0.3 - grads_ref[1]).abs().max().item()
        dv_atol = 2 * (grads_ref[2] + 0.3 - 0.3 - grads_ref[2]).abs().max().item()

        atol_pack = (dq_atol, dk_atol, dv_atol)
        for grad_flash, grad_triton, grad_ref, atol in zip(
            grads_flash, grads_triton, grads_ref, atol_pack
        ):
            if not torch.isfinite(grad_flash).all():
                raise AssertionError("grad_flash contains non-finite values")
            if not torch.isfinite(grad_triton).all():
                raise AssertionError("grad_triton contains non-finite values")
            if not torch.isfinite(grad_ref).all():
                raise AssertionError("grad_ref contains non-finite values")

            triton_error = (grad_triton - grad_ref).abs().max().item()
            flash_error = (
                (grad_flash - grad_ref.to(grad_flash.dtype)).abs().max().item()
            )
            if flash_error > rtol * triton_error + atol:
                raise AssertionError(
                    f"Flash error {flash_error:.2e} exceeds {rtol}x Triton error {triton_error:.2e} + {atol:.2e}"
                )

    return out_flash, out_triton, out_ref_fp32


@dataclass
class ScoreModCase:
    name: str
    score_mod_factory: Callable[[torch.dtype, str], Callable | None]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    seq_len: int = 512
    dim: int = 64
    requires_grad: bool = False


@dataclass
class MaskModCase:
    name: str
    mask_mod_factory: Callable[[torch.dtype, str], Callable]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    block_mask_num_heads: int | None = None
    seq_len: int = 512
    dim: int = 64
    score_mod_factory: Callable[[torch.dtype, str], Callable | None] | None = None
    requires_grad: bool = False


def score_case_name(case: ScoreModCase):
    return case.name


def mask_case_name(case: MaskModCase):
    return case.name


SCORE_MOD_CASES = [
    ScoreModCase("basic", lambda _dtype, _device: None),
    ScoreModCase("times_two", lambda _dtype, _device: _times_two),
    ScoreModCase("causal", lambda _dtype, _device: _causal),
    ScoreModCase("rel_bias", lambda _dtype, _device: _rel_bias),
    ScoreModCase("causal_unfriendly_127", lambda _dtype, _device: _causal, seq_len=127),
    ScoreModCase("causal_unfriendly_255", lambda _dtype, _device: _causal, seq_len=255),
    ScoreModCase("causal_unfriendly_383", lambda _dtype, _device: _causal, seq_len=383),
    ScoreModCase("causal_unfriendly_511", lambda _dtype, _device: _causal, seq_len=511),
    ScoreModCase(
        "alibi_learned",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "pos_bias_table",
        lambda dtype, device: create_pos_bias_table(seq_len=512, dtype=dtype),
    ),
    ScoreModCase(
        "head_scale",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "batch_bias",
        lambda dtype, device: create_batch_bias(batch_size=2, dtype=dtype),
    ),
    ScoreModCase(
        "batch_head_bias",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
    ),
    ScoreModCase(
        "dual_buffer_bias",
        lambda dtype, device: create_dual_buffer_bias(
            num_heads=4, seq_len=512, dtype=dtype
        ),
    ),
    ScoreModCase(
        "score_view_buffer",
        lambda dtype, device: create_score_view_mod(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "score_many_buffers",
        lambda dtype, device: create_complex_score_mod(
            batch_size=2, num_heads=4, seq_len=512, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "backward_times_two",
        lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_score_squared",
        lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_rel_bias",
        lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_softcap",
        lambda _dtype, _device: _softcap,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_offset",
        lambda _dtype, _device: _head_offset,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_distance_decay",
        lambda _dtype, _device: _distance_decay,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_alibi_captured",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_scale_captured",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_batch_head_bias_captured",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_pos_bias_table_captured",
        lambda dtype, device: create_pos_bias_table(seq_len=257, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "gqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "mqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "backward_gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_gqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
]

DETERMINISTIC_SCORE_MOD_CASES = [case for case in SCORE_MOD_CASES if case.requires_grad]


MASK_MOD_CASES = [
    MaskModCase("block_mask_causal", lambda _dtype, _device: _causal_mask),
    MaskModCase(
        "block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
    ),
    MaskModCase(
        "mask_mod_buffer",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "doc_mask",
        lambda _dtype, device: create_document_mask(
            ((16, 31, 25, 56), (40, 9, 23, 56)), device=device
        ),
        block_mask_num_heads=1,
        seq_len=128,
    ),
    MaskModCase(
        "mask_mod_view_buffer",
        lambda dtype, device: create_mask_mod_view_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        requires_grad=True,
    ),
    MaskModCase(
        "mask_mod_dual_buffers",
        lambda dtype, device: create_mask_mod_dual_buffers(
            batch_size=2, num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "score_and_mask_buffers",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        score_mod_factory=lambda dtype, device: create_score_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "backward_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_squared",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_rel_bias",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
]

DETERMINISTIC_MASK_MOD_CASES = [case for case in MASK_MOD_CASES if case.requires_grad]

GQA_MQA_BLOCK_MASK_CASES = [
    MaskModCase(
        "gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
]


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
class TestFlexFlash(InductorTestCase):
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["case"].requires_grad and IS_SM90,
    )
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", SCORE_MOD_CASES, name_fn=score_case_name)
    def test_flash_attention_score_mod_cases(self, device, dtype, case):
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=case.score_mod_factory(dtype, device),
        )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_SCORE_MOD_CASES, name_fn=score_case_name)
    def test_flash_attention_backward_deterministic_score_mod_cases(
        self, device, dtype, case
    ):
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        with DeterministicGuard(True):
            flash_vs_triton(
                q,
                k,
                v,
                score_mod=case.score_mod_factory(dtype, device)
                if case.score_mod_factory
                else None,
            )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_mask_mod_cases(self, device, dtype, case):
        if case.requires_grad:
            major, _ = torch.cuda.get_device_capability()
            if major < 9:
                self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=(
                case.score_mod_factory(dtype, device)
                if case.score_mod_factory
                else None
            ),
            block_mask=_create_block_mask_for_device(
                case.mask_mod_factory(dtype, device),
                case.batch_size,
                case.block_mask_num_heads or case.num_heads,
                case.seq_len,
                case.seq_len,
                device=device,
            ),
        )

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_backward_deterministic_block_mask_raises(
        self, device, dtype, case
    ):
        from torch._dynamo.exc import BackendCompilerFailed

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        block_mask = _create_block_mask_for_device(
            case.mask_mod_factory(dtype, device),
            case.batch_size,
            case.block_mask_num_heads or case.num_heads,
            case.seq_len,
            case.seq_len,
            device=device,
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with DeterministicGuard(True):
            with self.assertRaisesRegex(
                BackendCompilerFailed,
                "Deterministic backward for flex_attention with block_mask using the FLASH backend",
            ):
                out = compiled_fn(
                    q,
                    k,
                    v,
                    score_mod=(
                        case.score_mod_factory(dtype, device)
                        if case.score_mod_factory
                        else None
                    ),
                    block_mask=block_mask,
                    kernel_options={"BACKEND": "FLASH"},
                )
                out.sum().backward()

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", DETERMINISTIC_MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_backward_deterministic_warn_only_block_mask(
        self, device, dtype, case
    ):
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        block_mask = _create_block_mask_for_device(
            case.mask_mod_factory(dtype, device),
            case.batch_size,
            case.block_mask_num_heads or case.num_heads,
            case.seq_len,
            case.seq_len,
            device=device,
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with DeterministicGuard(True, warn_only=True):
            out = compiled_fn(
                q,
                k,
                v,
                score_mod=(
                    case.score_mod_factory(dtype, device)
                    if case.score_mod_factory
                    else None
                ),
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLASH"},
            )
            with self.assertWarnsRegex(
                UserWarning,
                "Deterministic backward for flex_attention with block_mask",
            ):
                out.sum().backward()

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", GQA_MQA_BLOCK_MASK_CASES, name_fn=mask_case_name)
    def test_flash_attention_gqa_mqa_block_mask_cases(self, device, dtype, case):
        if case.requires_grad:
            major, _ = torch.cuda.get_device_capability()
            if major < 9:
                self.skipTest("block sparse backward only supported on SM90+ for FLASH")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=dtype,
            device=device,
            requires_grad=case.requires_grad,
        )
        flash_vs_triton(
            q,
            k,
            v,
            score_mod=(
                case.score_mod_factory(dtype, device)
                if case.score_mod_factory
                else None
            ),
            block_mask=_create_block_mask_for_device(
                case.mask_mod_factory(dtype, device),
                case.batch_size,
                case.block_mask_num_heads or case.num_heads,
                case.seq_len,
                case.seq_len,
                device=device,
            ),
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_kernel_called(self, device, dtype):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        compiled_fn = torch.compile(flex_attention)

        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(q, k, v, score_mod=_causal, kernel_options={"BACKEND": "FLASH"})

        self.assertTrue(
            prof_result["found"],
            f"Flash attention kernel not found. Available kernels: {prof_result['kernel_names']}",
        )

        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(
                q, k, v, score_mod=_causal, kernel_options={"BACKEND": "TRITON"}
            )

        self.assertFalse(
            prof_result["found"],
            f"Flash attention kernel unexpectedly found when BACKEND='TRITON'. Kernels: {prof_result['kernel_names']}",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_impl_error_with_requires_grad(self, device, dtype):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)

        def score_mod_with_grad(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='FLASH' but flash attention cannot be used.*require gradients",
        ):
            compiled_fn(
                q,
                k,
                v,
                score_mod=score_mod_with_grad,
                kernel_options={"BACKEND": "FLASH"},
            )

    def test_mixed_dtypes(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        """Ensure flash attention rejects mixed dtypes (e.g., fp32 Q with fp16 K/V)"""
        B, H, S, D = 2, 8, 512, 64

        query = torch.randn(B, H, S, D, dtype=dtype_high, device=device)
        key = torch.randn(B, H, S, D, dtype=dtype_high, device=device).to(dtype_low)
        value = torch.randn(B, H, S, D, dtype=dtype_high, device=device).to(dtype_low)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        from torch._inductor.exc import InductorError

        with self.assertRaisesRegex(
            InductorError,
            "Mixed query, key, and value dtype is not supported on this platform",
        ):
            compiled_fn(query, key, value, kernel_options={"BACKEND": "FLASH"})

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_rejects_captured_buffer_with_grad(
        self, device, dtype
    ):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)

        def score_mod_with_capture(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        q.requires_grad_(True)
        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='FLASH' but flash attention cannot be used.*require gradients",
        ):
            compiled_fn(
                q,
                k,
                v,
                score_mod=score_mod_with_capture,
                kernel_options={"BACKEND": "FLASH"},
            ).sum().backward()

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_kernel_called(self, device, dtype):
        q, k, v = create_test_tensors(dim=128, dtype=dtype, device=device)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        def run_for_profile():
            q_run, k_run, v_run = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v)
            ]
            compiled_fn(
                q_run, k_run, v_run, kernel_options={"BACKEND": "FLASH"}
            ).sum().backward()

        with cuda_kernel_profiler("flash_attncuteflash_bwd") as prof_result:
            run_for_profile()

        self.assertTrue(
            prof_result["found"],
            f"Flash attention backward kernel not found. Kernels: {prof_result['kernel_names']}",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_forwards_deterministic_flag(self, device, dtype):
        q, k, v = create_test_tensors(dim=128, dtype=dtype, device=device)
        q.requires_grad_(True)

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        def run_for_code():
            return compiled_fn(q, k, v, kernel_options={"BACKEND": "FLASH"})

        _, code = run_fw_bw_and_get_code(run_for_code)
        code_str = "\n".join(code)
        self.assertIn(
            "are_deterministic_algorithms_enabled()",
            code_str,
            "Expected deterministic flag to be wired through flash backward",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_generates_cute_hash(self, device, dtype):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=_causal,
            kernel_options={"BACKEND": "FLASH"},
        )

        code_str = "\n".join(code)
        self.assertIn(
            "score_mod.__cute_hash__",
            code_str,
            "Generated code should set __cute_hash__ on score_mod for fast hashing",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_fused_qkv_reinterpret_view(self, device, dtype):
        B, M, H, D = 2, 256, 4, 64
        embed_dim = H * D

        def fn(x, weight):
            qkv = x @ weight
            qkv = qkv.view(B, M, 3, H, D)
            q, k, v = qkv.unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            return flex_attention(q, k, v, kernel_options={"BACKEND": "FLASH"})

        x = torch.randn(B, M, embed_dim, device=device, dtype=dtype)
        weight = torch.randn(embed_dim, 3 * embed_dim, device=device, dtype=dtype)

        compiled_fn = torch.compile(fn)
        out = compiled_fn(x, weight)
        self.assertEqual(out.shape, (B, H, M, D))

    @dtypes(torch.float16, torch.bfloat16)
    def test_gqa_expand_stride_zero_backward(self, device, dtype):
        """Test GQA backward with expand()-created K/V tensors (stride=0).

        Regression test for gradient buffer stride bug with expand().
        """
        batch_size = 1
        seqlen = 512
        headdim = 128
        n_heads = 4
        n_kv_heads = 1

        q, k_orig, v_orig = create_test_tensors(
            batch_size=batch_size,
            num_heads=n_heads,
            num_heads_kv=n_kv_heads,
            seq_len=seqlen,
            dim=headdim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        k = k_orig.expand(batch_size, n_heads, seqlen, headdim)
        v = v_orig.expand(batch_size, n_heads, seqlen, headdim)

        self.assertEqual(k.stride()[1], 0, "K should have stride=0 from expand()")
        self.assertEqual(v.stride()[1], 0, "V should have stride=0 from expand()")

        block_mask = _create_block_mask_for_device(
            _causal_mask, batch_size, n_heads, seqlen, seqlen, device=device
        )

        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_backend_raises_on_grad_logsumexp(self, device, dtype):
        from torch._dynamo.exc import BackendCompilerFailed

        q, k, v = create_test_tensors(dtype=dtype, device=device, requires_grad=True)
        lse_mask = torch.randn(2, 4, 512, device=device)

        compiled_flex = torch.compile(flex_attention)
        out, lse = compiled_flex(
            q, k, v, return_lse=True, kernel_options={"BACKEND": "FLASH"}
        )
        loss = out.mean() + (lse * lse_mask).sum()
        with self.assertRaisesRegex(
            BackendCompilerFailed,
            "FLASH backend backward does not support differentiating through logsumexp",
        ):
            loss.backward()


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
class TestFlexFlashDynamicShapes(InductorTestCase):
    """
    Dynamic-shape coverage for flex flash attention: score_mod captures and masks,
    plus backward, batch, and length variants.
    """

    def _run_dynamic_test(
        self, seq_lens, score_mod=None, block_mask_factory=None, requires_grad=False
    ):
        """Helper to run dynamic=True tests across multiple sequence lengths."""

        for seq_len in seq_lens:
            q, k, v = create_test_tensors(
                seq_len=seq_len,
                device="cuda",
                dtype=torch.float16,
                requires_grad=requires_grad,
            )
            block_mask = block_mask_factory(seq_len) if block_mask_factory else None
            flash_vs_triton(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                dynamic=True,
            )

    def _flash_triton_dynamic(self, q, k, v, **kwargs):
        flash_vs_triton(q, k, v, dynamic=True, **kwargs)

    def test_dynamic_seq_len_no_score_mod(self):
        """Test dynamic sequence lengths without score_mod."""
        self._run_dynamic_test(seq_lens=[128, 256, 512])

    def test_dynamic_seq_len_inline_literal(self):
        """Test dynamic sequence lengths with inline literal score_mod."""

        def score_mod(score, _b, _h, _q, _k):
            return score * 2.0  # Inline literal, not captured

        self._run_dynamic_test(seq_lens=[128, 256, 512], score_mod=score_mod)

    def test_dynamic_seq_len_captured_tensor_buffer(self):
        """Test dynamic sequence lengths with captured tensor buffer (ALiBi-style)."""
        num_heads = 4
        slopes = torch.exp2(
            -torch.linspace(1, 8, num_heads, device="cuda", dtype=torch.float16)
        )

        def alibi_score_mod(score, b, h, q_idx, kv_idx):
            return score + (kv_idx - q_idx) * slopes[h]

        self._run_dynamic_test(seq_lens=[128, 256, 512], score_mod=alibi_score_mod)

    def test_dynamic_seq_len_with_block_mask(self):
        """Test dynamic sequence lengths with block mask."""

        def block_mask_factory(seq_len):
            return _create_block_mask_for_device(
                _causal_mask, 2, 4, seq_len, seq_len, device="cuda"
            )

        self._run_dynamic_test(
            seq_lens=[128, 256, 512], block_mask_factory=block_mask_factory
        )

    def test_dynamic_batch_size(self):
        """Test dynamic batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            q, k, v = create_test_tensors(
                batch_size=batch_size, seq_len=256, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    @xfailIfSM90
    def test_dynamic_backward(self):
        """Test backward with dynamic sequence lengths."""
        self._run_dynamic_test(seq_lens=[128, 256, 512], requires_grad=True)

    @xfailIfSM90
    def test_dynamic_backward_with_score_mod(self):
        """Test backward with score_mod and dynamic sequence lengths."""

        def score_mod(score, _b, _h, _q, _k):
            return score * 2.0

        self._run_dynamic_test(
            seq_lens=[128, 256, 512], score_mod=score_mod, requires_grad=True
        )

    def test_dynamic_backward_with_block_mask(self):
        """Test backward with block mask and dynamic sequence lengths."""
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            self.skipTest("block sparse backward only supported on SM90+")

        def block_mask_factory(seq_len):
            return _create_block_mask_for_device(
                _causal_mask, 2, 4, seq_len, seq_len, device="cuda"
            )

        self._run_dynamic_test(
            seq_lens=[128, 256, 512],
            block_mask_factory=block_mask_factory,
            requires_grad=True,
        )

    def test_dynamic_gqa(self):
        """Test GQA with dynamic sequence lengths."""
        q_heads, kv_heads = 8, 2
        for seq_len in [128, 256, 512]:
            q = torch.randn(2, q_heads, seq_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            v = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v, score_mod=None, block_mask=None)

    def test_dynamic_mqa(self):
        """Test MQA with dynamic sequence lengths."""
        q_heads, kv_heads = 8, 1
        for seq_len in [128, 256, 512]:
            q = torch.randn(2, q_heads, seq_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            v = torch.randn(
                2, kv_heads, seq_len, 64, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    def test_dynamic_non_divisible_seq_len(self):
        """Test non-block-divisible sequence lengths with dynamic shapes."""
        for seq_len in [127, 255, 383, 511, 513]:
            q, k, v = create_test_tensors(
                seq_len=seq_len, device="cuda", dtype=torch.float16
            )
            self._flash_triton_dynamic(q, k, v)

    def test_dynamic_asymmetric_qkv_lengths(self):
        """Test asymmetric Q and KV lengths with dynamic shapes."""
        test_cases = [(256, 512), (512, 256), (128, 1024)]
        for q_len, kv_len in test_cases:
            q = torch.randn(2, 4, q_len, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(2, 4, kv_len, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(2, 4, kv_len, 64, device="cuda", dtype=torch.float16)
            self._flash_triton_dynamic(q, k, v)

    def test_captured_float_fails_with_dynamic(self):
        """Test that captured Python float fails with dynamic=True."""
        val = 2.0  # Captured float

        def score_mod(score, _b, _h, _q, _k):
            return score * val

        compiled_fn = torch.compile(flex_attention, dynamic=True)
        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(
            RuntimeError, r"captures a dynamic scalar \(SymInt/SymFloat\)"
        ):
            compiled_fn(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
            )

    def test_captured_int_fails_with_dynamic(self):
        """Captured Python int should fail with dynamic=True."""
        val = 2  # Captured int

        def score_mod(score, _b, _h, _q, _k):
            return score * val

        compiled_fn = torch.compile(flex_attention, dynamic=True)
        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        with self.assertRaisesRegex(
            RuntimeError, r"captures a dynamic scalar \(SymInt/SymFloat\)"
        ):
            compiled_fn(
                q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
            )

    def test_captured_float_works_with_static(self):
        """Test that captured Python float works with dynamic=False."""
        val = 2.0  # Captured float

        def score_mod(score, _b, _h, _q, _k):
            return score * val

        compiled_fn = torch.compile(flex_attention, dynamic=False)
        q, k, v = create_test_tensors(seq_len=256, device="cuda", dtype=torch.float16)

        out = compiled_fn(
            q, k, v, score_mod=score_mod, kernel_options={"BACKEND": "FLASH"}
        )
        self.assertEqual(out.shape, q.shape)

    def test_dynamic_mask_from_input_lengths_single_graph(self):
        """Dynamic mask creation driven by input lengths should stay single-graph."""
        counter = CompileCounterWithBackend("inductor")

        def _flex_attention_mask(b, h, q_idx, kv_idx, input_lengths):
            return (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])

        class Model(torch.nn.Module):
            def __init__(self, dim=128, heads=4):
                super().__init__()
                self.proj = torch.nn.Linear(dim, dim)
                self.heads: int = int(heads)

            def forward(self, x, input_lengths):
                x = self.proj(x)
                B, T, C = x.shape
                head_dim = C // self.heads
                x = x.view(B, T, self.heads, head_dim).permute(0, 2, 1, 3)

                max_time = x.size(-2)
                mask = torch.compile(create_block_mask, dynamic=True, fullgraph=False)(
                    functools.partial(
                        _flex_attention_mask, input_lengths=input_lengths
                    ),
                    B=B,
                    H=None,
                    Q_LEN=max_time,
                    KV_LEN=max_time,
                    device=x.device,
                )

                return torch.compile(
                    flex_attention, dynamic=True, fullgraph=True, backend=counter
                )(x, x, x, block_mask=mask, kernel_options={"BACKEND": "FLASH"})

        model = Model().cuda().half()
        B, T, F = 8, 64, 128
        for _ in range(3):
            x = torch.randn(B, T, F, device="cuda", dtype=torch.float16)
            lens = torch.randint(1, T + 1, (B,), device="cuda")
            model(x, lens)

        self.assertEqual(
            counter.frame_count, 1, f"Expected 1 graph, got {counter.frame_count}"
        )

    def test_dynamic_free_symbol_mask_single_graph(self):
        """Free-symbol dense mask under dynamic=True should not recompile."""
        counter = CompileCounterWithBackend("inductor")

        def make_mask(batch_shape, seq_len):
            rand_mask = torch.randint(
                0, 2, (batch_shape, seq_len), device="cuda"
            ).bool()
            return torch.compile(create_block_mask, dynamic=True)(
                B=batch_shape,
                BLOCK_SIZE=128,
                mask_mod=lambda b, h, q_idx, kv_idx: ~rand_mask[b, q_idx],
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device="cuda",
            )

        @torch.compile(dynamic=True, fullgraph=True, backend=counter)
        def run(q, k, v, block_mask):
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
            )

        seq_len = 128
        for batch_shape in [2, 4, 8]:
            q, k, v = create_test_tensors(
                batch_shape, 4, seq_len, 64, device="cuda", dtype=torch.float16
            )
            block_mask = make_mask(batch_shape, seq_len)
            run(q, k, v, block_mask)

        self.assertEqual(
            counter.frame_count, 1, f"Expected 1 graph, got {counter.frame_count}"
        )

    def test_dynamic_max_autotune_with_block_mask(self):
        """Dynamic=True with max-autotune should succeed for FLASH backend."""
        q, k, v = create_test_tensors(
            batch_size=2,
            num_heads=4,
            seq_len=256,
            dim=64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        block_mask = _create_block_mask_for_device(
            _causal_mask, 2, 4, 256, 256, device="cuda"
        )

        compiled_fn = torch.compile(
            flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
        )
        out = compiled_fn(
            q,
            k,
            v,
            block_mask=block_mask,
            kernel_options={"BACKEND": "FLASH"},
        )
        self.assertEqual(out.shape, q.shape)

    @xfailIfSM90
    def test_dynamic_captured_buffer_varying_heads(self):
        """Dynamic head_count with captured tensor buffer under FLASH/TRITON parity."""
        torch._dynamo.reset()

        def run_with_head_count(head_count):
            head_scale = torch.randn(
                head_count, device="cuda", dtype=torch.float16, requires_grad=False
            )

            def score_mod(score, batch, head, token_q, token_kv):
                return score * head_scale[head]

            q, k, v = create_test_tensors(
                batch_size=2,
                num_heads=head_count,
                seq_len=256,
                dim=64,
                dtype=torch.float16,
                device="cuda",
                requires_grad=True,
            )
            self._flash_triton_dynamic(q, k, v, score_mod=score_mod)

        for head_count in [4, 8, 4, 16, 4]:
            run_with_head_count(head_count)

    def test_dynamic_symbol_closure_in_score_mod(self):
        """Capturing a SymInt in score_mod should compile to one dynamic graph."""

        class SimpleAttention(torch.nn.Module):
            def __init__(self, dim=512, n_head=8):
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head: int = int(n_head)
                self.head_dim: int = int(dim // n_head)

            def forward(self, x):
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                return flex_attention(
                    q,
                    k,
                    v,
                    score_mod=lambda s, b, h, q_idx, kv_idx: s + B,
                    kernel_options={"BACKEND": "FLASH"},
                )

        model = SimpleAttention().cuda()
        backend = EagerAndRecordGraphs()
        model.compile(mode="default", dynamic=True, backend=backend)

        torch._dynamo.reset()
        for batch_shape in [2, 4, 8]:
            x = torch.randn(batch_shape, 256, 512, device="cuda")
            model(x)

        self.assertEqual(len(backend.graphs), 1, "Expected a single dynamic graph")


class TestHierarchicalIndex(InductorTestCase):
    def test_hierarchical_index_preserves_args(self):
        from sympy import Symbol

        b = Symbol("b")
        q_idx = Symbol("q_idx")
        idx = HierarchicalIndex(b, q_idx)

        self.assertIsInstance(idx, HierarchicalIndex)
        self.assertEqual(idx.args, (b, q_idx))

    def test_hierarchical_indexer_single_dim_no_wrap(self):
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[10])
        q_idx = Symbol("q_idx")

        self.assertEqual(indexer([q_idx]), q_idx)

    def test_hierarchical_indexer_multi_dim_wraps(self):
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[4, 128])
        b = Symbol("b")
        q_idx = Symbol("q_idx")

        result = indexer([b, q_idx])

        self.assertIsInstance(result, HierarchicalIndex)
        self.assertEqual(result.args, (b, q_idx))

    def test_hierarchical_indexer_3d_and_4d(self):
        from sympy import Symbol

        b, h, q_idx, kv_idx = (
            Symbol("b"),
            Symbol("h"),
            Symbol("q_idx"),
            Symbol("kv_idx"),
        )

        indexer_3d = _hierarchical_indexer_cute(size=[2, 4, 512])
        result_3d = indexer_3d([b, h, q_idx])
        self.assertIsInstance(result_3d, HierarchicalIndex)
        self.assertEqual(result_3d.args, (b, h, q_idx))

        indexer_4d = _hierarchical_indexer_cute(size=[2, 4, 512, 512])
        result_4d = indexer_4d([b, h, q_idx, kv_idx])
        self.assertIsInstance(result_4d, HierarchicalIndex)
        self.assertEqual(result_4d.args, (b, h, q_idx, kv_idx))

    def test_isinstance_detection_for_load(self):
        from sympy import Symbol

        b = Symbol("b")
        q_idx = Symbol("q_idx")

        self.assertIsInstance(HierarchicalIndex(b, q_idx), HierarchicalIndex)
        self.assertNotIsInstance(b * Symbol("S") + q_idx, HierarchicalIndex)

    def test_hierarchical_indexer_rank_mismatch(self):
        from sympy import Symbol

        indexer = _hierarchical_indexer_cute(size=[2, 4])
        b = Symbol("b")

        with self.assertRaises(AssertionError) as ctx:
            indexer([b])
        self.assertIn("Rank mismatch", str(ctx.exception))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_2d(self):
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_2d = torch.randn(batch_size, num_heads, device=device, dtype=dtype)

        def score_mod_2d(score, b, h, q_idx, kv_idx):
            return score + bias_2d[b, h]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_2d)

        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=score_mod_2d,
            kernel_options={"BACKEND": "FLASH"},
        )
        code_str = "\n".join(code)

        expected_pattern = "in_ptr4[tmp3, tmp4]"
        self.assertIn(
            expected_pattern,
            code_str,
            f"Expected '{expected_pattern}' in generated code.\nExcerpt:\n{code_str[:2000]}",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_3d(self):
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_3d = torch.randn(
            batch_size, num_heads, seq_len, device=device, dtype=dtype
        )

        def score_mod_3d(score, b, h, q_idx, kv_idx):
            return score + bias_3d[b, h, q_idx]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_3d)

        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=score_mod_3d,
            kernel_options={"BACKEND": "FLASH"},
        )
        code_str = "\n".join(code)

        expected_pattern = "in_ptr4[tmp4, tmp5, tmp6]"
        self.assertIn(
            expected_pattern,
            code_str,
            f"Expected '{expected_pattern}' in generated code.\nExcerpt:\n{code_str[:2000]}",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(
        not ensure_flash_available(), "Flash attention (CUTE) library not available"
    )
    def test_hierarchical_indexing_4d(self):
        batch_size, num_heads, seq_len, dim = 2, 4, 512, 64
        dtype = torch.float16
        device = "cuda"

        bias_4d = torch.randn(
            batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype
        )

        def score_mod_4d(score, b, h, q_idx, kv_idx):
            return score + bias_4d[b, h, q_idx, kv_idx]

        q, k, v = create_test_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        flash_vs_triton(q, k, v, score_mod=score_mod_4d)

        compiled_fn = torch.compile(flex_attention)
        _, code = run_and_get_code(
            compiled_fn,
            q,
            k,
            v,
            score_mod=score_mod_4d,
            kernel_options={"BACKEND": "FLASH"},
        )
        code_str = "\n".join(code)

        expected_pattern = "in_ptr4[tmp5, tmp6, tmp7, tmp8]"
        self.assertIn(
            expected_pattern,
            code_str,
            f"Expected '{expected_pattern}' in generated code.\nExcerpt:\n{code_str[:2000]}",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
