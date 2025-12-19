# Owner(s): ["module: inductor"]

import unittest
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from setuptools.config._apply_pyprojecttoml import _identity

import torch
from torch._inductor.kernel.flex.flex_flash_attention import ensure_flash_available
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    flex_attention,
)
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize


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
):
    shape = (batch_size, num_heads, seq_len, dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
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


def flash_vs_triton(q, k, v, score_mod=None, block_mask=None, rtol=2):
    compiled_fn = torch.compile(flex_attention)

    out_ref_fp32 = flex_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        score_mod=score_mod,
        block_mask=block_mask,
    ).to(q.dtype)

    out_flash = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        kernel_options={"BACKEND": "FLASH"},
    )
    out_triton = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        kernel_options={"BACKEND": "TRITON"},
    )

    assert out_flash.shape == out_ref_fp32.shape == out_triton.shape
    assert not torch.isnan(out_flash).any()
    assert not torch.isnan(out_triton).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_flash).all()
    assert torch.isfinite(out_triton).all()
    assert torch.isfinite(out_ref_fp32).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()

    triton_error = (out_triton - out_ref_fp32).abs().max().item()
    flash_error = (out_flash - out_ref_fp32).abs().max().item()

    assert flash_error <= rtol * triton_error + fwd_atol, (
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
            assert torch.isfinite(grad_flash).all()
            assert torch.isfinite(grad_triton).all()
            assert torch.isfinite(grad_ref).all()

            triton_error = (grad_triton - grad_ref).abs().max().item()
            flash_error = (
                (grad_flash - grad_ref.to(grad_flash.dtype)).abs().max().item()
            )
            assert flash_error <= rtol * triton_error + atol, (
                f"Flash error {flash_error:.2e} exceeds {rtol}x Triton error {triton_error:.2e} + {atol:.2e}"
            )

    return out_flash, out_triton, out_ref_fp32


@dataclass
class ScoreModCase:
    name: str
    score_mod_factory: Callable[[torch.dtype, str], Callable | None]
    batch_size: int = 2
    num_heads: int = 4
    seq_len: int = 512
    dim: int = 64
    requires_grad: bool = False


@dataclass
class MaskModCase:
    name: str
    mask_mod_factory: Callable[[torch.dtype, str], Callable]
    batch_size: int = 2
    num_heads: int = 4
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
]


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


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
class TestFlexFlash(InductorTestCase):
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("case", SCORE_MOD_CASES, name_fn=score_case_name)
    def test_flash_attention_score_mod_cases(self, device, dtype, case):
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
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
    @parametrize("case", MASK_MOD_CASES, name_fn=mask_case_name)
    def test_flash_attention_mask_mod_cases(self, device, dtype, case):
        if case.requires_grad:
            major, _ = torch.cuda.get_device_capability()
            if major != 10:
                self.skipTest("block sparse only supported on blackwell for now")

        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
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

    @dtypes(torch.float16, torch.bfloat16)
    def test_mixed_dtypes(self, device, dtype):
        """Ensure flash attention rejects mixed dtypes (e.g., fp32 Q with fp16 K/V)"""
        B, H, S, D = 2, 8, 512, 64

        query = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        key = torch.randn(B, H, S, D, dtype=dtype, device=device).to(
            torch.float8_e4m3fn
        )
        value = torch.randn(B, H, S, D, dtype=dtype, device=device).to(
            torch.float8_e4m3fn
        )

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        from torch._inductor.exc import InductorError

        with self.assertRaisesRegex(
            InductorError,
            "Mixed query, key, and value dtype is not supported on this platform",
        ):
            compiled_fn(
                query, key, value, _identity, kernel_options={"BACKEND": "FLASH"}
            )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_backward_rejects_mask_mod_on_unsupported_gpu(
        self, device, dtype
    ):
        major, _ = torch.cuda.get_device_capability()
        if major == 10:
            self.skipTest("Block sparsity backward is supported on SM100")
        q, k, v = create_test_tensors(dtype=dtype, device=device)

        def causal_mask(_b, _h, q_idx, kv_idx):
            return q_idx >= kv_idx

        q.requires_grad_(True)
        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"NYI: Block sparsity in backward only supported on SM100",
        ):
            compiled_fn(
                q,
                k,
                v,
                block_mask=_create_block_mask_for_device(
                    causal_mask, 2, 4, 512, 512, device=device
                ),
                kernel_options={"BACKEND": "FLASH"},
            ).sum().backward()

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

        flash_vs_triton(q, k, v)

        compiled_fn = torch.compile(flex_attention)

        def run_for_profile():
            q_run, k_run, v_run = (
                t.detach().clone().requires_grad_(True) for t in (q, k, v)
            )
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


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
