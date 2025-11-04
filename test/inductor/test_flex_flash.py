# Owner(s): ["module: inductor"]

import unittest
from contextlib import contextmanager

import torch
from torch._inductor.kernel.flex.flex_flash_attention import ensure_flash_available
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
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


def _rel_bias(score, _b, _h, token_q, token_kv):
    return score + (token_q - token_kv)


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


def create_test_tensors(
    batch_size=2, num_heads=4, seq_len=512, dim=64, dtype=torch.float16, device="cuda"
):
    shape = (batch_size, num_heads, seq_len, dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    return q, k, v


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
        kernel_options={"force_flash": True},
    )
    out_triton = compiled_fn(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=block_mask,
        kernel_options={"force_flash": False},
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

    return out_flash, out_triton, out_ref_fp32


def name_fn(score_mod):
    return score_mod.__name__.lstrip("_")


@unittest.skipIf(
    not ensure_flash_available(), "Flash attention (CUTE) library is not available"
)
class TestFlexFlash(InductorTestCase):
    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_basic(self, device, dtype):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        flash_vs_triton(q, k, v)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("score_mod", [_times_two, _causal, _rel_bias], name_fn=name_fn)
    def test_flash_attention_with_score_mod(self, device, dtype, score_mod):
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("seq_len", [127, 255, 383, 511])
    def test_flash_attention_unfriendly_seqlen_with_causal(
        self, device, dtype, seq_len
    ):
        """Test flash attention with unfriendly sequence lengths and causal masking."""
        q, k, v = create_test_tensors(seq_len=seq_len, dtype=dtype, device=device)
        flash_vs_triton(q, k, v, score_mod=_causal)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_kernel_called(self, device, dtype):
        """Test that flash attention kernel is actually called when force_flash=True."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        compiled_fn = torch.compile(flex_attention)

        # Test that flash kernel is called with force_flash=True
        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(
                q, k, v, score_mod=_causal, kernel_options={"force_flash": True}
            )

        self.assertTrue(
            prof_result["found"],
            f"Flash attention kernel not found. Available kernels: {prof_result['kernel_names']}",
        )

        # Test that flash kernel is NOT called with force_flash=False
        with cuda_kernel_profiler("flash_attncute") as prof_result:
            compiled_fn(
                q, k, v, score_mod=_causal, kernel_options={"force_flash": False}
            )

        self.assertFalse(
            prof_result["found"],
            f"Flash attention kernel unexpectedly found when force_flash=False. Kernels: {prof_result['kernel_names']}",
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_alibi_learned(self, device, dtype):
        """Test flash attention with ALiBi learned slopes (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_alibi_learned(num_heads=4, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_pos_bias_table(self, device, dtype):
        """Test flash attention with position bias table (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_pos_bias_table(seq_len=512, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_head_scale(self, device, dtype):
        """Test flash attention with head scaling (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_head_scale(num_heads=4, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_batch_bias(self, device, dtype):
        """Test flash attention with batch bias (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_batch_bias(batch_size=2, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_batch_head_bias(self, device, dtype):
        """Test flash attention with batch-head bias matrix (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_batch_head_bias(batch_size=2, num_heads=4, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_dual_buffer_bias(self, device, dtype):
        """Test flash attention with dual buffer loading (tensor loading)."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)
        score_mod = create_dual_buffer_bias(num_heads=4, seq_len=512, dtype=dtype)
        flash_vs_triton(q, k, v, score_mod=score_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_score_view_buffer(self, device, dtype):
        """Score modifier should load from a non-contiguous view."""
        num_heads = 4
        q, k, v = create_test_tensors(num_heads=num_heads, dtype=dtype, device=device)

        base_scales = torch.rand(num_heads, 2, device=device, dtype=dtype) + 0.5
        scales_view = base_scales[:, 0]
        assert not scales_view.is_contiguous()

        def score_view_mod(score, b, h, q_idx, kv_idx):
            return score + scales_view[h]

        flash_vs_triton(q, k, v, score_mod=score_view_mod)

    @dtypes(torch.float16, torch.bfloat16)
    def test_force_flash_error_with_requires_grad(self, device, dtype):
        """Test that force_flash=True raises error when tensor requires gradients."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)

        bias = torch.randn(4, device=device, dtype=dtype, requires_grad=True)

        def score_mod_with_grad(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"force_flash=True but flash attention cannot be used.*require gradients",
        ):
            compiled_fn(
                q,
                k,
                v,
                score_mod=score_mod_with_grad,
                kernel_options={"force_flash": True},
            )

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_block_mask(self, device, dtype):
        """Test flash attention with block mask and mask_mod."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 2, 4, 512, 512, device=device)
        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_block_mask_with_score_mod(self, device, dtype):
        """Test flash attention with both block mask and score_mod."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 2, 4, 512, 512, device=device)
        flash_vs_triton(q, k, v, score_mod=_times_two, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_mask_mod_buffer(self, device, dtype):
        """Test flash attention with mask_mod that loads from buffer."""
        q, k, v = create_test_tensors(
            batch_size=2, num_heads=4, dtype=dtype, device=device
        )

        mask_bias = torch.randn(4, device=device, dtype=dtype) * 0.1

        def custom_mask(b, h, q_idx, kv_idx):
            bias_value = mask_bias[h]
            return (q_idx >= kv_idx) | (bias_value > 0)

        block_mask = create_block_mask(custom_mask, 2, 4, 512, 512, device=device)
        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_doc_mask(self, device, dtype):
        """Test flash attention with a document-aware mask_mod."""
        # Use shorter sequences to make the document layout explicit.
        seq_len = 128
        q, k, v = create_test_tensors(
            batch_size=2, num_heads=4, seq_len=seq_len, dtype=dtype, device=device
        )
        lengths_per_batch = (
            (16, 31, 25, 56),  # batch 0
            (40, 9, 23, 56),  # batch 1 uses a different document arrangement
        )
        document_ids = []
        for lengths in lengths_per_batch:
            assert sum(lengths) == seq_len
            doc_tokens = []
            for doc_id, length in enumerate(lengths):
                doc_tokens.extend([doc_id] * length)
            document_ids.append(doc_tokens)
        document_ids = torch.tensor(document_ids, device=device, dtype=torch.long)

        def document_mask(b, _h, q_idx, kv_idx):
            doc_id_q = document_ids[b, q_idx // 2]
            doc_id_kv = document_ids[b, kv_idx]
            return doc_id_q == doc_id_kv

        block_mask = create_block_mask(
            document_mask, 2, 1, seq_len, seq_len, device=device
        )
        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_mask_mod_with_view_buffer(self, device, dtype):
        """Mask modifier should support buffers that are non-contiguous views."""
        batch_size, num_heads, seq_len = 2, 4, 512
        q, k, v = create_test_tensors(
            batch_size=batch_size, num_heads=num_heads, dtype=dtype, device=device
        )

        base_bias = torch.randn(num_heads, 3, device=device, dtype=dtype)
        mask_bias_view = base_bias[:, 1]
        assert not mask_bias_view.is_contiguous()

        def mask_with_view_buffer(b, h, q_idx, kv_idx):
            bias_value = mask_bias_view[h]
            double_bias = bias_value * 2
            return (q_idx >= kv_idx) | (double_bias > 0)

        block_mask = create_block_mask(
            mask_with_view_buffer,
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=device,
        )
        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_mask_mod_with_dual_buffers(self, device, dtype):
        """Mask modifier should support multiple captured buffers."""
        batch_size, num_heads, seq_len = 2, 4, 512
        q, k, v = create_test_tensors(
            batch_size=batch_size, num_heads=num_heads, dtype=dtype, device=device
        )

        head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2
        batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.2

        def dual_buffer_mask(b, h, q_idx, kv_idx):
            head_term = head_bias[h]
            batch_term = batch_bias[b]
            causal = q_idx >= kv_idx
            bias_cond = (head_term + batch_term).to(torch.float32) > 0
            return causal | bias_cond

        block_mask = create_block_mask(
            dual_buffer_mask, batch_size, num_heads, seq_len, seq_len, device=device
        )
        flash_vs_triton(q, k, v, block_mask=block_mask)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_score_mod_with_many_buffer_indexing(self, device, dtype):
        batch_size, num_heads, seq_len = 2, 4, 512
        q, k, v = create_test_tensors(
            batch_size=batch_size, num_heads=num_heads, dtype=dtype, device=device
        )

        head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.15
        query_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
        kv_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
        batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.1

        def complex_score(score, b, h, q_idx, kv_idx):
            head_term = head_bias[h]
            query_term = query_scale[q_idx]
            kv_term = kv_scale[kv_idx]
            batch_term = batch_bias[b]
            return score + head_term + query_term - kv_term + batch_term

        flash_vs_triton(q, k, v, score_mod=complex_score)

    @dtypes(torch.float16, torch.bfloat16)
    def test_flash_attention_with_score_and_mask_buffers(self, device, dtype):
        """Test flash attention with both score_mod and mask_mod using buffers."""
        q, k, v = create_test_tensors(
            batch_size=2, num_heads=4, dtype=dtype, device=device
        )

        score_bias = torch.randn(4, device=device, dtype=dtype) * 0.2
        mask_bias = torch.randn(4, device=device, dtype=dtype) * 0.1

        def score_with_buffer(score, b, h, q_idx, kv_idx):
            return score + score_bias[h]

        def mask_with_buffer(b, h, q_idx, kv_idx):
            bias_value = mask_bias[h]
            return (q_idx >= kv_idx) | (bias_value > 0)

        block_mask = create_block_mask(mask_with_buffer, 2, 4, 512, 512, device=device)
        flash_vs_triton(q, k, v, score_mod=score_with_buffer, block_mask=block_mask)


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
