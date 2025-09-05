# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.nn.attention.flex_attention import flex_attention
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


def flash_vs_triton(q, k, v, score_mod=None, rtol=5e-3, atol=5e-3):
    compiled_fn = torch.compile(flex_attention)
    out_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"disable_flash": False}
    )
    out_no_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"disable_flash": True}
    )
    torch.testing.assert_close(out_flash, out_no_flash, rtol=rtol, atol=atol)
    return out_flash, out_no_flash


def name_fn(score_mod):
    return score_mod.__name__.lstrip("_")


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
    def test_flash_attention_different_seq_lens(self, device, dtype):
        for seq_len in [128, 256, 1024, 2048]:
            q, k, v = create_test_tensors(seq_len=seq_len, dtype=dtype, device=device)
            compiled_fn = torch.compile(flex_attention)
            out = compiled_fn(q, k, v, kernel_options={"disable_flash": False})
            self.assertEqual(out.shape, q.shape)

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


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
