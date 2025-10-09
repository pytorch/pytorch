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


def flash_vs_triton(q, k, v, score_mod=None, rtol=5e-3, atol=5e-3):
    compiled_fn = torch.compile(flex_attention)
    out_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"force_flash": True}
    )
    out_no_flash = compiled_fn(
        q, k, v, score_mod=score_mod, kernel_options={"force_flash": False}
    )
    torch.testing.assert_close(out_flash, out_no_flash, rtol=rtol, atol=atol)
    return out_flash, out_no_flash


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
    def test_force_flash_error_with_block_mask(self, device, dtype):
        """Test that force_flash=True raises error when BlockMask is provided."""
        q, k, v = create_test_tensors(dtype=dtype, device=device)

        # Create a causal block mask
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 2, 4, 512, 512, device=device)

        compiled_fn = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError,
            r"force_flash=True but flash attention cannot be used.*BlockMask.*not supported",
        ):
            compiled_fn(
                q, k, v, block_mask=block_mask, kernel_options={"force_flash": True}
            )

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


instantiate_device_type_tests(TestFlexFlash, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
