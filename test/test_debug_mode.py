# Owner(s): ["module: nn"]

import contextlib
import unittest

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_nn import TestCase
from torch.testing._internal.common_utils import NOTEST_CPU, parametrize, run_tests


# Context manager to set env variable for debug mode
@contextlib.contextmanager
def debug_mode(enable_debug_mode: bool):
    r"""
    This context manager can be used to temporarily enable or disable debug mode
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch._C.get_runtime_debug()
    try:
        torch._C.set_runtime_debug(enable_debug_mode)
        yield {}
    finally:
        torch._C.set_runtime_debug(previous_mode)


class TestDebugMode(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_embedding(self, device):
        e = nn.Embedding(10, 10, device=device)
        x = torch.tensor([[10]], device=device)
        error_type = IndexError if device == "cpu" else RuntimeError
        error_message = (
            "index out of range in self"
            if device == "cpu"
            else "CUDA error: device-side assert triggered"
        )
        if device == "cpu":
            # TODO Catching that cuda assert is hard!
            with self.assertRaisesRegex(error_type, error_message):
                e(x)
        with debug_mode(True):
            debug_error_message = (
                r"IndexError embedding index out of bounds, "
                r"value: 10 at index 0, is out of bounds of size : \[10\]"
            )
            with self.assertRaisesRegex(RuntimeError, debug_error_message):
                e(x)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Memory efficient attention is not supported on this platform.",
    )
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_sdpa(self, device, dtype):
        shape = (1, 1, 16, 32)
        q = torch.rand(shape, dtype=dtype, device=device)
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        mask = torch.rand(1, 1, 16, 16, dtype=dtype, device=device)

        mask[0, 0, 3, :] = torch.finfo(dtype).min
        mask[0, 0, 8, :] = torch.finfo(dtype).min

        error_message = (
            r"Attn Mask contains a row that is completely masked. "
            r"This will cause NaNs in the output. Masked out row indexes: \[3, 8\]."
        )
        # Test no error message is raised when debug mode is disabled
        with debug_mode(False):
            scaled_dot_product_attention(q, k, v, attn_mask=mask)

        with debug_mode(True):
            with torch.backends.cuda.sdp_kernel(
                enable_mem_efficient=True, enable_flash=False, enable_math=False
            ):
                with self.assertRaisesRegex(RuntimeError, error_message):
                    scaled_dot_product_attention(q, k, v, attn_mask=mask)

            with torch.backends.cuda.sdp_kernel(
                enable_math=True, enable_mem_efficient=False, enable_flash=False
            ):
                # Math path should not error with finfo.min
                scaled_dot_product_attention(q, k, v, attn_mask=mask)
                # Math path should error with -inf
                mask[0, 0, 3, :] = -torch.inf
                mask[0, 0, 8, :] = -torch.inf
                with self.assertRaisesRegex(RuntimeError, error_message):
                    scaled_dot_product_attention(q, k, v, attn_mask=mask)


if NOTEST_CPU:
    device_types = ("cuda",)
else:
    device_types = ("cpu", "cuda")

instantiate_device_type_tests(TestDebugMode, globals(), only_for=device_types)


if __name__ == "__main__":
    run_tests()
