# Owner(s): ["module: inductor"]

import unittest
import warnings

import torch
import torch._inductor.compile_fx as inductor_compile_fx
import torch._inductor.fx_passes.fuse_attention as fuse_attention
from torch.testing._internal.common_utils import run_tests, TestCase


def _has_cuda_sm80() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (8, 0)
    )


class InductorWarningTests(TestCase):
    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    def test_trivial_matmul_compile_no_user_warning(self):
        orig_cuda_precision = torch.backends.cuda.matmul.fp32_precision
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            inductor_compile_fx._warn_tf32_disabled.cache_clear()
            torch._dynamo.reset()

            x = torch.eye(2, device="cuda")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                actual = torch.compile(
                    lambda y: y @ y, backend="inductor", fullgraph=True
                )(x)
                torch.cuda.synchronize()

            self.assertEqual(actual, x)
            self.assertEqual([str(w.message) for w in caught], [])
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)
            torch.backends.cuda.matmul.fp32_precision = orig_cuda_precision
            torch._dynamo.reset()

    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    def test_fuse_attention_tf32_advisory_no_user_warning(self):
        orig_cuda_precision = torch.backends.cuda.matmul.fp32_precision
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            fuse_attention._warn_tf32_disabled.cache_clear()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                fuse_attention._warn_tf32_disabled()

            self.assertEqual([str(w.message) for w in caught], [])
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)
            torch.backends.cuda.matmul.fp32_precision = orig_cuda_precision


if __name__ == "__main__":
    run_tests()
