# Owner(s): ["module: inductor"]

import unittest
import warnings

import torch
import torch._inductor.compile_fx as inductor_compile_fx
import torch._inductor.fx_passes.fuse_attention as fuse_attention
from torch._inductor.utils import run_and_get_code
from torch._logging import warning_once
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_cuda import BF16X9_SUPPORTED
from torch.testing._internal.common_utils import (
    recover_orig_fp32_precision,
    run_tests,
    TestCase,
)
from torch.testing._internal.logging_utils import logs_to_string


TF32_ADVISORY = "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled."


def _has_cuda_sm80() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (8, 0)
    )


class InductorWarningTests(TestCase):
    @unittest.skipUnless(
        BF16X9_SUPPORTED, "requires CUDA 12.9+ and compute capability 10.0 or 10.3"
    )
    @recover_orig_fp32_precision
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_flex_attention_warns_and_uses_ieee_for_bfx9(self):
        def fn(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                kernel_options={
                    "BACKEND": "TRITON",
                    "FLOAT32_PRECISION": "'tf32'",
                },
            )

        torch.backends.cuda.matmul.fp32_precision = "bfx9"
        warning_once.cache_clear()
        self.addCleanup(warning_once.cache_clear)
        q = torch.randn(1, 1, 128, 64, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with (
            self.assertLogs(
                "torch._inductor.kernel.flex.flex_attention", level="WARNING"
            ) as logs,
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings("ignore", message="dynamo_pgo force disabled by.*")
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), q, k, v)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(len(logs.output), 1)
        self.assertIn("using IEEE precision instead", logs.output[0])
        source = "\n".join(code)
        self.assertIn("FLOAT32_PRECISION : tl.constexpr = 'ieee'", source)
        self.assertNotIn("FLOAT32_PRECISION : tl.constexpr = 'tf32'", source)

    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    @recover_orig_fp32_precision
    def test_trivial_matmul_compile_no_user_warning(self):
        # recover_orig_fp32_precision restores the per-backend flags; the
        # legacy enum still needs the set_float32_matmul_precision below.
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            inductor_compile_fx._warn_tf32_disabled.cache_clear()
            torch._dynamo.reset()

            x = torch.eye(2, device="cuda")
            log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "perf_hints")
            with ctx(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                actual = torch.compile(
                    lambda y: y @ y, backend="inductor", fullgraph=True
                )(x)
                torch.cuda.synchronize()

            self.assertEqual(actual, x)
            self.assertEqual([str(w.message) for w in caught], [])
            self.assertIn(TF32_ADVISORY, log_stream.getvalue())
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)
            torch._dynamo.reset()

    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    @recover_orig_fp32_precision
    def test_fuse_attention_tf32_advisory_no_user_warning(self):
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            fuse_attention._warn_tf32_disabled.cache_clear()

            log_stream, ctx = logs_to_string(
                "torch._inductor.fx_passes.fuse_attention", "perf_hints"
            )
            with ctx(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                fuse_attention._warn_tf32_disabled()

            self.assertEqual([str(w.message) for w in caught], [])
            self.assertIn(TF32_ADVISORY, log_stream.getvalue())
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)


if __name__ == "__main__":
    run_tests()
