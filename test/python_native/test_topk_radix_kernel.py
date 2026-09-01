# Owner(s): ["module: dsl-native-ops"]
"""Tests for the topk radix kernel's compile parameters and AOT builder.

The kernel body and its AOT ``build(spec)`` entry point live in
``cutedsl_kernels.py`` alongside the JIT wrappers. These tests check the builder
contract and the dtype/index-width compile parameters against stock aten.
"""

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


class TestRadixKernelBuilder(TestCase):
    @skipIfNoCuteDSL
    def test_build_contract(self):
        from torch._native.ops.topk.cutedsl_kernels import build

        b = build({"dtype": "float32", "N": 4096, "K": 64, "deterministic": False})
        self.assertEqual(b["prefix"], "topk_radix_f32_n4096_k64_nondet")
        self.assertEqual(len(b["fake_args"]), 4)
        self.assertEqual(
            [t["name"] for t in b["tensor_args"]], ["mX", "mValues", "mIndices"]
        )

    @skipIfNoCuteDSL
    def test_build_prefixes_unique_across_grid(self):
        from torch._native.ops.topk.cutedsl_kernels import build

        prefixes = set()
        for dtype in ("float32", "bfloat16"):
            for det in (False, True):
                b = build({"dtype": dtype, "N": 4096, "K": 64, "deterministic": det})
                prefixes.add(b["prefix"])
        self.assertEqual(len(prefixes), 4)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestRadixKernelParams(TestCase):
    """Numerics of the in_dtype / index_dtype compile parameters, checked
    against stock aten (python_native disabled for the reference)."""

    def _compile_and_run(self, x, k, in_dtype, index_dtype, index_torch_dtype):
        import math

        import cutlass.cute as cute

        from torch._native.ops.topk.cutedsl_kernels import (
            _make_fake_tensor,
            _RadixSelectTopK,
        )

        M, N = x.shape
        batch = cute.sym_int()
        x_f = _make_fake_tensor(in_dtype, (batch, N), math.gcd(4, N))
        v_f = _make_fake_tensor(in_dtype, (batch, k), math.gcd(4, k))
        i_f = _make_fake_tensor(index_dtype, (batch, k), math.gcd(4, k))
        compiled = cute.compile(
            _RadixSelectTopK(
                N, k, deterministic=True, in_dtype=in_dtype, index_dtype=index_dtype
            ),
            x_f,
            v_f,
            i_f,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        values = torch.empty(M, k, dtype=x.dtype, device=x.device)
        indices = torch.empty(M, k, dtype=index_torch_dtype, device=x.device)
        compiled(x, values, indices)
        return values, indices

    def _reference(self, x, k):
        pn = torch.backends.python_native
        with pn.cutedsl.disabled():
            return torch.topk(x, k, dim=-1)

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_int64_indices_match_aten(self, dtype):
        from cutlass import BFloat16, Float32, Int64

        torch.manual_seed(0)
        x = torch.randn(512, 4096, device="cuda", dtype=dtype)
        ref_v, ref_i = self._reference(x, 64)
        in_dtype = Float32 if dtype == torch.float32 else BFloat16
        v, i = self._compile_and_run(x, 64, in_dtype, Int64, torch.int64)
        self.assertEqual(v, ref_v, atol=0, rtol=0)
        self.assertEqual(i, ref_i, atol=0, rtol=0)

    def test_int32_indices_match_aten_fp32(self):
        from cutlass import Float32, Int32

        torch.manual_seed(1)
        x = torch.randn(512, 4096, device="cuda")
        ref_v, ref_i = self._reference(x, 64)
        v, i = self._compile_and_run(x, 64, Float32, Int32, torch.int32)
        self.assertEqual(v, ref_v, atol=0, rtol=0)
        self.assertEqual(i.long(), ref_i, atol=0, rtol=0)

    def test_bf16_heavy_ties_deterministic(self):
        from cutlass import BFloat16, Int64

        torch.manual_seed(2)
        x = torch.randint(0, 4, (512, 4096), device="cuda").bfloat16()
        ref_v, ref_i = self._reference(x, 128)
        v, i = self._compile_and_run(x, 128, BFloat16, Int64, torch.int64)
        self.assertEqual(v, ref_v, atol=0, rtol=0)
        self.assertEqual(i, ref_i, atol=0, rtol=0)


instantiate_parametrized_tests(TestRadixKernelParams)


if __name__ == "__main__":
    run_tests()
