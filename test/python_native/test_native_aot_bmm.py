# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for the native-AOT bmm outer-product kernels @ CUDA.

First Triton-kind AOT op: kernels are compiled by triton.tools.compile
into flat-signature C entry points (cubin embedded, grid baked in) and
dispatched by per-spec range guards over (M, N) -- the block-size
buckets the JIT wrapper picks dynamically. The narrow starter grid is
fp32/bf16 x BLOCK_M {32 (M in (32,96]), 64 (M in (96,192])} with
N >= 128; everything else (fp16, other buckets, small N, non-CUDA
accelerators) stays with the JIT override, which shares the same
@triton.jit kernel body from aot_kernel.py.

bmm outer-product is a pure per-element product (K == 1): no
accumulation order, so no det-mode carve-out, and results should be
exact vs the reference for matching dtypes.

Tests that require the AOT kernel library skip unless it was loaded at
import; correctness tests run everywhere.
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


def _aot_lib_loaded() -> bool:
    from torch._native import _native_aot_embedded

    return _native_aot_embedded()


def skipIfNoAotLib(fn):
    return unittest.skipUnless(
        _aot_lib_loaded(), "AOT kernels not embedded in this build"
    )(fn)


def _ran_aot(fn) -> bool:
    # The AOT entry points embed the kernel under its jit name; the JIT
    # path launches the same-named kernel, so profiler name alone cannot
    # distinguish the layers -- tests that need layer attribution mask
    # the JIT side via the registry-disabling context instead.
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return any(
        "_bmm_outer_product_aot_kernel" in e.name
        for e in prof.events()
        if e.device_type.name == "CUDA"
    )


def _load_covered_axes():
    # The declaration module is stdlib-only and loaded by file path (it
    # is not an importable member of the torch package).
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(torch.__file__), "_native", "ops", "bmm_outer_product", "aot.py"
    )
    spec = importlib.util.spec_from_file_location("bmm_outer_product_aot_t", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.covered_axes


def _reference(a, b):
    # triton.disabled() removes the JIT override AND flips the shared
    # native-AOT Context switch, so the reference is stock cublas.
    with torch.backends.python_native.triton.disabled():
        return torch.bmm(a, b)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestNativeAotBmmOuter(TestCase):
    def _outer(self, B, M, N, dtype=torch.float32, seed=0):
        torch.manual_seed(seed)
        a = torch.randn(B, M, 1, device="cuda", dtype=dtype)
        b = torch.randn(B, 1, N, device="cuda", dtype=dtype)
        return a, b

    @skipIfNoAotLib
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("m", [48, 128])  # one M per BLOCK_M bucket
    def test_covered_buckets_route_to_aot(self, dtype, m):
        a, b = self._outer(16, m, 256, dtype)
        ref = _reference(a, b)
        # Mask the JIT layer (deregister overrides) so the profiled
        # kernel can only come from the AOT stub.
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            self.assertTrue(
                _ran_aot(lambda: torch.bmm(a, b)),
                f"AOT kernel did not fire for {dtype} M={m}",
            )
            out = torch.bmm(a, b)
        finally:
            pn.triton.enable()
        self.assertEqual(out, ref, atol=0, rtol=0)

    @skipIfNoAotLib
    def test_bucket_edges(self):
        # Guard edges: M=32 uncovered, 33 covered (bm32), 96/97 bucket
        # boundary, 192 covered (bm64), 193 uncovered.
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            for m, covered in (
                (32, False),
                (33, True),
                (96, True),
                (97, True),
                (192, True),
                (193, False),
            ):
                a, b = self._outer(8, m, 128, seed=m)
                self.assertEqual(
                    _ran_aot(lambda a=a, b=b: torch.bmm(a, b)),
                    covered,
                    f"M={m}: expected covered={covered}",
                )
                out = torch.bmm(a, b)
                self.assertEqual(out, _reference(a, b), atol=0, rtol=0)
        finally:
            pn.triton.enable()

    @skipIfNoAotLib
    def test_uncovered_stay_jit_or_stock(self):
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            cases = {
                "fp16": self._outer(8, 48, 256, torch.float16),
                "small N": self._outer(8, 48, 64),
                "K>1 (not outer)": (
                    torch.randn(8, 48, 4, device="cuda"),
                    torch.randn(8, 4, 256, device="cuda"),
                ),
            }
            for label, (a, b) in cases.items():
                self.assertFalse(
                    _ran_aot(lambda a=a, b=b: torch.bmm(a, b)),
                    f"{label} must not route to AOT",
                )
        finally:
            pn.triton.enable()

    @skipIfNoAotLib
    def test_covered_shape_prefers_aot_over_jit(self):
        # With the JIT layer live, coverage subtraction sends the covered
        # bucket to AOT: profiler can't attribute (same kernel name), so
        # assert via the JIT compile cache staying cold.
        from torch._native.ops.bmm_outer_product.triton_kernels import (
            _bmm_outer_product_kernel,
        )

        a, b = self._outer(16, 48, 256, seed=42)

        def jit_cache_size():
            k = _bmm_outer_product_kernel.jit_kernel
            return sum(len(c) for c in getattr(k, "device_caches", {}).values())

        before = jit_cache_size()
        out = torch.bmm(a, b)
        self.assertEqual(jit_cache_size(), before, "covered call must not JIT-compile")
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoAotLib
    def test_noncontiguous_inputs(self):
        # Strides are runtime args; transposed views must work and match.
        torch.manual_seed(1)
        a = torch.randn(16, 1, 48, device="cuda").transpose(1, 2)
        b = torch.randn(16, 256, 1, device="cuda").transpose(1, 2)
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            self.assertTrue(_ran_aot(lambda: torch.bmm(a, b)))
            out = torch.bmm(a, b)
        finally:
            pn.triton.enable()
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoAotLib
    def test_out_variant_routes_to_aot(self):
        a, b = self._outer(16, 48, 256, seed=3)
        out = torch.empty(16, 48, 256, device="cuda")
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            self.assertTrue(_ran_aot(lambda: torch.bmm(a, b, out=out)))
        finally:
            pn.triton.enable()
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoAotLib
    def test_disabled_context_masks_aot(self):
        a, b = self._outer(16, 48, 256, seed=4)
        with torch.backends.python_native.triton.disabled():
            self.assertFalse(_ran_aot(lambda: torch.bmm(a, b)))

    def test_covered_axes_function_directly(self):
        # bind() short-circuits shape access behind is_outer -- exercising
        # it on non-3D inputs must not raise.
        bind = _load_covered_axes()

        # The new covered_axes folds bucket/min-N checks into "outer".
        a, b = torch.empty(4, 48, 1), torch.empty(4, 1, 256)
        self.assertTrue(bind(a, b)["outer"])
        self.assertFalse(bind(torch.empty(4, 48, 2), b)["outer"])  # K > 1
        self.assertFalse(bind(torch.empty(4, 8), torch.empty(4, 8))["outer"])  # 2-D
        self.assertFalse(bind(torch.empty(4, 16, 1), b)["outer"])  # off-bucket M
        self.assertFalse(bind(a, torch.empty(4, 1, 64))["outer"])  # small N

    def test_covered_call_correct_regardless_of_routing(self):
        a, b = self._outer(16, 48, 256, seed=5)
        out = torch.bmm(a, b)
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)


instantiate_parametrized_tests(TestNativeAotBmmOuter)


if __name__ == "__main__":
    run_tests()
