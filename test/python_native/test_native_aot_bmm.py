# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for the native-AOT bmm outer-product kernels @ CUDA.

Compared exactly: K == 1 is a per-element product, with no accumulation order to differ
from the reference. Tests needing the AOT library skip without it."""

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoTritonDSL,
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
    # Both layers launch the same-named kernel, so attribution needs the JIT side
    # masked rather than a profiler name match.
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
    with torch.backends.python_native.triton.disabled():
        return torch.bmm(a, b)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoTritonDSL
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
    def test_every_m_is_covered_across_the_bucket_boundary(self):
        pn = torch.backends.python_native
        pn.triton.disable()
        try:
            for m, covered in (
                (1, True),
                (32, True),
                (96, True),
                (97, True),
                (192, True),
                (1024, True),
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
    def test_uncovered_dtypes_and_small_n_avoid_aot(self):
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
    def test_a_covered_shape_leaves_the_jit_compile_cache_cold(self):
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
    def test_a_transposed_view_is_served_and_matches(self):
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

    def test_covered_axes_answers_without_raising_on_odd_shapes(self):
        bind = _load_covered_axes()
        a, b = torch.empty(4, 48, 1), torch.empty(4, 1, 256)
        self.assertTrue(bind(a, b)["outer"])
        self.assertFalse(bind(torch.empty(4, 48, 2), b)["outer"])  # K > 1
        self.assertFalse(bind(torch.empty(4, 8), torch.empty(4, 8))["outer"])  # 2-D
        self.assertTrue(bind(torch.empty(4, 16, 1), b)["outer"])  # small M, now covered
        self.assertTrue(bind(torch.empty(4, 4096, 1), b)["outer"])  # and large M
        self.assertFalse(bind(a, torch.empty(4, 1, 64))["outer"])  # small N
        self.assertFalse(bind(torch.empty(4, 0, 1), b)["outer"])  # empty declines

    def test_the_result_is_exact_whichever_layer_serves_it(self):
        a, b = self._outer(16, 48, 256, seed=5)
        out = torch.bmm(a, b)
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)


instantiate_parametrized_tests(TestNativeAotBmmOuter)


if __name__ == "__main__":
    run_tests()
