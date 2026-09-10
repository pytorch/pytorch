# Owner(s): ["module: dsl-native-ops"]
"""End-to-end tests for the native-AOT bmm outer-product kernels @ CUDA.

Compared exactly: K == 1 is a per-element product, with no accumulation order to differ
from the reference. Tests needing the AOT library skip without it."""

import contextlib
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoNativeAot,
    skipIfNoTritonDSL,
    TestCase,
)


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
    # By path, through the loader the declaration tooling itself uses: aot.py is
    # stdlib-only at import, and nothing imports it as a module.
    import os

    from torchgen.native_aot_decl import load_by_path

    path = os.path.join(
        os.path.dirname(torch.__file__), "_native", "ops", "bmm_outer_product", "aot.py"
    )
    return load_by_path("bmm_outer_product_aot_t", path).covered_axes


def _reference(a, b):
    # disabled() masks the AOT stubs as well, so this is stock aten.
    with torch.backends.python_native.triton.disabled():
        return torch.bmm(a, b)


@contextlib.contextmanager
def _jit_masked():
    """Deregister the JIT overrides, leaving the AOT stubs live.

    Narrower than disabled(): what runs inside is exactly what the embedded
    kernels serve, so _ran_aot attributes a launch to the AOT layer.
    """
    pn = torch.backends.python_native
    pn.triton.disable()
    try:
        yield
    finally:
        pn.triton.enable()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoTritonDSL
class TestNativeAotBmmOuter(TestCase):
    def _outer(self, B, M, N, dtype=torch.float32, seed=0):
        torch.manual_seed(seed)
        a = torch.randn(B, M, 1, device="cuda", dtype=dtype)
        b = torch.randn(B, 1, N, device="cuda", dtype=dtype)
        return a, b

    def _uncovered_shapes(self):
        """No exported point serves these: an uncovered dtype, an N below the smallest
        exported tile, and a K > 1 matmul that is not an outer product."""
        return {
            "fp16": self._outer(8, 48, 256, torch.float16),
            "small N": self._outer(8, 48, 64),
            "K>1 (not outer)": (
                torch.randn(8, 48, 4, device="cuda"),
                torch.randn(8, 4, 256, device="cuda"),
            ),
        }

    def _declining_layouts(self):
        """Layouts the exported points bake out: they hint 16B alignment and a stride of
        1 innermost, so a kernel addressing these as if contiguous reads the wrong
        elements."""
        torch.manual_seed(7)
        return {
            "a inner stride 2": (
                torch.randn(16, 48, 2, device="cuda")[:, :, :1],
                torch.randn(16, 1, 256, device="cuda"),
            ),
            "b inner stride 2": (
                torch.randn(16, 48, 1, device="cuda"),
                torch.randn(16, 1, 512, device="cuda")[:, :, ::2],
            ),
            "a offset 4B": (
                torch.randn(16, 49, 1, device="cuda")[:, 1:, :],
                torch.randn(16, 1, 256, device="cuda"),
            ),
        }

    @skipIfNoNativeAot
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("m", [48, 128])  # one M per BLOCK_M bucket
    def test_covered_buckets_route_to_aot(self, dtype, m):
        a, b = self._outer(16, m, 256, dtype)
        ref = _reference(a, b)
        with _jit_masked():
            self.assertTrue(
                _ran_aot(lambda: torch.bmm(a, b)),
                f"AOT kernel did not fire for {dtype} M={m}",
            )
            out = torch.bmm(a, b)
        self.assertEqual(out, ref, atol=0, rtol=0)

    @skipIfNoNativeAot
    @parametrize("n", [129, 200, 255, 384])
    def test_n_tails_are_masked_and_exact(self, n):
        # BLOCK_N is 128 for every exported point, so each of these N values leaves a
        # partial tile: an unmasked store would write past the row or drop its tail.
        a, b = self._outer(8, 48, n, seed=n)
        ref = _reference(a, b)
        with _jit_masked():
            self.assertTrue(_ran_aot(lambda: torch.bmm(a, b)), f"N={n} did not fire")
            out = torch.bmm(a, b)
        self.assertEqual(out, ref, atol=0, rtol=0)

    @skipIfNoNativeAot
    def test_layouts_the_kernels_do_not_serve_decline(self):
        # Declining is not enough: the call must also reach aten intact.
        for label, (a, b) in self._declining_layouts().items():
            with _jit_masked():
                self.assertFalse(
                    _ran_aot(lambda a=a, b=b: torch.bmm(a, b)),
                    f"{label} must not route to AOT",
                )
                out = torch.bmm(a, b)
            self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    def test_mixed_dtypes_raise_instead_of_being_served(self):
        # Neither layer runs aten's dtype-equality check (it lives in the meta
        # function), and both specialize on self's dtype: without their own checks a
        # bf16 mat2 would be read as f32 and the call would return garbage.
        a, b = self._outer(8, 48, 256)
        b16 = self._outer(8, 48, 256, torch.bfloat16)[1]
        with self.assertRaises(RuntimeError):
            torch.bmm(a, b16)
        with self.assertRaises(RuntimeError):
            torch.bmm(
                a, b, out=torch.empty(8, 48, 256, device="cuda", dtype=torch.bfloat16)
            )

    @skipIfNoNativeAot
    def test_cpp_coverage_matches_what_the_stub_serves(self):
        # covers_bmm is what the router consults to drop a call's JIT route. Claiming
        # a call the stub then declines loses both accelerated routes, so the two
        # answers have to agree shape for shape.
        cases = {
            "covered f32": self._outer(16, 48, 256),
            "covered bf16": self._outer(16, 48, 256, torch.bfloat16),
            "small M": self._outer(16, 1, 256),
            "large M": self._outer(4, 4096, 256),
            "N tail": self._outer(8, 97, 129),
            "f64": self._outer(8, 48, 256, torch.float64),
            **self._uncovered_shapes(),
            **self._declining_layouts(),
        }
        for label, (a, b) in cases.items():
            claimed = torch.ops._native_aot.covers_bmm(a, b)
            with _jit_masked():
                fired = _ran_aot(lambda a=a, b=b: torch.bmm(a, b))
            self.assertEqual(
                claimed,
                fired,
                f"{label}: coverage says {claimed} but the stub {'fired' if fired else 'declined'}",
            )

    def test_python_coverage_declines_int32_overflowing_shapes(self):
        # The ABI narrows sizes and strides to int32_t, and a wrapped offset writes
        # outside the buffer. Meta tensors, since the shapes are unallocatable.
        bind = _load_covered_axes()
        b = torch.empty(2, 1, 256, device="meta")
        self.assertTrue(bind(torch.empty(2, 1024, 1, device="meta"), b)["outer"])
        # out's numel: 2 * 2**24 * 256 elements.
        self.assertFalse(bind(torch.empty(2, 2**24, 1, device="meta"), b)["outer"])
        # self.stride(0), with numel still in range.
        self.assertFalse(bind(torch.empty(2, 2**31, 1, device="meta"), b)["outer"])

    @skipIfNoNativeAot
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "requires at least 2 visible CUDA devices"
    )
    def test_a_covered_shape_runs_on_a_non_current_device(self):
        # The launcher caches one CUfunction per device and establishes a context if
        # the calling thread has none; both are per-device state a single-GPU run
        # never exercises.
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            torch.manual_seed(13)
            a = torch.randn(16, 48, 1, device="cuda:1")
            b = torch.randn(16, 1, 256, device="cuda:1")
            ref = _reference(a, b)
            with _jit_masked():
                self.assertTrue(_ran_aot(lambda: torch.bmm(a, b)))
                out = torch.bmm(a, b)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out.device, torch.device("cuda:1"))
            self.assertEqual(out, ref, atol=0, rtol=0)
        finally:
            torch.cuda.set_device(old_device)

    @skipIfNoNativeAot
    def test_every_m_is_covered_across_the_bucket_boundary(self):
        with _jit_masked():
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

    @skipIfNoNativeAot
    def test_uncovered_dtypes_and_small_n_avoid_aot(self):
        with _jit_masked():
            for label, (a, b) in self._uncovered_shapes().items():
                self.assertFalse(
                    _ran_aot(lambda a=a, b=b: torch.bmm(a, b)),
                    f"{label} must not route to AOT",
                )

    @skipIfNoNativeAot
    def test_a_covered_shape_leaves_the_jit_compile_cache_cold(self):
        from torch._native.instrumentation import _triton_cache_size
        from torch._native.ops.bmm_outer_product.triton_kernels import (
            _bmm_outer_product_kernel,
        )

        a, b = self._outer(16, 48, 256, seed=42)
        kernel = _bmm_outer_product_kernel.jit_kernel
        before = _triton_cache_size(kernel)
        self.assertIsNotNone(before, "triton no longer exposes device_caches")
        out = torch.bmm(a, b)
        self.assertEqual(
            _triton_cache_size(kernel), before, "covered call must not JIT-compile"
        )
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoNativeAot
    def test_a_transposed_view_is_served_and_matches(self):
        torch.manual_seed(1)
        a = torch.randn(16, 1, 48, device="cuda").transpose(1, 2)
        b = torch.randn(16, 256, 1, device="cuda").transpose(1, 2)
        with _jit_masked():
            self.assertTrue(_ran_aot(lambda: torch.bmm(a, b)))
            out = torch.bmm(a, b)
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoNativeAot
    def test_out_variant_routes_to_aot(self):
        a, b = self._outer(16, 48, 256, seed=3)
        out = torch.empty(16, 48, 256, device="cuda")
        with _jit_masked():
            self.assertTrue(_ran_aot(lambda: torch.bmm(a, b, out=out)))
        self.assertEqual(out, _reference(a, b), atol=0, rtol=0)

    @skipIfNoNativeAot
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
