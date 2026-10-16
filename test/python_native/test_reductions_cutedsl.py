# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture. OpInfo tests cover
# numerical behavior.

import contextlib
import math
import sys
import unittest
import warnings
from unittest import mock

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native.ops.reductions import (
    kernel_coltile,
    kernel_general,
    kernel_rowtile,
    kernel_xcta,
)


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


@contextlib.contextmanager
def _aot_disabled():
    from torch import _native

    enabled = _native.aot_enabled()
    try:
        _native.set_aot_enabled(False)
        yield
    finally:
        _native.set_aot_enabled(enabled)


_FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
)


_OUT_CASES = (
    ("sum_dim", lambda x, o: torch.sum(x, dim=1, out=o[0]), ((8,),), (torch.float64,)),
    (
        "sum_full",
        lambda x, o: torch.ops.aten.sum.out(x, out=o[0]),
        ((),),
        (torch.float32,),
    ),
    (
        "mean_dim",
        lambda x, o: torch.mean(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float64,),
    ),
    ("mean_full", lambda x, o: torch.mean(x, out=o[0]), ((),), (torch.float32,)),
    (
        "nansum",
        lambda x, o: torch.nansum(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float64,),
    ),
    (
        "nansum_int",
        lambda x, o: torch.nansum(x, dim=1, dtype=torch.int32, out=o[0]),
        ((8,),),
        (torch.int32,),
    ),
    ("amax", lambda x, o: torch.amax(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    ("amin", lambda x, o: torch.amin(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    (
        "prod_dim",
        lambda x, o: torch.prod(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float64,),
    ),
    (
        "prod_full",
        lambda x, o: torch.ops.aten.prod.out(x, out=o[0]),
        ((),),
        (torch.float32,),
    ),
    (
        "argmax",
        lambda x, o: torch.argmax(x, dim=1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "argmin",
        lambda x, o: torch.argmin(x, dim=1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "max_dim",
        lambda x, o: torch.max(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.int64),
    ),
    ("max_full", lambda x, o: torch.max(x, out=o[0]), ((),), (torch.float32,)),
    (
        "min_dim",
        lambda x, o: torch.min(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.int64),
    ),
    ("min_full", lambda x, o: torch.min(x, out=o[0]), ((),), (torch.float32,)),
    ("var", lambda x, o: torch.var(x, dim=1, out=o[0]), ((8,),), (torch.float64,)),
    ("std", lambda x, o: torch.std(x, dim=1, out=o[0]), ((8,),), (torch.float64,)),
    (
        "vector_norm",
        lambda x, o: torch.linalg.vector_norm(x, dim=1, dtype=torch.float64, out=o[0]),
        ((8,),),
        (torch.float64,),
    ),
    ("all_dim", lambda x, o: torch.all(x, dim=1, out=o[0]), ((8,),), (torch.uint8,)),
    (
        "all_dims",
        lambda x, o: torch.all(x, dim=(0, 1), out=o[0]),
        ((),),
        (torch.uint8,),
    ),
    ("all_full", lambda x, o: torch.all(x, out=o[0]), ((),), (torch.uint8,)),
    ("any_dim", lambda x, o: torch.any(x, dim=1, out=o[0]), ((8,),), (torch.uint8,)),
    (
        "any_dims",
        lambda x, o: torch.any(x, dim=(0, 1), out=o[0]),
        ((),),
        (torch.uint8,),
    ),
    ("any_full", lambda x, o: torch.any(x, out=o[0]), ((),), (torch.uint8,)),
    (
        "count_nonzero_dim",
        lambda x, o: torch.ops.aten.count_nonzero.out(x, 1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "count_nonzero_dims",
        lambda x, o: torch.ops.aten.count_nonzero.dim_IntList_out(x, [0, 1], out=o[0]),
        ((),),
        (torch.int64,),
    ),
    (
        "var_mean",
        lambda x, o: torch.ops.aten.var_mean.correction_out(
            x, [1], correction=1, keepdim=False, out0=o[0], out1=o[1]
        ),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
    (
        "std_mean",
        lambda x, o: torch.ops.aten.std_mean.correction_out(
            x, [1], correction=1, keepdim=False, out0=o[0], out1=o[1]
        ),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
    (
        "aminmax",
        lambda x, o: torch.aminmax(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestCuTeDSLReductionWiring(TestCase):
    def _fired_count(self, fn):
        names = ("reduce_dim", "reduce_dim2", "reduce_all", "reduce_all2")
        orig = {nm: getattr(kernel_general, nm) for nm in names}
        n = [0]

        def wrap(f):
            def counting(*a, **k):
                n[0] += 1
                return f(*a, **k)

            return counting

        for nm in names:
            setattr(kernel_general, nm, wrap(orig[nm]))
        try:
            with _aot_disabled():
                fn()
        finally:
            for nm in names:
                setattr(kernel_general, nm, orig[nm])
        return n[0]

    def test_int64_index_split(self):
        """Carry Int64 winners through split scratch when the reduced extent requires it."""
        from torch._native.ops.reductions import overrides

        x = torch.zeros(1, 65537, device="cuda")
        x[:, 40000] = 1
        x[:, 50000] = 1
        out = []
        with mock.patch.object(overrides, "_INT32_LIMIT", 4):
            self.assertEqual(
                self._fired_count(lambda: out.append(torch.argmax(x, dim=-1))), 2
            )
        self.assertEqual(
            out[0], torch.tensor([40000], device="cuda", dtype=torch.int64)
        )

    @parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_reductions_are_served(self, dtype):
        """Serve complex reduction families with ATen-compatible values and dtypes."""
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        base = torch.complex(
            torch.randn(17, 9, device="cuda", dtype=real_dtype),
            torch.randn(17, 9, device="cuda", dtype=real_dtype),
        )
        base[0, 0] = complex(float("nan"), 2)
        base[1, 0] = complex(1, float("nan"))
        x = base.t().conj()
        multi = base.reshape(17, 3, 3).permute(1, 0, 2).conj()
        calls = (
            lambda: torch.sum(x, dim=-1),
            lambda: torch.sum(multi, dim=(0, 2)),
            lambda: torch.mean(x, dim=-1),
            lambda: torch.nansum(x, dim=-1),
            lambda: torch.prod(x, dim=-1),
            lambda: torch.var_mean(x, dim=-1),
            lambda: torch.var_mean(multi, dim=(0, 2)),
            lambda: torch.std(x, dim=-1),
            lambda: torch.linalg.vector_norm(x, dim=-1),
            lambda: torch.linalg.vector_norm(x, ord=float("inf"), dim=-1),
            lambda: torch.all(x, dim=-1),
            lambda: torch.any(x, dim=-1),
            lambda: torch.count_nonzero(x, dim=-1),
        )
        tol = 1e-3 if dtype is torch.complex64 else 1e-10
        for fn in calls:
            with self.subTest(fn=fn), _disabled():
                expected = fn()
            self.assertEqual(self._fired_count(fn), 1)
            self.assertEqual(fn(), expected, rtol=tol, atol=tol, exact_dtype=True)

    def test_complex_dtype_conversion_is_served(self):
        """Apply supported complex input and output conversions inside the override."""
        real = torch.randn(8, 16, device="cuda")
        complex_input = torch.complex(real, torch.randn_like(real))
        calls = (
            lambda: torch.sum(real, dim=1, dtype=torch.complex64),
            lambda: torch.mean(complex_input, dim=1, dtype=torch.float64),
            lambda: torch.nansum(complex_input, dim=1, dtype=torch.complex128),
            lambda: torch.prod(real, dim=1, dtype=torch.complex64),
        )
        for fn in calls:
            with self.subTest(fn=fn), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with _disabled():
                    expected = fn()
                self.assertEqual(self._fired_count(fn), 1)
                self.assertEqual(fn(), expected, rtol=1e-5, atol=1e-5, exact_dtype=True)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fp32_widening_is_fused(self, dtype):
        """Fuse low-precision widening into reduction loads to avoid an FP32 temporary."""
        from torch._native.ops.reductions import overrides

        x = torch.empty(8, 16, device="cuda", dtype=dtype)
        seen = []

        def record(make_trait, key, source, red, keepdim, out_dtype, **kwargs):
            seen.append((key, source.dtype, out_dtype))
            return torch.empty(8, device=source.device, dtype=out_dtype)

        with unittest.mock.patch.object(overrides, "_run1", side_effect=record):
            overrides._sum_impl(x, dim=1, dtype=torch.float32)
            overrides._mean_impl(x, dim=1, dtype=torch.float32)
            overrides._prod_impl(x, dim=1, dtype=torch.float32)
            overrides._vector_norm_impl(x, 2, dim=1, dtype=torch.float32)
        self.assertEqual(
            seen,
            [(key, dtype, torch.float32) for key in ("sum", "mean", "prod", "vnorm2")],
        )

    def test_integral_mean_dtype_is_served(self):
        """Serve integral mean only when an explicit floating or complex dtype makes it valid."""
        input_dtypes = (
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
        output_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        values = torch.arange(28, device="cuda").reshape(4, 7) % 5
        for input_dtype in input_dtypes:
            x = values.ne(0) if input_dtype is torch.bool else values.to(input_dtype)
            for output_dtype in output_dtypes:
                fn = lambda: torch.mean(  # noqa: E731
                    x, dim=1, dtype=output_dtype
                )
                with (
                    self.subTest(input_dtype=input_dtype, output_dtype=output_dtype),
                    _disabled(),
                ):
                    expected = fn()
                self.assertEqual(self._fired_count(fn), 1)
                self.assertEqual(fn(), expected, rtol=1e-3, atol=1e-3)

    @parametrize("dtype", _FLOAT8_DTYPES)
    def test_float8_reduction_conversions_are_served(self, dtype):
        """Accumulate float8 inputs in supported output dtypes without falling back."""
        values = torch.tensor(
            [[0.5, 1, 2, 1, 0.5, 1, 2, 1]] * 4,
            device="cuda",
        )
        x = values.to(dtype)
        calls = (
            ("sum_f32", lambda: torch.sum(x, dim=1, dtype=torch.float32)),
            ("sum_c64", lambda: torch.sum(x, dim=1, dtype=torch.complex64)),
            ("mean_f32", lambda: torch.mean(x, dim=1, dtype=torch.float32)),
            ("nansum_f32", lambda: torch.nansum(x, dim=1, dtype=torch.float32)),
            ("prod_f32", lambda: torch.prod(x, dim=1, dtype=torch.float32)),
            ("prod_i32", lambda: torch.prod(x, dim=1, dtype=torch.int32)),
            ("count", lambda: torch.count_nonzero(x, dim=1)),
        )
        for name, fn in calls:
            with self.subTest(name=name), _disabled():
                expected = fn()
            self.assertEqual(self._fired_count(fn), 1)
            self.assertEqual(fn(), expected, rtol=1e-4, atol=1e-4)

    def test_nansum_integer_conversion_contract(self):
        """Match nansum conversion support at integer, float8, and complex boundaries."""
        fp8 = torch.ones(4, 8, device="cuda", dtype=torch.float8_e4m3fn)
        complex64 = torch.ones(4, 8, device="cuda", dtype=torch.complex64)
        out = torch.empty(4, device="cuda", dtype=torch.int32)
        calls = (
            lambda: torch.nansum(fp8, dim=1, dtype=torch.int32),
            lambda: torch.nansum(complex64, dim=1, out=out),
        )
        for fn in calls:
            with self.subTest(fn=fn), self.assertRaises(RuntimeError):
                fn()

    def test_argmax_declines_bool(self):
        """Preserve ATen's bool argmax error instead of accepting an unsupported dtype."""
        x = torch.ones(64, 64, device="cuda", dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "does not support bool"):
            torch.argmax(x, dim=-1)

    def test_bool_dim_reductions_canonicalize_storage_bytes(self):
        """Treat every nonzero bool storage byte as true before reducing dimensions."""
        raw = torch.tensor(
            [[2, 1, 0, 3], [2, 1, 3, 4], [0, 2, 0, 3], [0, 0, 0, 0]],
            device="cuda",
            dtype=torch.uint8,
        )
        x = raw.view(torch.bool)
        for op in (torch.max, torch.min):
            with self.subTest(op=op):
                with _disabled():
                    expected_raw = op(raw, dim=1)
                    expected_bool = op(x, dim=1)
                # uint8 and its bool view have the same descriptor; their compile-time
                # canonicalization setting must still select distinct cached kernels.
                self.assertEqual(op(raw, dim=1), expected_raw)
                self.assertEqual(op(x, dim=1), expected_bool)

    def test_noncontiguous_full_reductions_are_served(self):
        """Serve full reductions over noncontiguous layouts via indexed addressing."""
        xt = torch.randn(64, 128, device="cuda").t()
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.argmax(t),
        ):
            with self.subTest(fn=fn):
                self.assertEqual(self._fired_count(lambda: fn(xt)), 1)
                with _disabled():
                    ref = fn(xt)
                self.assertEqual(fn(xt), ref, exact_dtype=True)

    def test_empty_reduction_without_identity_raises(self):
        """Preserve required errors for empty extrema and index reductions."""
        empty_axis = torch.randn(5, 0, device="cuda")
        for fn in (torch.amax, torch.amin, torch.argmax, torch.aminmax):
            with self.subTest(fn=fn):
                with self.assertRaises((RuntimeError, IndexError)):
                    fn(empty_axis, dim=1)

    def test_int32_sum_does_not_wrap_at_2_31(self):
        # The accumulator must be Int64: 512 * 2**24 overflows int32, and aten's integral
        # sum does not wrap there, so neither may ours. (An Int32 acc silently wrapped.)
        """Promote int32 sums before accumulation so totals above INT32_MAX stay correct."""
        x = torch.full((4, 512), 1 << 24, device="cuda", dtype=torch.int32)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        got = torch.sum(x, dim=-1)
        self.assertEqual(got.dtype, torch.int64)
        self.assertEqual(got, torch.full((4,), 512 * (1 << 24), device="cuda"))

    @staticmethod
    def _compiled_kernel_count():
        caches = (
            kernel_general._COMPILE_CACHE,
            kernel_rowtile._CACHE,
            kernel_coltile._CACHE,
            kernel_xcta._PLAN,
        )
        return sum(len(c) for c in caches), caches

    def test_kernel_count_does_not_scale_with_shape(self):
        """Keep dynamic shape variation from causing unbounded kernel compilation."""
        # These sizes share a vector class and bucket, so they must share kernels.
        few = [4096, 4104]
        many = [4096, 4104, 4112, 4120, 4128, 4136, 4144, 4152, 4160, 4168]

        def run(sizes):
            _, caches = self._compiled_kernel_count()
            for c in caches:
                c.clear()
            for n in sizes:
                x = torch.randn(512, n, device="cuda")
                torch.sum(x, dim=-1)
                torch.amax(x, dim=-1)
                # column path too: its split factor and stage-2 mapping are shape-derived
                torch.sum(x, dim=0)
                del x
            return self._compiled_kernel_count()[0]

        n_few = run(few)
        n_many = run(many)
        self.assertEqual(
            n_many,
            n_few,
            f"compiled kernels grew from {n_few} (2 shapes) to {n_many} (10 shapes): a "
            f"size-derived const_expr has crept back in",
        )

    def test_scalar_is_served(self):
        """Serve scalar reductions with the same shape and dtype contracts as ATen."""
        s = torch.tensor(3.5, device="cuda")
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0, keepdim=True),
        ):
            with self.subTest(fn=fn):
                self.assertGreaterEqual(self._fired_count(lambda: fn(s)), 1)
                with _disabled():
                    ref = fn(s)
                self.assertEqual(fn(s), ref, exact_dtype=True)
        for bad in (1, -2, [0, -1]):
            with self.subTest(dim=bad), self.assertRaises((RuntimeError, IndexError)):
                torch.sum(s, dim=bad)

    def test_cow_input_served_and_preserved(self):
        # A COW input is SERVED (it exports read-only, so from_dlpack reads const_data_ptr()) and
        # must STAY COW: reading it must not materialize it.
        """Read copy-on-write inputs without materializing or mutating their storage."""
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_is_served(self):
        """Compile and launch against the input device rather than the current device."""
        x0 = torch.randn(128, 512, device="cuda:0")
        x1 = x0.to("cuda:1")
        with _disabled():
            expected = torch.sum(x1, dim=-1)
        with torch.cuda.device(0):
            torch.sum(x0, dim=-1)
            self.assertEqual(self._fired_count(lambda: torch.sum(x1, dim=-1)), 1)
            actual = torch.sum(x1, dim=-1)
            self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)

    def test_graph_capturable(self):
        """Keep reduction overrides capture-safe across real and complex fast paths."""
        # The override must capture into a CUDA graph and replay correctly (the
        # earlier _stream() bug made cute launches deadlock / produce empty graphs).
        for dtype, shape in (
            (torch.float32, (8192, 1024)),
            (torch.complex64, (512, 1024)),
        ):
            x = torch.randn(shape, device="cuda", dtype=dtype)
            f = lambda: torch.sum(x, dim=-1)  # noqa: E731
            with _disabled():
                ref = f()
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = f()
            g.replay()
            torch.cuda.synchronize()
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    def test_strided_single_element_view_is_served(self):
        # REGRESSION: is_contiguous() is True for ANY single-element tensor, since with one element
        # the stride is unobservable -- so a.diagonal(offset=2) is a contiguous shape-(1,) tensor that
        # still declares stride (4,), which the DSL rejected outright. The wrap restrides it, and
        # these are SERVED: declining would give up coverage for a difference nothing can observe.
        """Serve a one-element strided view without assuming scalar contiguity."""
        a = torch.randn(5, 3, device="cuda", dtype=torch.float64)
        d = a.diagonal(offset=2)
        self.assertEqual(d.shape, torch.Size([1]))
        self.assertNotEqual(d.stride(), (1,))  # the leftover stride is the whole point
        with _disabled():
            ref = d.sum()
        self.assertEqual(d.sum(), ref)

    def test_misaligned_inputs_are_served(self):
        """Route misaligned row, cross-CTA, and column inputs to alignment-safe kernels."""
        cases = (
            ("row", (256, 2), lambda t: torch.amax(t, dim=1)),
            ("cross-CTA row", (2, 32768), lambda t: torch.amax(t, dim=1)),
            ("column", (4096, 256), lambda t: torch.amax(t, dim=0)),
            ("general", (8, 16, 32), lambda t: torch.sum(t, dim=1)),
        )
        for name, shape, fn in cases:
            base = torch.randn(math.prod(shape) + 1, device="cuda", dtype=torch.float64)
            t = base[1:].view(shape)
            self.assertNotEqual(t.data_ptr() % 16, 0)
            with _disabled():
                expected = fn(t)
            with self.subTest(name=name):
                self.assertEqual(self._fired_count(lambda: fn(t)), 1)
                self.assertEqual(fn(t), expected, atol=1e-10, rtol=1e-10)

    @parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
    def test_promoting_unsigned_reductions_are_served(self, dtype):
        """Match promoted unsigned sum and product semantics in the override."""
        x = torch.arange(1, 25, device="cuda", dtype=torch.int64).to(dtype).view(4, 6)
        for fn in (
            lambda: torch.sum(x, dim=1),
            lambda: torch.nansum(x, dim=1),
            lambda: torch.prod(x, dim=1),
            lambda: torch.count_nonzero(x, dim=1),
        ):
            with self.subTest(fn=fn):
                with _disabled():
                    expected = fn()
                self.assertEqual(self._fired_count(fn), 1)
                self.assertEqual(fn(), expected)

    def test_uint64_to_int64_promotion_wraps_like_aten(self):
        """Match ATen's uint64-to-int64 modular promotion for sum."""
        x = torch.tensor([-1, 2], device="cuda", dtype=torch.int64).view(torch.uint64)
        with _disabled():
            expected = x.sum()
        self.assertEqual(self._fired_count(lambda: x.sum()), 1)
        self.assertEqual(x.sum(), expected)

    def test_invalid_degrees_of_freedom_warn_and_are_served(self):
        """Match warnings and NaNs for Welford corrections with no valid divisor."""
        x = torch.randn(4, 16, device="cuda")
        cases = (
            ("var", lambda: torch.var(x, dim=1, correction=16)),
            ("std", lambda: torch.std(x, dim=1, correction=16)),
            ("var_mean", lambda: torch.var_mean(x, dim=1, correction=16)),
            ("std_mean", lambda: torch.std_mean(x, dim=1, correction=16)),
        )
        for name, fn in cases:
            with (
                _disabled(),
                self.assertWarnsRegex(UserWarning, "degrees of freedom is <= 0"),
            ):
                expected = fn()
            actual = []
            with (
                self.subTest(name=name),
                self.assertWarnsRegex(UserWarning, "degrees of freedom is <= 0"),
            ):
                fired = self._fired_count(lambda: actual.append(fn()))
            self.assertEqual(fired, 1)
            self.assertEqual(actual[0], expected)

    @parametrize(
        "correction",
        [float("-inf"), float("inf"), float("nan")],
        name_fn=lambda correction: str(correction),
    )
    def test_nonfinite_correction_is_served(self, correction):
        """Handle nonfinite Welford corrections without declining the override."""
        x = torch.tensor([[1.0, 2.0], [1.0, 1.0]], device="cuda")
        with _disabled(), warnings.catch_warnings(record=True) as expected_warnings:
            expected = torch.var_mean(x, dim=1, correction=correction)
        actual = []
        with warnings.catch_warnings(record=True) as actual_warnings:
            fired = self._fired_count(
                lambda: actual.append(torch.var_mean(x, dim=1, correction=correction))
            )
        self.assertEqual(fired, 1)
        self.assertEqual(actual[0], expected)
        self.assertEqual(
            ["degrees of freedom is <= 0" in str(w.message) for w in actual_warnings],
            ["degrees of freedom is <= 0" in str(w.message) for w in expected_warnings],
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestCuTeDSLReductionOut(TestCase):
    @parametrize(
        "name,call,shapes,dtypes",
        _OUT_CASES,
        name_fn=lambda name, call, shapes, dtypes: name,
    )
    def test_out_overload(self, device, name, call, shapes, dtypes):
        """Preserve out identity, resize, dtype conversion, and routing for core overloads."""
        x = torch.randn(8, 16, device=device)
        if name.startswith("nansum"):
            x[0, 0] = float("nan")

        def make_outs():
            return tuple(
                torch.empty(shape, device=device, dtype=dtype)
                for shape, dtype in zip(shapes, dtypes)
            )

        expected = make_outs()
        with _disabled():
            call(x, expected)

        names = ("reduce_dim", "reduce_dim2", "reduce_all")
        original = {entry: getattr(kernel_general, entry) for entry in names}
        fired = [0]

        def wrap(fn):
            def counting(*args, **kwargs):
                fired[0] += 1
                return fn(*args, **kwargs)

            return counting

        for entry, fn in original.items():
            setattr(kernel_general, entry, wrap(fn))
        actual = make_outs()
        if name == "sum_dim":
            actual = (torch.empty(1, device=device, dtype=dtypes[0]),)
        try:
            with _aot_disabled():
                if name == "sum_dim":
                    with self.assertWarnsRegex(
                        UserWarning, "An output with one or more elements was resized"
                    ):
                        result = call(x, actual)
                else:
                    result = call(x, actual)
        finally:
            for entry, fn in original.items():
                setattr(kernel_general, entry, fn)

        self.assertGreater(fired[0], 0)
        returned = result if isinstance(result, tuple) else (result,)
        for got, out in zip(returned, actual):
            self.assertIs(got, out)
        for got, ref in zip(actual, expected):
            tol = 1e-10 if ref.dtype is torch.float64 else 1e-2
            self.assertEqual(got, ref, rtol=tol, atol=tol)

    def test_mean_integral_out(self, device):
        """Match mean's integral out-dtype validation for real and complex inputs."""
        for input_dtype in (torch.float32, torch.complex64):
            for output_dtype in (
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                for n in (1, 3):
                    real = torch.arange(2 * n, device=device).reshape(2, n) + 0.75
                    x = (
                        torch.complex(real, real + 1)
                        if input_dtype.is_complex
                        else real
                    )
                    expected = torch.empty(2, device=device, dtype=output_dtype)
                    with _disabled(), warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        torch.mean(x, dim=1, out=expected)
                    actual = torch.empty_like(expected)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        result = torch.mean(x, dim=1, out=actual)
                    with self.subTest(
                        input_dtype=input_dtype, output_dtype=output_dtype, n=n
                    ):
                        self.assertIs(result, actual)
                        self.assertEqual(actual, expected)

    @parametrize("output_dtype", [torch.complex64, torch.complex128])
    def test_nansum_complex_out(self, device, output_dtype):
        """Convert complex nansum results into supported real and complex out tensors."""
        x = torch.tensor(
            [[float("nan"), 2, 3], [4, float("nan"), 5]],
            device=device,
        )
        expected = torch.empty(2, device=device, dtype=output_dtype)
        with _disabled():
            torch.nansum(x, dim=1, out=expected)
        actual = torch.empty_like(expected)
        result = torch.nansum(x, dim=1, out=actual)
        self.assertIs(result, actual)
        self.assertEqual(actual, expected)

    @parametrize("dtype", _FLOAT8_DTYPES)
    def test_float8_out_overloads(self, device, dtype):
        """Serve float8 reductions through supported out dtypes and resize behavior."""
        x = torch.tensor(
            [[0.5, 1, 2, 1, 0.5, 1, 2, 1]] * 4,
            device=device,
        ).to(dtype)

        def run():
            outs = (
                torch.empty(4, device=device),
                torch.empty(4, device=device),
                torch.empty(4, device=device, dtype=torch.complex64),
                torch.empty(4, device=device, dtype=torch.int32),
                torch.empty(4, device=device),
                torch.empty(4, device=device, dtype=torch.float16),
                torch.empty(4, device=device, dtype=torch.int64),
            )
            torch.sum(x, dim=1, out=outs[0])
            torch.mean(x, dim=1, out=outs[1])
            torch.nansum(x, dim=1, out=outs[2])
            torch.prod(x, dim=1, out=outs[3])
            torch.var(x, dim=1, out=outs[4])
            torch.std(x, dim=1, out=outs[5])
            torch.ops.aten.count_nonzero.out(x, 1, out=outs[6])
            return outs

        with _disabled():
            expected = run()
        actual = run()
        for got, ref in zip(actual, expected):
            self.assertEqual(got, ref, rtol=1e-3, atol=1e-3)

    @parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_out_overloads(self, device, dtype):
        """Serve complex reduction families through all supported out conversions."""
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        x = torch.complex(
            torch.randn(8, 16, device=device, dtype=real_dtype),
            torch.randn(8, 16, device=device, dtype=real_dtype),
        )

        def run():
            sum_out = torch.empty(8, device=device, dtype=dtype)
            var_out = torch.empty(8, device=device, dtype=real_dtype)
            norm_out = torch.empty(8, device=device, dtype=real_dtype)
            pair = (
                torch.empty(8, device=device, dtype=real_dtype),
                torch.empty(8, device=device, dtype=dtype),
            )
            torch.sum(x, dim=1, out=sum_out)
            torch.var(x, dim=1, out=var_out)
            torch.linalg.vector_norm(x, dim=1, out=norm_out)
            torch.ops.aten.var_mean.correction_out(
                x, [1], correction=1, keepdim=False, out0=pair[0], out1=pair[1]
            )
            return sum_out, var_out, norm_out, *pair

        with _disabled():
            expected = run()
        actual = run()
        tol = 1e-3 if dtype is torch.complex64 else 1e-10
        for got, ref in zip(actual, expected):
            self.assertEqual(got, ref, rtol=tol, atol=tol, exact_dtype=True)


instantiate_parametrized_tests(TestCuTeDSLReductionWiring)
instantiate_device_type_tests(TestCuTeDSLReductionOut, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
