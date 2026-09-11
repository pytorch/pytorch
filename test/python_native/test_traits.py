# Owner(s): ["module: dsl-native-ops"]
# Structural checks for reduction traits and the local var/std divisor clamp.
import inspect
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


# Guard before importing the cutlass-dependent module; sys.exit keeps direct runs successful.
if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass
import cutlass.cute as cute

from torch._native.cutedsl import launch as _L
from torch._native.ops.reductions import traits as T


class TestTraitProtocol(TestCase):
    def _traits(self):
        found = {
            name: obj
            for name, obj in vars(T).items()
            if inspect.isclass(obj) and name.endswith("Ops")
        }
        # Without this, a rename would make every assertion below vacuously pass on an empty set.
        self.assertGreater(len(found), 5)
        return found

    def _make(self, trait):
        params = inspect.signature(trait.__init__).parameters
        kwargs = {"acc": cutlass.Float32}
        if "p" in params and params["p"].default is inspect.Parameter.empty:
            kwargs["p"] = 2.0  # a norm has no default exponent
        return trait(**kwargs)

    def test_every_trait_implements_the_protocol(self):
        # Trees use leaf then combine; transforms implemented only in reduce silently fold raw values.
        for name, trait in sorted(self._traits().items()):
            with self.subTest(trait=name):
                for method in ("init", "leaf", "combine", "reduce", "project"):
                    self.assertTrue(
                        callable(getattr(trait, method, None)),
                        f"{name}.{method} is missing",
                    )
                self.assertEqual(
                    sorted(inspect.signature(trait.leaf).parameters),
                    ["idx", "self", "val"],
                    f"{name}.leaf must take (val, idx)",
                )

    def test_field_count_matches_field_dtypes(self):
        # nfields sizes partial buffers and tuples; fdtypes types them, so disagreement misallocates.
        for name, trait in sorted(self._traits().items()):
            with self.subTest(trait=name):
                t = self._make(trait)
                self.assertEqual(len(t.fdtypes), t.nfields)
                self.assertGreaterEqual(t.nfields, 1)

    def test_init_returns_one_accumulator(self):
        # init must return exactly the accumulator tuple combine expects, so the two compose.
        for name, trait in sorted(self._traits().items()):
            with self.subTest(trait=name):
                t = self._make(trait)
                self.assertEqual(len(t.init()), t.nfields)

    def test_butterfly_width_is_a_power_of_two(self):
        self.assertEqual(T._offsets(1), [])
        self.assertEqual(T._offsets(8), [4, 2, 1])
        self.assertEqual(T._offsets(8, ascending=True), [1, 2, 4])
        for width in (0, 3, 6):
            with (
                self.subTest(width=width),
                self.assertRaisesRegex(ValueError, "positive power of two"),
            ):
                T._offsets(width)

    @unittest.skipUnless(TEST_CUDA, "CUDA required")
    def test_welford_divisor_clamps_at_zero(self):
        # correction >= n must divide by zero, yielding ATen's +inf, not negative variance.
        # A one-thread kernel probes the cute.jit helper directly.
        @cute.kernel
        def probe(dst: cute.Tensor, nf: cutlass.Float32, correction: cutlass.Constexpr):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                dst[0] = T._welford_denom(cutlass.Float32, nf, correction)

        @cute.jit
        def run(
            dst: cute.Tensor, nf: cutlass.Float32, correction: cutlass.Constexpr, stream
        ):
            probe(dst, nf, correction).launch(
                grid=[1, 1, 1], block=[1, 1, 1], stream=stream
            )

        out = torch.zeros(1, device="cuda")
        # Test both clamp cases and an ordinary positive divisor.
        for nf, correction, want in (
            (8.0, 1.0, 7.0),
            (8.0, 8.0, 0.0),
            (8.0, 15.0, 0.0),
        ):
            with self.subTest(nf=nf, correction=correction):
                fn = _L.compile_kernel(
                    run,
                    _L.fake_compact(cutlass.Float32, (1,), align=4),
                    cutlass.Float32(0.0),
                    correction,
                    _L.stream(),
                )
                # correction is compile-time; the callable takes only output, n, and stream.
                fn(out, nf, _L.stream())
                torch.cuda.synchronize()
                self.assertEqual(out.item(), want)

    @unittest.skipUnless(TEST_CUDA, "CUDA required")
    def test_welford_empty_accumulator_is_identity(self):
        trait = T.WelfordOps(acc=cutlass.Float32)

        @cute.kernel
        def probe(dst: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                empty = trait.init()
                value = trait.leaf(cutlass.Float32(1e20), cutlass.Int32(0))
                left = trait.combine(empty, value)
                right = trait.combine(value, empty)
                dst[0] = left[0]
                dst[1] = left[1]
                dst[2] = left[2]
                dst[3] = right[0]
                dst[4] = right[1]
                dst[5] = right[2]

        @cute.jit
        def run(dst: cute.Tensor, stream):
            probe(dst).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

        out = torch.empty(6, device="cuda")
        fn = _L.compile_kernel(
            run,
            _L.fake_compact(cutlass.Float32, (6,), align=4),
            _L.stream(),
        )
        fn(out, _L.stream())
        torch.cuda.synchronize()
        self.assertEqual(
            out,
            torch.tensor(
                [1e20, 0.0, 1.0, 1e20, 0.0, 1.0],
                device="cuda",
            ),
        )


if __name__ == "__main__":
    run_tests()
