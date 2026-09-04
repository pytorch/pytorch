# Owner(s): ["module: dsl-native-ops"]
#
# Structural conformance for the reduction trait protocol -- what these guard is the protocol
# itself: a trait that implements only part of it folds the wrong thing silently rather than
# failing. With this in place the kernels can be tested per fold ORDER instead of per trait x order.
#
# SHAPE, with one numeric exception. The protocol's LAW -- combine(leaf(a), leaf(b)) ==
# reduce(reduce(init(), a), b), whose violation is what made norm fold raw values and return a
# plausible wrong number -- cannot be checked here: it takes two real fold shapes to compare, and
# only one exists at this commit. It is pinned once the second lands, by the inner-tree order's
# test_tree_fold_matches_the_serial_fold_for_every_value_trait. The exception is the var/std
# divisor clamp: the trait methods are @cute.jit and reject host-side values ("only float is
# supported"), but a one-thread probe kernel needs no reduction kernel, so that claim is asserted
# here rather than two commits later.
#
import inspect
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


# The traits module imports cutlass at module scope, so the guard has to precede the import rather
# than decorate the class: on an image without the runtime, importing at collection time fails the
# whole file instead of skipping it. sys.exit keeps a direct `python test_traits.py` a success.
if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass
import cutlass.cute as cute

from torch._native.ops._cutedsl import launch as _L, traits as T


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
        # THREE value methods, and the split between them is what lets any trait ride any fold
        # order: leaf turns one element into an accumulator, combine merges two accumulators,
        # reduce is the serial update. A tree fold calls leaf then combine, so a trait carrying
        # its per-element transform (|x|**p for a norm, a 0/1 flag for all/any) only in reduce
        # would fold RAW values -- a wrong answer, not an error.
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
        # nfields sizes every partials buffer and accumulator tuple; fdtypes types them. A
        # disagreement is a mis-sized gmem allocation, not a type error, so assert it here.
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

    @unittest.skipUnless(TEST_CUDA, "CUDA required")
    def test_welford_divisor_clamps_at_zero(self):
        # `correction >= n` must divide by ZERO -- +inf, which is what aten returns -- never by a
        # negative number, which returned a NEGATIVE variance. _welford_denom is @cute.jit and so
        # rejects host-side values, but it needs no reduction kernel either: a one-thread probe
        # evaluates it directly, which keeps the claim asserted where the clamp is defined.
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
        # n itself and beyond it: both clamp. The correction < n case proves the clamp is not
        # swallowing the ordinary divisor.
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
                # `correction` is a Constexpr, baked at compile time, so the compiled callable takes
                # only the real operand, the runtime scalar and the stream.
                fn(out, nf, _L.stream())
                torch.cuda.synchronize()
                self.assertEqual(out.item(), want)


if __name__ == "__main__":
    run_tests()
