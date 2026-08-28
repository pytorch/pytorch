# Owner(s): ["module: dsl-native-ops"]
#
# Structural conformance for the reduction trait protocol. No kernel launches -- what these guard
# is the protocol itself: a trait that implements only part of it folds the wrong thing silently
# rather than failing. With this in place the kernels can be tested per fold ORDER instead of per
# trait x order.
#
# SHAPE only. The protocol's LAW -- combine(leaf(a), leaf(b)) == reduce(reduce(init(), a), b),
# whose violation is what made norm fold raw values and return a plausible wrong number -- cannot
# be checked here: the trait methods are @cute.jit and reject host-side values ("only float is
# supported"), so it takes two real fold shapes to compare. It is pinned in
# test_inner_tree_order.test_tree_fold_matches_the_serial_fold_for_every_value_trait, which runs
# the serial and tree folds over the same input for every value trait.
#
# The traits module imports cutlass at module scope, so every import here is inside a test body
# and the suite is gated on the CuteDSL runtime being present -- importing at collection time
# fails the whole file on an image without it.

import inspect

from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@skipIfNoCuteDSL
class TestTraitProtocol(TestCase):
    def _traits(self):
        from torch._native.ops._cutedsl import traits as T

        found = {
            name: obj
            for name, obj in vars(T).items()
            if inspect.isclass(obj) and name.endswith("Ops")
        }
        # Without this, a rename would make every assertion below vacuously pass on an empty set.
        self.assertGreater(len(found), 5)
        return found

    def _make(self, trait):
        import cutlass

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


if __name__ == "__main__":
    run_tests()
