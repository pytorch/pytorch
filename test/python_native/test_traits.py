# Owner(s): ["module: dsl-native-ops"]
#
# Structural conformance for the reduction trait protocol. Host-only -- no kernel launches --
# because what these guard is the protocol itself: a trait that implements only part of it folds
# the wrong thing silently rather than failing. With this in place the kernels can be tested per
# fold ORDER instead of per trait x order.

import inspect

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _traits():
    from torch._native.ops._cutedsl import traits as T

    return {
        name: obj
        for name, obj in vars(T).items()
        if inspect.isclass(obj) and name.endswith("Ops")
    }


def _make(trait):
    import cutlass

    params = inspect.signature(trait.__init__).parameters
    kwargs = {"acc": cutlass.Float32}
    if "p" in params and params["p"].default is inspect.Parameter.empty:
        kwargs["p"] = 2.0  # a norm has no default exponent
    return trait(**kwargs)


class TestTraitProtocol(TestCase):
    def test_discovery_finds_the_traits(self):
        # Without this, a rename would make every test below vacuously pass on an empty set.
        self.assertGreater(len(_traits()), 5)

    @parametrize("name", sorted(_traits()))
    def test_trait_implements_the_protocol(self, name):
        # THREE value methods, and the split between them is what lets any trait ride any fold
        # order: leaf turns one element into an accumulator, combine merges two accumulators,
        # reduce is the serial update. A tree fold calls leaf then combine, so a trait carrying
        # its per-element transform (|x|**p for a norm, a 0/1 flag for all/any) only in reduce
        # would fold RAW values -- which is a wrong answer, not an error.
        trait = _traits()[name]
        for method in ("init", "leaf", "combine", "reduce", "project"):
            self.assertTrue(
                callable(getattr(trait, method, None)), f"{name}.{method} is missing"
            )
        self.assertEqual(
            sorted(inspect.signature(trait.leaf).parameters),
            ["idx", "self", "val"],
            f"{name}.leaf must take (val, idx)",
        )

    @parametrize("name", sorted(_traits()))
    def test_field_count_matches_field_dtypes(self, name):
        # nfields sizes every partials buffer and accumulator tuple; fdtypes types them. A
        # disagreement is a mis-sized gmem allocation, not a type error, so assert it here.
        trait = _make(_traits()[name])
        self.assertEqual(len(trait.fdtypes), trait.nfields)
        self.assertGreaterEqual(trait.nfields, 1)

    @parametrize("name", sorted(_traits()))
    def test_leaf_returns_one_accumulator(self, name):
        # leaf must return exactly the accumulator tuple combine expects, so the two compose.
        trait = _make(_traits()[name])
        self.assertEqual(len(trait.init()), trait.nfields)


instantiate_parametrized_tests(TestTraitProtocol)


if __name__ == "__main__":
    run_tests()
