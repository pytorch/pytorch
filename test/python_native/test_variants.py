# Owner(s): ["module: dsl-native-ops"]
#
# Variant selection: which of an override's implementations runs. DSL-free --
# the store is plain Python and the tests never dispatch an op.

import torch.backends.python_native as pn
from torch._native import variants
from torch.testing._internal.common_utils import run_tests, TestCase


OP = "torch_nn::_linear_cross_entropy_batch_chunked"
OTHER = "aten::scatter_add_"


class TestVariants(TestCase):
    def tearDown(self):
        for op in (OP, OTHER):
            variants.set_variant(op, None)
        super().tearDown()

    def test_default_until_selected(self):
        self.assertEqual(variants.get_variant(OP, "kernel"), "kernel")
        variants.set_variant(OP, variants.PASSTHROUGH)
        self.assertEqual(variants.get_variant(OP, "kernel"), variants.PASSTHROUGH)
        variants.set_variant(OP, None)
        self.assertEqual(variants.get_variant(OP, "kernel"), "kernel")

    def test_selection_is_per_op(self):
        variants.set_variant(OP, variants.PASSTHROUGH)
        self.assertEqual(variants.get_variant(OTHER, "kernel"), "kernel")

    def test_context_manager_restores(self):
        with variants.variant(OP, variants.PASSTHROUGH):
            self.assertEqual(variants.get_variant(OP, "kernel"), variants.PASSTHROUGH)
        self.assertEqual(variants.get_variant(OP, "kernel"), "kernel")

    def test_context_manager_restores_a_previous_selection(self):
        variants.set_variant(OP, "first")
        with variants.variant(OP, "second"):
            self.assertEqual(variants.get_variant(OP, "kernel"), "second")
        self.assertEqual(variants.get_variant(OP, "kernel"), "first")

    def test_context_manager_restores_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with variants.variant(OP, variants.PASSTHROUGH):
                raise RuntimeError("boom")
        self.assertEqual(variants.get_variant(OP, "kernel"), "kernel")

    def test_backends_surface(self):
        pn.set_override_variant(OP, variants.PASSTHROUGH)
        self.assertEqual(variants.get_variant(OP, "kernel"), variants.PASSTHROUGH)
        pn.set_override_variant(OP, None)
        with pn.override_variant(OP, variants.PASSTHROUGH):
            self.assertEqual(variants.get_variant(OP, "kernel"), variants.PASSTHROUGH)
        self.assertEqual(variants.get_variant(OP, "kernel"), "kernel")

    def test_env_parsing(self):
        """The env var seeds the selection at import for one-process-per-config
        harness runs, so a malformed entry must be dropped rather than raised
        on -- raising would make `import torch` fail."""
        self.assertEqual(
            variants._parse_env("ns::op=a;ns2::op2=b"),
            {"ns::op": "a", "ns2::op2": "b"},
        )
        self.assertEqual(variants._parse_env(" ns::op = a ; ; "), {"ns::op": "a"})
        self.assertEqual(variants._parse_env(None), {})
        for malformed in ("nocolons=a", "ns::op", "ns::op=", "=a"):
            self.assertEqual(variants._parse_env(malformed), {}, malformed)

    def test_unknown_variant_names_the_declared_ones(self):
        from torch._native.ops.linear_cross_entropy import cutedsl_impl

        op_symbol = "_linear_cross_entropy_batch_chunked"
        dispatch = {row[0]: row[2] for row in cutedsl_impl._OVERRIDES}[op_symbol]
        with variants.variant(OP, "no_such_variant"):
            with self.assertRaisesRegex(ValueError, "declares .*passthrough"):
                dispatch()


if __name__ == "__main__":
    run_tests()
