# Owner(s): ["module: custom-operators"]
import torch
from torch import Tensor
from torch._library._out_variant import check_out_variant, to_out_variant
from torch._library.utils import is_out
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOutVariant(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOutVariant", "FRAGMENT")  # noqa: TOR901

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_single_out(self):
        self.lib.define("single_return_kwarg(Tensor x, Tensor y) -> Tensor")
        self.lib.define(
            "single_return_kwarg.out(Tensor x, Tensor y, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        check_out_variant(
            torch.ops._TestOutVariant.single_return_kwarg.default,
            torch.ops._TestOutVariant.single_return_kwarg.out,
        )

        self.lib.define("single_no_return(Tensor x, Tensor y) -> Tensor")
        with self.assertRaisesRegex(ValueError, "must return all mutable arguments"):
            self.lib.define(
                "single_no_return.out(Tensor x, Tensor y, *, Tensor(a!) result) -> ()",
                tags=[torch.Tag.out],
            )

    def test_multiple_out(self):
        self.lib.define("multi_return(Tensor x, Tensor y) -> (Tensor, Tensor)")
        self.lib.define(
            "multi_return.out(Tensor x, Tensor y, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out],
        )

        check_out_variant(
            torch.ops._TestOutVariant.multi_return.default,
            torch.ops._TestOutVariant.multi_return.out,
        )

    def test_multi_out_overload(self):
        self.lib.define(
            "overloaded_multi.Tensor(Tensor x, Tensor scale) -> (Tensor, Tensor)",
        )
        self.lib.define(
            "overloaded_multi.Tensor_out(Tensor x, Tensor scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out],
        )
        self.lib.define(
            "overloaded_multi.scalar(Tensor x, float scale) -> (Tensor, Tensor)",
        )
        self.lib.define(
            "overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out],
        )

        check_out_variant(
            torch.ops._TestOutVariant.overloaded_multi.Tensor,
            torch.ops._TestOutVariant.overloaded_multi.Tensor_out,
        )
        check_out_variant(
            torch.ops._TestOutVariant.overloaded_multi.scalar,
            torch.ops._TestOutVariant.overloaded_multi.scalar_out,
        )

    def test_multiple_overloads_with_out_variants(self):
        self.lib.define("multi_overload_int(Tensor x, int n) -> Tensor")
        self.lib.define(
            "multi_overload_int.out(Tensor x, int n, *, Tensor(a!) out) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        self.lib.define("multi_overload_float(Tensor x, float n) -> Tensor")
        self.lib.define(
            "multi_overload_float.out(Tensor x, float n, *, Tensor(a!) out) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        check_out_variant(
            torch.ops._TestOutVariant.multi_overload_int.default,
            torch.ops._TestOutVariant.multi_overload_int.out,
        )
        check_out_variant(
            torch.ops._TestOutVariant.multi_overload_float.default,
            torch.ops._TestOutVariant.multi_overload_float.out,
        )

    def test_no_out_variant_registered(self):
        self.lib.define("no_out_op(Tensor x) -> Tensor")

        out_op = to_out_variant(torch.ops._TestOutVariant.no_out_op.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.no_out_op.default,
                None,
            )

    def test_out_variant_missing_tag(self):
        self.lib.define("untagged(Tensor x) -> Tensor")
        self.lib.define(
            "untagged.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
        )

        out_op = to_out_variant(torch.ops._TestOutVariant.untagged.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.untagged.default,
                torch.ops._TestOutVariant.untagged.out,
            )

    def test_out_variant_signature_mismatch(self):
        self.lib.define("sig_mismatch_op(Tensor x, Tensor y) -> Tensor")
        self.lib.define(
            "sig_mismatch_op.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        out_op = to_out_variant(torch.ops._TestOutVariant.sig_mismatch_op.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.sig_mismatch_op.default,
                torch.ops._TestOutVariant.sig_mismatch_op.out,
            )

        self.lib.define("optional_mismatch(Tensor x) -> Tensor")
        self.lib.define(
            "optional_mismatch.out(Tensor? x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.optional_mismatch.default,
                torch.ops._TestOutVariant.optional_mismatch.out,
            )

        self.lib.define("default_mismatch(Tensor x, int n=0) -> Tensor")
        self.lib.define(
            "default_mismatch.out(Tensor x, int n=1, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.default_mismatch.default,
                torch.ops._TestOutVariant.default_mismatch.out,
            )

    def test_mutable_op_not_functional(self):
        self.lib.define("mutating_op(Tensor(a!) x, Tensor y) -> Tensor")

        with self.assertRaisesRegex(RuntimeError, "schema is not functional"):
            to_out_variant(torch.ops._TestOutVariant.mutating_op.default)

    def test_trailing_underscore_not_functional(self):
        self.lib.define("my_func_(Tensor x) -> Tensor")

        with self.assertRaisesRegex(RuntimeError, "schema is not functional"):
            to_out_variant(torch.ops._TestOutVariant.my_func_.default)

    def test_out_variant_bad_return(self):
        # Validation now happens at define-time when the out tag is present
        self.lib.define("bad_ret(Tensor x) -> Tensor")
        with self.assertRaisesRegex(ValueError, "must alias mutable arg"):
            self.lib.define(
                "bad_ret.out(Tensor x, *, Tensor(a!) out) -> Tensor",
                tags=[torch.Tag.out],
            )

        self.lib.define("bad_alias_order(Tensor x) -> (Tensor, Tensor)")
        with self.assertRaisesRegex(ValueError, "must alias mutable arg"):
            self.lib.define(
                "bad_alias_order.out(Tensor x, *, Tensor(b!) out1, Tensor(a!) out2) -> (Tensor(a!), Tensor(b!))",
                tags=[torch.Tag.out],
            )

        self.lib.define("bad_num_ret(Tensor x) -> Tensor")
        with self.assertRaisesRegex(ValueError, "must return all mutable arguments"):
            self.lib.define(
                "bad_num_ret.out(Tensor x, *, Tensor(a!) result) -> (Tensor(a!), Tensor)",
                tags=[torch.Tag.out],
            )

    def test_compile_out_variant(self):
        # TODO: use the out tag here once torch.compile supports custom ops
        # with aliased returns.
        self.lib.define("div(Tensor x, Tensor y) -> Tensor")
        self.lib.impl("div", lambda x, y: x / y, "CompositeExplicitAutograd")
        self.lib.impl("div", lambda x, y: torch.empty_like(x), "Meta")

        self.lib.define(
            "div.out(Tensor x, Tensor y, *, Tensor! result) -> ()",
        )

        def div_out_impl(x: Tensor, y: Tensor, *, result: Tensor) -> None:
            result.copy_(x / y)

        self.lib.impl("div.out", div_out_impl, "CompositeExplicitAutograd")
        self.lib.impl("div.out", lambda x, y, *, result: None, "Meta")

        def fn(x, y, out):
            torch.ops._TestOutVariant.div.out(x, y, result=out)
            return out

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        out = torch.empty(3, 4)

        compiled_fn = torch.compile(fn)
        compiled_fn(x, y, out)
        self.assertEqual(out, x / y)

    def test_is_out(self):
        self.lib.define("is_out_func(Tensor x) -> Tensor")
        self.lib.define(
            "is_out_func.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out],
        )
        self.assertTrue(is_out(torch.ops._TestOutVariant.is_out_func.out))
        self.assertFalse(is_out(torch.ops._TestOutVariant.is_out_func.default))

    def test_is_out_native(self):
        # Hand-written out= op (defined in native_functions.yaml)
        self.assertTrue(is_out(torch.ops.aten.abs.out))
        self.assertFalse(is_out(torch.ops.aten.abs.default))
        # Auto-generated out= op (via autogen directive)
        self.assertTrue(is_out(torch.ops.aten.randn_like.out))
        self.assertFalse(is_out(torch.ops.aten.randn_like.default))
        # In-place op (not an out op)
        self.assertFalse(is_out(torch.ops.aten.abs_.default))
        # Mutable op (has mutable positional args, not an out op)
        self.assertFalse(is_out(torch.ops.aten._native_batch_norm_legit.default))

    def test_define_out_tag_no_mutable_args(self):
        with self.assertRaisesRegex(ValueError, "at least one mutable argument"):
            self.lib.define(
                "no_mutable(Tensor x) -> Tensor",
                tags=[torch.Tag.out],
            )

    def test_define_out_tag_mutable_positional_arg(self):
        with self.assertRaisesRegex(ValueError, "keyword-only"):
            self.lib.define(
                "mut_pos(Tensor x, Tensor(a!) out) -> Tensor(a!)",
                tags=[torch.Tag.out],
            )

    def test_define_out_tag_optional_mutable_arg(self):
        with self.assertRaisesRegex(
            ValueError, "only supports Tensor mutable arguments"
        ):
            self.lib.define(
                "mut_opt(Tensor x, *, Tensor(a!)? out=None) -> Tensor(a!)?",
                tags=[torch.Tag.out],
            )

    def test_define_out_tag_tensorlist_mutable_arg(self):
        with self.assertRaisesRegex(
            ValueError, "only supports Tensor mutable arguments"
        ):
            self.lib.define(
                "mut_list(Tensor x, *, Tensor(a!)[] out) -> ()",
                tags=[torch.Tag.out],
            )

    def test_define_out_tag_return_not_mutable_alias(self):
        with self.assertRaisesRegex(ValueError, "mutable alias"):
            self.lib.define(
                "bad_ret_alias(Tensor x, *, Tensor(a!) out) -> Tensor(a)",
                tags=[torch.Tag.out],
            )


if __name__ == "__main__":
    run_tests()
