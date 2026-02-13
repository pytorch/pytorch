# Owner(s): ["module: custom-operators"]
import torch
from torch import Tensor
from torch._library._out_variant import check_out_variant, to_out_variant
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOutVariant(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOutVariant", "FRAGMENT")  # noqa: TOR901

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_single_out(self):
        self.lib.define("single_return_arg(Tensor x, Tensor y) -> Tensor")
        self.lib.define(
            "single_return_arg.out(Tensor x, Tensor y, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        check_out_variant(
            torch.ops._TestOutVariant.single_return_arg.default,
            torch.ops._TestOutVariant.single_return_arg.out,
        )

        self.lib.define("single_return_kwarg(Tensor x, Tensor y) -> Tensor")
        self.lib.define(
            "single_return_kwarg.out(Tensor x, Tensor y, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        check_out_variant(
            torch.ops._TestOutVariant.single_return_kwarg.default,
            torch.ops._TestOutVariant.single_return_kwarg.out,
        )

        self.lib.define("single_no_return(Tensor x, Tensor y) -> Tensor")
        self.lib.define(
            "single_no_return.out(Tensor x, Tensor y, Tensor(a!) result) -> ()",
            tags=[torch.Tag.out_variant],
        )

        check_out_variant(
            torch.ops._TestOutVariant.single_no_return.default,
            torch.ops._TestOutVariant.single_no_return.out,
        )

    def test_multiple_out(self):
        self.lib.define("multi_return(Tensor x, Tensor y) -> (Tensor, Tensor)")
        self.lib.define(
            "multi_return.out(Tensor x, Tensor y, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
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
            tags=[torch.Tag.out_variant],
        )
        self.lib.define(
            "overloaded_multi.scalar(Tensor x, float scale) -> (Tensor, Tensor)",
        )
        self.lib.define(
            "overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
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
        @torch.library.custom_op("_TestOutVariant::multi_overload_int", mutates_args=())
        def multi_overload_int(x: Tensor, n: int) -> Tensor:
            return x * n

        @torch.library.custom_op(
            "_TestOutVariant::multi_overload_float", mutates_args=()
        )
        def multi_overload_float(x: Tensor, n: float) -> Tensor:
            return x + n

        @torch.library.custom_op(
            "_TestOutVariant::multi_overload_int.out",
            mutates_args=["out"],
            tags=[torch.Tag.out_variant],
        )
        def multi_overload_int_out(x: Tensor, n: int, out: Tensor) -> None:
            return x * n

        @torch.library.custom_op(
            "_TestOutVariant::multi_overload_float.out",
            mutates_args=["out"],
            tags=[torch.Tag.out_variant],
        )
        def multi_overload_float_out(x: Tensor, n: float, out: Tensor) -> None:
            return x + n

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
            tags=[torch.Tag.out_variant],
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
            tags=[torch.Tag.out_variant],
        )

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.optional_mismatch.default,
                torch.ops._TestOutVariant.optional_mismatch.out,
            )

        self.lib.define("default_mismatch(Tensor x, int n=0) -> Tensor")
        self.lib.define(
            "default_mismatch.out(Tensor x, int n=1, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
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
        self.lib.define("bad_ret(Tensor x) -> Tensor")
        self.lib.define(
            "bad_ret.out(Tensor x, *, Tensor(a!) out) -> Tensor",
            tags=[torch.Tag.out_variant],
        )

        with self.assertRaisesRegex(RuntimeError, "invalid returns"):
            to_out_variant(torch.ops._TestOutVariant.bad_ret.default)

        self.lib.define("bad_alias_order(Tensor x) -> (Tensor, Tensor)")
        self.lib.define(
            "bad_alias_order.out(Tensor x, *, Tensor(b!) out1, Tensor(a!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
        )

        with self.assertRaisesRegex(RuntimeError, "invalid returns"):
            to_out_variant(torch.ops._TestOutVariant.bad_alias_order.default)

        self.lib.define("bad_num_ret(Tensor x) -> Tensor")
        self.lib.define(
            "bad_num_ret.out(Tensor x, *, Tensor(a!) result) -> (Tensor(a!), Tensor)",
            tags=[torch.Tag.out_variant],
        )

        with self.assertRaisesRegex(RuntimeError, "invalid returns"):
            to_out_variant(torch.ops._TestOutVariant.bad_num_ret.default)

    def test_compile_out_variant(self):
        self.lib.define("div(Tensor x, Tensor y) -> Tensor")
        self.lib.impl("div", lambda x, y: x / y, "CompositeExplicitAutograd")
        self.lib.impl("div", lambda x, y: torch.empty_like(x), "Meta")

        self.lib.define(
            "div.out(Tensor x, Tensor y, *, Tensor! result) -> ()",
            tags=[torch.Tag.out_variant],
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

        check_out_variant(
            torch.ops._TestOutVariant.div.default,
            torch.ops._TestOutVariant.div.out,
        )


if __name__ == "__main__":
    run_tests()
