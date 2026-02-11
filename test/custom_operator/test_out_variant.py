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
        self.lib.define(
            "single_return(Tensor x, Tensor y) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::single_return", "CompositeExplicitAutograd"
        )
        def single_return_impl(x: Tensor, y: Tensor) -> Tensor:
            return x + y

        self.lib.define(
            "single_return.out(Tensor x, Tensor y, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::single_return.out", "CompositeExplicitAutograd"
        )
        def single_return_out_impl(x: Tensor, y: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(torch.ops._TestOutVariant.single_return.default(x, y))
            return result

        check_out_variant(
            torch.ops._TestOutVariant.single_return.default,
            torch.ops._TestOutVariant.single_return.out,
        )

    def test_multiple_out(self):
        self.lib.define(
            "multi_return(Tensor x, Tensor y) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::multi_return", "CompositeExplicitAutograd"
        )
        def multi_return_impl(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return x * y, x + y

        self.lib.define(
            "multi_return.out(Tensor x, Tensor y, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::multi_return.out", "CompositeExplicitAutograd"
        )
        def multi_return_out_impl(
            x: Tensor, y: Tensor, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            result1, result2 = torch.ops._TestOutVariant.multi_return.default(x, y)
            out1.copy_(result1)
            out2.copy_(result2)
            return out1, out2

        check_out_variant(
            torch.ops._TestOutVariant.multi_return.default,
            torch.ops._TestOutVariant.multi_return.out,
        )

    def test_multi_out_overload(self):
        self.lib.define(
            "overloaded_multi.Tensor(Tensor x, Tensor scale) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded_multi.Tensor", "CompositeExplicitAutograd"
        )
        def overloaded_multi_tensor_impl(
            x: Tensor, scale: Tensor
        ) -> tuple[Tensor, Tensor]:
            return x * scale, x + scale

        self.lib.define(
            "overloaded_multi.Tensor_out(Tensor x, Tensor scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded_multi.Tensor_out", "CompositeExplicitAutograd"
        )
        def overloaded_multi_tensor_out_impl(
            x: Tensor, scale: Tensor, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            result1, result2 = torch.ops._TestOutVariant.overloaded_multi.Tensor(
                x, scale
            )
            out1.copy_(result1)
            out2.copy_(result2)
            return out1, out2

        self.lib.define(
            "overloaded_multi.scalar(Tensor x, float scale) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded_multi.scalar", "CompositeExplicitAutograd"
        )
        def overloaded_multi_scalar_impl(
            x: Tensor, scale: float
        ) -> tuple[Tensor, Tensor]:
            return x * scale, x + scale

        self.lib.define(
            "overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded_multi.scalar_out", "CompositeExplicitAutograd"
        )
        def overloaded_multi_scalar_out_impl(
            x: Tensor, scale: float, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            result1, result2 = torch.ops._TestOutVariant.overloaded_multi.scalar(
                x, scale
            )
            out1.copy_(result1)
            out2.copy_(result2)
            return out1, out2

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

        self.lib.define(
            "multi_overload_int.out(Tensor x, SymInt n, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )
        self.lib.define(
            "multi_overload_float.out(Tensor x, float n, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::multi_overload_int.out", "CompositeExplicitAutograd"
        )
        def multi_overload_int_out_impl(x: Tensor, n: int, *, result: Tensor) -> Tensor:
            result.copy_(torch.ops._TestOutVariant.multi_overload_int.default(x, n))
            return result

        @torch.library.impl(
            "_TestOutVariant::multi_overload_float.out", "CompositeExplicitAutograd"
        )
        def multi_overload_float_out_impl(
            x: Tensor, n: float, *, result: Tensor
        ) -> Tensor:
            result.copy_(torch.ops._TestOutVariant.multi_overload_float.default(x, n))
            return result

        check_out_variant(
            torch.ops._TestOutVariant.multi_overload_int.default,
            torch.ops._TestOutVariant.multi_overload_int.out,
        )
        check_out_variant(
            torch.ops._TestOutVariant.multi_overload_float.default,
            torch.ops._TestOutVariant.multi_overload_float.out,
        )

    def test_no_out_variant_registered(self):
        self.lib.define(
            "no_out_op(Tensor x) -> Tensor",
        )

        @torch.library.impl("_TestOutVariant::no_out_op", "CompositeExplicitAutograd")
        def no_out_op_impl(x: Tensor) -> Tensor:
            return x * 2

        out_op = to_out_variant(torch.ops._TestOutVariant.no_out_op.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.no_out_op.default,
                None,
            )

    def test_out_variant_missing_tag(self):
        self.lib.define("untagged(Tensor x) -> Tensor")

        @torch.library.impl("_TestOutVariant::untagged", "CompositeExplicitAutograd")
        def untagged_impl(x: Tensor) -> Tensor:
            return x.sin()

        self.lib.define(
            "untagged.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
        )

        @torch.library.impl(
            "_TestOutVariant::untagged.out", "CompositeExplicitAutograd"
        )
        def untagged_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x.sin())
            return result

        out_op = to_out_variant(torch.ops._TestOutVariant.untagged.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.untagged.default,
                torch.ops._TestOutVariant.untagged.out,
            )

    def test_out_variant_signature_mismatch(self):
        self.lib.define(
            "sig_mismatch_op(Tensor x, Tensor y) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::sig_mismatch_op", "CompositeExplicitAutograd"
        )
        def sig_mismatch_op_impl(x: Tensor, y: Tensor) -> Tensor:
            return x + y

        self.lib.define(
            "sig_mismatch_op.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::sig_mismatch_op.out", "CompositeExplicitAutograd"
        )
        def sig_mismatch_op_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x * 2)
            return result

        out_op = to_out_variant(torch.ops._TestOutVariant.sig_mismatch_op.default)
        self.assertIsNone(out_op)

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.sig_mismatch_op.default,
                torch.ops._TestOutVariant.sig_mismatch_op.out,
            )

    def test_op_with_out_suffix_in_name(self):
        # Ops with _out in their name are functional (no write args, no trailing _)
        # and simply have no out variant registered.
        self.lib.define(
            "my_custom_out(Tensor x) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::my_custom_out", "CompositeExplicitAutograd"
        )
        def my_custom_out_impl(x: Tensor) -> Tensor:
            return x * 2

        out_op = to_out_variant(torch.ops._TestOutVariant.my_custom_out.default)
        self.assertIsNone(out_op)

    def test_mutable_op_not_functional(self):
        self.lib.define(
            "mutating_op(Tensor(a!) x, Tensor y) -> Tensor",
        )

        @torch.library.impl("_TestOutVariant::mutating_op", "CompositeExplicitAutograd")
        def mutating_op_impl(x: Tensor, y: Tensor) -> Tensor:
            x.add_(y)
            return x.clone()

        with self.assertRaisesRegex(RuntimeError, "Failed to find out variant"):
            to_out_variant(torch.ops._TestOutVariant.mutating_op.default)

    def test_trailing_underscore_not_functional(self):
        # Operators with trailing underscore are treated as non-functional
        # (inplace convention), even if they have no mutable args.
        self.lib.define(
            "my_func_(Tensor x) -> Tensor",
        )

        @torch.library.impl("_TestOutVariant::my_func_", "CompositeExplicitAutograd")
        def my_func__impl(x: Tensor) -> Tensor:
            return x * 2

        with self.assertRaisesRegex(RuntimeError, "not functional"):
            to_out_variant(torch.ops._TestOutVariant.my_func_.default)

    def test_out_variant_bad_return(self):
        # Out variant returns a fresh Tensor (no alias) instead of void
        # or the mutable args
        self.lib.define("bad_ret(Tensor x) -> Tensor")

        @torch.library.impl("_TestOutVariant::bad_ret", "CompositeExplicitAutograd")
        def bad_ret_impl(x: Tensor) -> Tensor:
            return x.sin()

        self.lib.define(
            "bad_ret.out(Tensor x, *, Tensor(a!) out) -> Tensor",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl("_TestOutVariant::bad_ret.out", "CompositeExplicitAutograd")
        def bad_ret_out_impl(x: Tensor, *, out: Tensor) -> Tensor:
            out.copy_(x.sin())
            return x.cos()

        with self.assertRaisesRegex(RuntimeError, "invalid returns"):
            to_out_variant(torch.ops._TestOutVariant.bad_ret.default)

    def test_wrong_number_of_out_tensors(self):
        # Functional returns 2 tensors but out variant only has 1 out tensor.
        self.lib.define(
            "wrong_out_count(Tensor x) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::wrong_out_count", "CompositeExplicitAutograd"
        )
        def wrong_out_count_impl(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x + 1

        self.lib.define(
            "wrong_out_count.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::wrong_out_count.out", "CompositeExplicitAutograd"
        )
        def wrong_out_count_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x * 2)
            return result

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.wrong_out_count.default,
                torch.ops._TestOutVariant.wrong_out_count.out,
            )

    def test_optional_tensor_mismatch(self):
        # Functional takes Tensor, out variant takes Tensor? - signatures don't match
        self.lib.define(
            "optional_mismatch(Tensor x) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::optional_mismatch", "CompositeExplicitAutograd"
        )
        def optional_mismatch_impl(x: Tensor) -> Tensor:
            return x * 2

        self.lib.define(
            "optional_mismatch.out(Tensor? x, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::optional_mismatch.out", "CompositeExplicitAutograd"
        )
        def optional_mismatch_out_impl(x: Tensor | None, *, result: Tensor) -> Tensor:
            if x is not None:
                result.copy_(x * 2)
            return result

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.optional_mismatch.default,
                torch.ops._TestOutVariant.optional_mismatch.out,
            )

    def test_default_value_mismatch(self):
        # Functional has n=0, out variant has n=1 - signatures don't match
        self.lib.define(
            "default_mismatch(Tensor x, int n=0) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::default_mismatch", "CompositeExplicitAutograd"
        )
        def default_mismatch_impl(x: Tensor, n: int = 0) -> Tensor:
            return x * n

        self.lib.define(
            "default_mismatch.out(Tensor x, int n=1, *, Tensor(a!) result) -> Tensor(a!)",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::default_mismatch.out", "CompositeExplicitAutograd"
        )
        def default_mismatch_out_impl(
            x: Tensor, n: int = 1, *, result: Tensor
        ) -> Tensor:
            result.copy_(x * n)
            return result

        with self.assertRaisesRegex(AssertionError, "We did not find an out variant"):
            check_out_variant(
                torch.ops._TestOutVariant.default_mismatch.default,
                torch.ops._TestOutVariant.default_mismatch.out,
            )

    def test_compile_out_variant(self):
        self.lib.define("div(Tensor x, Tensor y) -> Tensor")

        @torch.library.impl("_TestOutVariant::div", "CompositeExplicitAutograd")
        def div_impl(x: Tensor, y: Tensor) -> Tensor:
            return x / y

        @torch.library.impl("_TestOutVariant::div", "Meta")
        def div_fake(x: Tensor, y: Tensor) -> Tensor:
            return torch.empty_like(x)

        self.lib.define(
            "div.out(Tensor x, Tensor y, *, Tensor! result) -> ()",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl("_TestOutVariant::div.out", "CompositeExplicitAutograd")
        def div_out_impl(x: Tensor, y: Tensor, *, result: Tensor) -> None:
            result.copy_(x / y)
            return
            return result

        @torch.library.impl("_TestOutVariant::div.out", "Meta")
        def div_out_fake(x: Tensor, y: Tensor, *, result: Tensor) -> None:
            return
            return result

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

    def test_out_variant_tag_single_output(self):
        @torch.library.custom_op(
            "_TestOutVariant::sin",
            mutates_args=(),
            tags=[torch.Tag.out_variant],
        )
        def sin_impl(x: Tensor) -> Tensor:
            return x.sin()

        @sin_impl.register_fake
        def sin_fake(x: Tensor) -> Tensor:
            return torch.empty_like(x)

        self.lib.define(
            "sin.out(Tensor x, *, Tensor(a!) out) -> ()",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl("_TestOutVariant::sin.out", "CompositeExplicitAutograd")
        def sin_out_impl(x: Tensor, *, out: Tensor) -> None:
            out.copy_(x.sin())

        @torch.library.impl("_TestOutVariant::sin.out", "Meta")
        def sin_out_fake(x: Tensor, *, out: Tensor) -> None:
            pass

        def fn(x, out):
            torch.ops._TestOutVariant.sin.out(x, out=out)
            return out

        x = torch.randn(3, 4)
        out = torch.empty(3, 4)

        compiled_fn = torch.compile(fn)
        compiled_fn(x, out)
        self.assertEqual(out, x.sin())

    def test_out_variant_tag_multi_output(self):
        self.lib.define(
            "sincos.out(Tensor x, *, Tensor(a!) sin_out, Tensor(b!) cos_out) -> ()",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl("_TestOutVariant::sincos.out", "CompositeExplicitAutograd")
        def sincos_out_impl(x: Tensor, *, sin_out: Tensor, cos_out: Tensor) -> None:
            sin_out.copy_(x.sin())
            cos_out.copy_(x.cos())

        @torch.library.impl("_TestOutVariant::sincos.out", "Meta")
        def sincos_out_fake(x: Tensor, *, sin_out: Tensor, cos_out: Tensor) -> None:
            pass

        def fn(x, sin_out, cos_out):
            torch.ops._TestOutVariant.sincos.out(x, sin_out=sin_out, cos_out=cos_out)
            return sin_out, cos_out

        x = torch.randn(3, 4)
        sin_out = torch.empty(3, 4)
        cos_out = torch.empty(3, 4)

        compiled_fn = torch.compile(fn)
        compiled_fn(x, sin_out, cos_out)
        self.assertEqual(sin_out, x.sin())
        self.assertEqual(cos_out, x.cos())

    def test_out_variant_tag_no_data_dependency(self):
        self.lib.define(
            "fill_ones.out(Tensor x, *, Tensor(a!) out) -> ()",
            tags=[torch.Tag.out_variant],
        )

        @torch.library.impl(
            "_TestOutVariant::fill_ones.out", "CompositeExplicitAutograd"
        )
        def fill_ones_out_impl(x: Tensor, *, out: Tensor) -> None:
            out.copy_(torch.ones_like(x))

        @torch.library.impl("_TestOutVariant::fill_ones.out", "Meta")
        def fill_ones_out_fake(x: Tensor, *, out: Tensor) -> None:
            pass

        from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        x = torch.randn(3, 4)
        out = torch.randn(3, 4)
        op = torch.ops._TestOutVariant.fill_ones.out

        # With out_variant tag, the out arg should be allocated fresh via
        # empty_like rather than cloned. Verify the result doesn't depend
        # on the input out buffer values.
        result = auto_functionalized_dense(op, x=x, out=out)
        # result is (None, out_new) - None because return is ()
        out_new = result[1]
        self.assertEqual(out_new, torch.ones_like(x))


if __name__ == "__main__":
    run_tests()
