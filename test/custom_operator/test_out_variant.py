# Owner(s): ["module: custom-operators"]
import torch
from torch import Tensor
from torch._library._out_variant import check_out_variant, to_out_variant
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOutVariant(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOutVariant", "FRAGMENT")

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_single_out_variant(self):
        self.lib.define(
            "single_return(Tensor x) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::single_return", "CompositeExplicitAutograd"
        )
        def single_return_impl(x: Tensor) -> Tensor:
            return x * 2

        self.lib.define(
            "single_return.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::single_return.out", "CompositeExplicitAutograd"
        )
        def single_return_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x * 2)
            return result

        x = torch.randn(3, 4)
        expected = torch.ops._TestOutVariant.single_return(x)

        self.assertTrue("out" in torch.ops._TestOutVariant.single_return.overloads())
        self.assertEqual(
            str(torch.ops._TestOutVariant.single_return.out._schema),
            "_TestOutVariant::single_return.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)",
        )

        out = torch.empty(3, 4)
        torch.ops._TestOutVariant.single_return.out(x, result=out)
        self.assertEqual(out, expected)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.single_return.default,
                torch.ops._TestOutVariant.single_return.out,
            )
        )

    def test_multiple_out_variant(self):
        self.lib.define(
            "multi_return(Tensor x) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::multi_return", "CompositeExplicitAutograd"
        )
        def multi_return_impl(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x * 3

        self.lib.define(
            "multi_return.out(Tensor x, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
        )

        @torch.library.impl(
            "_TestOutVariant::multi_return.out", "CompositeExplicitAutograd"
        )
        def multi_return_out_impl(
            x: Tensor, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            out1.copy_(x * 2)
            out2.copy_(x * 3)
            return out1, out2

        x = torch.randn(3, 4)
        expected1, expected2 = torch.ops._TestOutVariant.multi_return(x)

        self.assertTrue("out" in torch.ops._TestOutVariant.multi_return.overloads())
        self.assertEqual(
            str(torch.ops._TestOutVariant.multi_return.out._schema),
            "_TestOutVariant::multi_return.out(Tensor x, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
        )

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestOutVariant.multi_return.out(x, out1=out1, out2=out2)
        self.assertEqual(out1, expected1)
        self.assertEqual(out2, expected2)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.multi_return.default,
                torch.ops._TestOutVariant.multi_return.out,
            )
        )

    def test_single_out_variant_with_multiple_inputs(self):
        self.lib.define(
            "single_out_multi_inputs(Tensor x, Tensor y) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::single_out_multi_inputs", "CompositeExplicitAutograd"
        )
        def single_out_multi_inputs_impl(x: Tensor, y: Tensor) -> Tensor:
            return x + y

        self.lib.define(
            "single_out_multi_inputs.out(Tensor x, Tensor y, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::single_out_multi_inputs.out", "CompositeExplicitAutograd"
        )
        def single_out_multi_inputs_out_impl(
            x: Tensor, y: Tensor, *, result: Tensor
        ) -> Tensor:
            result.copy_(x + y)
            return result

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        expected = torch.ops._TestOutVariant.single_out_multi_inputs(x, y)

        self.assertTrue(
            "out" in torch.ops._TestOutVariant.single_out_multi_inputs.overloads()
        )

        out = torch.empty(3, 4)
        torch.ops._TestOutVariant.single_out_multi_inputs.out(x, y, result=out)
        self.assertEqual(out, expected)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.single_out_multi_inputs.default,
                torch.ops._TestOutVariant.single_out_multi_inputs.out,
            )
        )

    def test_multiple_out_variant_with_multiple_inputs(self):
        self.lib.define(
            "multi_out_multi_inputs(Tensor x, Tensor y) -> (Tensor, Tensor)",
        )

        @torch.library.impl(
            "_TestOutVariant::multi_out_multi_inputs", "CompositeExplicitAutograd"
        )
        def multi_out_multi_inputs_impl(
            x: Tensor, y: Tensor
        ) -> tuple[Tensor, Tensor]:
            return x * y, x + y

        self.lib.define(
            "multi_out_multi_inputs.out(Tensor x, Tensor y, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
        )

        @torch.library.impl(
            "_TestOutVariant::multi_out_multi_inputs.out", "CompositeExplicitAutograd"
        )
        def multi_out_multi_inputs_out_impl(
            x: Tensor, y: Tensor, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            out1.copy_(x * y)
            out2.copy_(x + y)
            return out1, out2

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        expected1, expected2 = torch.ops._TestOutVariant.multi_out_multi_inputs(x, y)

        self.assertTrue(
            "out" in torch.ops._TestOutVariant.multi_out_multi_inputs.overloads()
        )

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestOutVariant.multi_out_multi_inputs.out(x, y, out1=out1, out2=out2)
        self.assertEqual(out1, expected1)
        self.assertEqual(out2, expected2)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.multi_out_multi_inputs.default,
                torch.ops._TestOutVariant.multi_out_multi_inputs.out,
            )
        )

    def test_overload_single_out_variant(self):
        # Define an op with a .int overload and its .int_out variant
        self.lib.define(
            "overloaded.int(Tensor x, int n) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded.int", "CompositeExplicitAutograd"
        )
        def overloaded_int_impl(x: Tensor, n: int) -> Tensor:
            return x * n

        self.lib.define(
            "overloaded.int_out(Tensor x, int n, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded.int_out", "CompositeExplicitAutograd"
        )
        def overloaded_int_out_impl(x: Tensor, n: int, *, result: Tensor) -> Tensor:
            result.copy_(x * n)
            return result

        x = torch.randn(3, 4)
        expected = torch.ops._TestOutVariant.overloaded.int(x, 3)

        self.assertTrue("int_out" in torch.ops._TestOutVariant.overloaded.overloads())
        self.assertEqual(
            str(torch.ops._TestOutVariant.overloaded.int_out._schema),
            "_TestOutVariant::overloaded.int_out(Tensor x, int n, *, Tensor(a!) result) -> Tensor(a!)",
        )

        out = torch.empty(3, 4)
        torch.ops._TestOutVariant.overloaded.int_out(x, 3, result=out)
        self.assertEqual(out, expected)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.overloaded.int,
                torch.ops._TestOutVariant.overloaded.int_out,
            )
        )

    def test_overload_multiple_out_variant(self):
        # Define an op with a .scalar overload returning multiple tensors
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
            "overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
        )

        @torch.library.impl(
            "_TestOutVariant::overloaded_multi.scalar_out", "CompositeExplicitAutograd"
        )
        def overloaded_multi_scalar_out_impl(
            x: Tensor, scale: float, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            out1.copy_(x * scale)
            out2.copy_(x + scale)
            return out1, out2

        x = torch.randn(3, 4)
        expected1, expected2 = torch.ops._TestOutVariant.overloaded_multi.scalar(
            x, 2.5
        )

        self.assertTrue(
            "scalar_out" in torch.ops._TestOutVariant.overloaded_multi.overloads()
        )
        self.assertEqual(
            str(torch.ops._TestOutVariant.overloaded_multi.scalar_out._schema),
            "_TestOutVariant::overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
        )

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestOutVariant.overloaded_multi.scalar_out(
            x, 2.5, out1=out1, out2=out2
        )
        self.assertEqual(out1, expected1)
        self.assertEqual(out2, expected2)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.overloaded_multi.scalar,
                torch.ops._TestOutVariant.overloaded_multi.scalar_out,
            )
        )

    def test_multiple_overloads_with_out_variants(self):
        # Define an op with multiple overloads, each with their own out variant
        self.lib.define(
            "multi_overload.int(Tensor x, int n) -> Tensor",
        )
        self.lib.define(
            "multi_overload.float(Tensor x, float n) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::multi_overload.int", "CompositeExplicitAutograd"
        )
        def multi_overload_int_impl(x: Tensor, n: int) -> Tensor:
            return x * n

        @torch.library.impl(
            "_TestOutVariant::multi_overload.float", "CompositeExplicitAutograd"
        )
        def multi_overload_float_impl(x: Tensor, n: float) -> Tensor:
            return x + n

        self.lib.define(
            "multi_overload.int_out(Tensor x, int n, *, Tensor(a!) result) -> Tensor(a!)"
        )
        self.lib.define(
            "multi_overload.float_out(Tensor x, float n, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::multi_overload.int_out", "CompositeExplicitAutograd"
        )
        def multi_overload_int_out_impl(x: Tensor, n: int, *, result: Tensor) -> Tensor:
            result.copy_(x * n)
            return result

        @torch.library.impl(
            "_TestOutVariant::multi_overload.float_out", "CompositeExplicitAutograd"
        )
        def multi_overload_float_out_impl(
            x: Tensor, n: float, *, result: Tensor
        ) -> Tensor:
            result.copy_(x + n)
            return result

        x = torch.randn(3, 4)

        # Test int overload
        expected_int = torch.ops._TestOutVariant.multi_overload.int(x, 3)
        self.assertTrue(
            "int_out" in torch.ops._TestOutVariant.multi_overload.overloads()
        )
        out_int = torch.empty(3, 4)
        torch.ops._TestOutVariant.multi_overload.int_out(x, 3, result=out_int)
        self.assertEqual(out_int, expected_int)

        # Test float overload
        expected_float = torch.ops._TestOutVariant.multi_overload.float(x, 2.5)
        self.assertTrue(
            "float_out" in torch.ops._TestOutVariant.multi_overload.overloads()
        )
        out_float = torch.empty(3, 4)
        torch.ops._TestOutVariant.multi_overload.float_out(x, 2.5, result=out_float)
        self.assertEqual(out_float, expected_float)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.multi_overload.int,
                torch.ops._TestOutVariant.multi_overload.int_out,
            )
        )
        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.multi_overload.float,
                torch.ops._TestOutVariant.multi_overload.float_out,
            )
        )

    def test_custom_op_single_out_variant(self):
        @torch.library.custom_op(
            "_TestOutVariant::custom_single_return", mutates_args=()
        )
        def custom_single_return(x: Tensor) -> Tensor:
            return x * 2

        self.lib.define(
            "custom_single_return.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::custom_single_return.out", "CompositeExplicitAutograd"
        )
        def custom_single_return_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x * 2)
            return result

        x = torch.randn(3, 4)
        expected = torch.ops._TestOutVariant.custom_single_return(x)

        self.assertTrue(
            "out" in torch.ops._TestOutVariant.custom_single_return.overloads()
        )

        out = torch.empty(3, 4)
        torch.ops._TestOutVariant.custom_single_return.out(x, result=out)
        self.assertEqual(out, expected)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.custom_single_return.default,
                torch.ops._TestOutVariant.custom_single_return.out,
            )
        )

    def test_custom_op_multiple_out_variant(self):
        @torch.library.custom_op(
            "_TestOutVariant::custom_multi_return", mutates_args=()
        )
        def custom_multi_return(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x * 3

        self.lib.define(
            "custom_multi_return.out(Tensor x, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
        )

        @torch.library.impl(
            "_TestOutVariant::custom_multi_return.out", "CompositeExplicitAutograd"
        )
        def custom_multi_return_out_impl(
            x: Tensor, *, out1: Tensor, out2: Tensor
        ) -> tuple[Tensor, Tensor]:
            out1.copy_(x * 2)
            out2.copy_(x * 3)
            return out1, out2

        x = torch.randn(3, 4)
        expected1, expected2 = torch.ops._TestOutVariant.custom_multi_return(x)

        self.assertTrue(
            "out" in torch.ops._TestOutVariant.custom_multi_return.overloads()
        )

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestOutVariant.custom_multi_return.out(x, out1=out1, out2=out2)
        self.assertEqual(out1, expected1)
        self.assertEqual(out2, expected2)

        self.assertTrue(
            check_out_variant(
                torch.ops._TestOutVariant.custom_multi_return.default,
                torch.ops._TestOutVariant.custom_multi_return.out,
            )
        )

    def test_no_out_variant_registered(self):
        # Define an op without any out variant
        self.lib.define(
            "no_out_op(Tensor x) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::no_out_op", "CompositeExplicitAutograd"
        )
        def no_out_op_impl(x: Tensor) -> Tensor:
            return x * 2

        # No out variant registered, so to_out_variant should return None
        out_op = to_out_variant(torch.ops._TestOutVariant.no_out_op.default)
        self.assertIsNone(out_op)

        # check_out_variant should raise with helpful error message
        with self.assertRaisesRegex(
            AssertionError,
            r"to_out_variant.*returned None.*Check that the out variant overload name follows the correct naming convention",
        ):
            check_out_variant(
                torch.ops._TestOutVariant.no_out_op.default,
                None,  # No expected out op
            )

    def test_out_variant_signature_mismatch(self):
        # Define an op and an out variant with mismatched signature
        self.lib.define(
            "sig_mismatch_op(Tensor x, Tensor y) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::sig_mismatch_op", "CompositeExplicitAutograd"
        )
        def sig_mismatch_op_impl(x: Tensor, y: Tensor) -> Tensor:
            return x + y

        # Register out variant with wrong signature (missing y argument)
        self.lib.define(
            "sig_mismatch_op.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)"
        )

        @torch.library.impl(
            "_TestOutVariant::sig_mismatch_op.out", "CompositeExplicitAutograd"
        )
        def sig_mismatch_op_out_impl(x: Tensor, *, result: Tensor) -> Tensor:
            result.copy_(x * 2)
            return result

        # The out variant has wrong signature, so to_out_variant should not find it
        out_op = to_out_variant(torch.ops._TestOutVariant.sig_mismatch_op.default)
        self.assertIsNone(out_op)

        # check_out_variant should raise with helpful error message
        with self.assertRaisesRegex(
            AssertionError,
            r"to_out_variant.*returned None.*Check that the out variant overload name follows the correct naming convention",
        ):
            check_out_variant(
                torch.ops._TestOutVariant.sig_mismatch_op.default,
                torch.ops._TestOutVariant.sig_mismatch_op.out,
            )


if __name__ == "__main__":
    run_tests()
