# Owner(s): ["module: custom-operators"]
import unittest

import torch
from torch import Tensor
from torch._library._out_variant import (
    check_out_variant,
    generate_out_variant,
    to_out_variant,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


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
            "single_return.out(Tensor x, Tensor y, *, Tensor(a!) result) -> Tensor(a!)"
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
            "multi_return.out(Tensor x, Tensor y, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
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
            "overloaded_multi.Tensor_out(Tensor x, Tensor scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
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
            "overloaded_multi.scalar_out(Tensor x, float scale, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
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
            "multi_overload_int.out(Tensor x, SymInt n, *, Tensor(a!) result) -> Tensor(a!)"
        )
        self.lib.define(
            "multi_overload_float.out(Tensor x, float n, *, Tensor(a!) result) -> Tensor(a!)"
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
            "sig_mismatch_op.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)"
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
        # torchgen reserves _out suffix for out variants, so operators with _out
        # in their name will fail schema parsing
        self.lib.define(
            "my_custom_out(Tensor x) -> Tensor",
        )

        @torch.library.impl(
            "_TestOutVariant::my_custom_out", "CompositeExplicitAutograd"
        )
        def my_custom_out_impl(x: Tensor) -> Tensor:
            return x * 2

        with self.assertRaisesRegex(ValueError, "Failed to parse schema"):
            to_out_variant(torch.ops._TestOutVariant.my_custom_out.default)

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

    def test_trailing_underscore_functional(self):
        # Operators with trailing underscore but no mutable args violate
        # torchgen's inplace naming convention, causing schema parsing to fail
        self.lib.define(
            "my_func_(Tensor x) -> Tensor",
        )

        @torch.library.impl("_TestOutVariant::my_func_", "CompositeExplicitAutograd")
        def my_func__impl(x: Tensor) -> Tensor:
            return x * 2

        with self.assertRaisesRegex(ValueError, "Failed to parse schema"):
            to_out_variant(torch.ops._TestOutVariant.my_func_.default)

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
            "wrong_out_count.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)"
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
            "optional_mismatch.out(Tensor? x, *, Tensor(a!) result) -> Tensor(a!)"
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
            "default_mismatch.out(Tensor x, int n=1, *, Tensor(a!) result) -> Tensor(a!)"
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


class TestAutogenOut(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestAutogenOut", "FRAGMENT")  # noqa: TOR901

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_single_output1(self):
        @torch.library.custom_op(
            "_TestAutogenOut::single_out_test", mutates_args=(), autogen_out=["result"]
        )
        def single_out_test(x: Tensor) -> Tensor:
            return x * 2

        x = torch.randn(3, 4)
        result = torch.ops._TestAutogenOut.single_out_test(x)

        self.assertTrue("out" in torch.ops._TestAutogenOut.single_out_test.overloads())
        self.assertExpectedInline(
            str(torch.ops._TestAutogenOut.single_out_test.out._schema),
            """_TestAutogenOut::single_out_test.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)""",
        )

        out = torch.empty(3, 4)
        torch.ops._TestAutogenOut.single_out_test.out(x, result=out)
        self.assertEqual(out, result)

        check_out_variant(
            torch.ops._TestAutogenOut.single_out_test.default,
            torch.ops._TestAutogenOut.single_out_test.out,
        )

    def test_multiple_outputs1(self):
        @torch.library.custom_op(
            "_TestAutogenOut::multi_out_test",
            mutates_args=(),
            autogen_out=["output", "output_scale"],
        )
        def multi_out_test(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x * 3

        self.assertTrue("out" in torch.ops._TestAutogenOut.multi_out_test.overloads())
        self.assertExpectedInline(
            str(torch.ops._TestAutogenOut.multi_out_test.out._schema),
            """_TestAutogenOut::multi_out_test.out(Tensor x, *, Tensor(a!) output, Tensor(b!) output_scale) -> (Tensor(a!), Tensor(b!))""",  # noqa: B950
        )

        x = torch.randn(3, 4)
        res1, res2 = torch.ops._TestAutogenOut.multi_out_test(x)

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestAutogenOut.multi_out_test.out(x, output=out1, output_scale=out2)
        self.assertEqual(out1, res1)
        self.assertEqual(out2, res2)

        check_out_variant(
            torch.ops._TestAutogenOut.multi_out_test.default,
            torch.ops._TestAutogenOut.multi_out_test.out,
        )

    def test_overload1(self):
        @torch.library.custom_op(
            "_TestAutogenOut::overloaded_op.scalar",
            mutates_args=(),
            autogen_out=["result"],
        )
        def overloaded_op_scalar(x: Tensor, n: int) -> Tensor:
            return x * n

        self.assertTrue(
            "scalar_out" in torch.ops._TestAutogenOut.overloaded_op.overloads()
        )
        self.assertExpectedInline(
            str(torch.ops._TestAutogenOut.overloaded_op.scalar_out._schema),
            """_TestAutogenOut::overloaded_op.scalar_out(Tensor x, SymInt n, *, Tensor(a!) result) -> Tensor(a!)""",
        )

        x = torch.randn(3, 4)
        res = torch.ops._TestAutogenOut.overloaded_op.scalar(x, 3)
        out = torch.empty(3, 4)
        torch.ops._TestAutogenOut.overloaded_op.scalar_out(x, 3, result=out)
        self.assertEqual(out, res)

        check_out_variant(
            torch.ops._TestAutogenOut.overloaded_op.scalar,
            torch.ops._TestAutogenOut.overloaded_op.scalar_out,
        )

    def test_output_resizing(self):
        @torch.library.custom_op(
            "_TestAutogenOut::resize_test", mutates_args=(), autogen_out=["out"]
        )
        def resize_test(x: Tensor) -> Tensor:
            return x.sum(dim=0)  # Changes shape

        x = torch.randn(3, 4)
        out = torch.empty(10, 10)  # Wrong shape

        torch.ops._TestAutogenOut.resize_test.out(x, out=out)
        self.assertEqual(out.shape, torch.Size([4]))
        self.assertEqual(out, x.sum(dim=0))

        check_out_variant(
            torch.ops._TestAutogenOut.resize_test.default,
            torch.ops._TestAutogenOut.resize_test.out,
        )

    def test_invalid_out_names(self):
        with self.assertRaisesRegex(
            ValueError, "out_names has 1 names but op returns 2 tensors"
        ):

            @torch.library.custom_op(
                "_TestAutogenOut::wrong_count_test",
                mutates_args=(),
                autogen_out=["out"],
            )
            def wrong_count_test(x: Tensor) -> tuple[Tensor, Tensor]:
                return x, x

    def test_non_tensor_output(self):
        with self.assertRaisesRegex(ValueError, "Found non-tensor return type"):

            @torch.library.custom_op(
                "_TestAutogenOut::non_tensor_test", mutates_args=(), autogen_out=["out"]
            )
            def non_tensor_test(x: Tensor) -> tuple[Tensor, int]:
                return x, 42

    def test_out_variant_already_exists(self):
        @torch.library.custom_op("_TestAutogenOut::existing_out_test", mutates_args=())
        def existing_out_test(x: Tensor) -> Tensor:
            return x * 2

        self.lib.define(
            "existing_out_test.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)"
        )
        self.lib.impl("existing_out_test.out", lambda x, *, out: out.copy_(x * 2))

        with self.assertRaisesRegex(
            ValueError,
            "Cannot generate out variant: _TestAutogenOut::existing_out_test.out already exists.",  # noqa: B950
        ):
            generate_out_variant(
                existing_out_test._opoverload, ["out"], lib=existing_out_test._lib
            )

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_dispatch_key(self):
        @torch.library.custom_op(
            "_TestAutogenOut::cpu_only_test",
            mutates_args=(),
            device_types="cpu",
            autogen_out=["out"],
        )
        def cpu_only_test(x: Tensor) -> Tensor:
            return x * 2

        x = torch.randn(3, 4)
        out = torch.empty(3, 4)
        torch.ops._TestAutogenOut.cpu_only_test.out(x, out=out)
        self.assertEqual(out, x * 2)

        check_out_variant(
            torch.ops._TestAutogenOut.cpu_only_test.default,
            torch.ops._TestAutogenOut.cpu_only_test.out,
        )

        x_cuda = torch.randn(3, 4, device="cuda")
        out_cuda = torch.empty(3, 4, device="cuda")
        with self.assertRaisesRegex(
            NotImplementedError,
            "Could not run '_TestAutogenOut::cpu_only_test' with arguments from the 'CUDA' backend.",  # noqa: B950
        ):
            torch.ops._TestAutogenOut.cpu_only_test.out(x_cuda, out=out_cuda)

    def test_single_output2(self):
        self.lib.define("single_return2(Tensor x) -> Tensor")

        @torch.library.impl(
            "_TestAutogenOut::single_return2", "CompositeExplicitAutograd"
        )
        def single_return_impl(x: Tensor) -> Tensor:
            return x * 2

        torch.library._generate_out_variant(
            torch.ops._TestAutogenOut.single_return2.default,
            ["result"],
        )

        self.assertTrue("out" in torch.ops._TestAutogenOut.single_return2.overloads())

        check_out_variant(
            torch.ops._TestAutogenOut.single_return2.default,
            torch.ops._TestAutogenOut.single_return2.out,
        )

        x = torch.randn(3, 4)
        expected = torch.ops._TestAutogenOut.single_return2(x)

        out = torch.empty(3, 4)
        torch.ops._TestAutogenOut.single_return2.out(x, result=out)
        self.assertEqual(out, expected)

    def test_multiple_outputs2(self):
        self.lib.define("multi_return2(Tensor x) -> (Tensor, Tensor)")

        @torch.library.impl(
            "_TestAutogenOut::multi_return2", "CompositeExplicitAutograd"
        )
        def multi_return_impl(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x + 1

        torch.library._generate_out_variant(
            torch.ops._TestAutogenOut.multi_return2.default,
            ["out1", "out2"],
            lib=self.lib,
        )

        self.assertTrue("out" in torch.ops._TestAutogenOut.multi_return2.overloads())

        check_out_variant(
            torch.ops._TestAutogenOut.multi_return2.default,
            torch.ops._TestAutogenOut.multi_return2.out,
        )

        x = torch.randn(3, 4)
        expected1, expected2 = torch.ops._TestAutogenOut.multi_return2(x)

        out1 = torch.empty(3, 4)
        out2 = torch.empty(3, 4)
        torch.ops._TestAutogenOut.multi_return2.out(x, out1=out1, out2=out2)
        self.assertEqual(out1, expected1)
        self.assertEqual(out2, expected2)

    def test_overload2(self):
        self.lib.define("overloaded2.scalar(Tensor x, float scale) -> Tensor")

        @torch.library.impl(
            "_TestAutogenOut::overloaded2.scalar", "CompositeExplicitAutograd"
        )
        def overloaded_scalar_impl(x: Tensor, scale: float) -> Tensor:
            return x * scale

        self.lib._generate_out_variant(
            "overloaded2.scalar",
            ["result"],
        )

        check_out_variant(
            torch.ops._TestAutogenOut.overloaded2.scalar,
            torch.ops._TestAutogenOut.overloaded2.scalar_out,
        )

        x = torch.randn(3, 4)
        expected = torch.ops._TestAutogenOut.overloaded2.scalar(x, 2.5)

        out = torch.empty(3, 4)
        torch.ops._TestAutogenOut.overloaded2.scalar_out(x, 2.5, result=out)
        self.assertEqual(out, expected)


if __name__ == "__main__":
    run_tests()
