# Owner(s): ["module: custom-operators"]
import unittest

import torch
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


class TestAutogenOut(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestAutogenOut", "FRAGMENT")  # noqa: TOR901

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_single_output(self):
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

    def test_multiple_outputs(self):
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

    def test_overload(self):
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

    def test_invalid_out_names(self):
        with self.assertRaisesRegex(
            ValueError, "autogen_out has 1 names but op returns 2 tensors"
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
        from torch._library._autogen_out import generate_out_variant

        @torch.library.custom_op("_TestAutogenOut::existing_out_test", mutates_args=())
        def existing_out_test(x: Tensor) -> Tensor:
            return x * 2

        self.lib.define(
            "existing_out_test.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)"
        )
        self.lib.impl("existing_out_test.out", lambda x, *, out: out.copy_(x * 2))

        with self.assertRaisesRegex(
            ValueError,
            "Cannot autogen out variant: _TestAutogenOut::existing_out_test.out already exists.",  # noqa: B950
        ):
            generate_out_variant(existing_out_test, ["out"])

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

        x_cuda = torch.randn(3, 4, device="cuda")
        out_cuda = torch.empty(3, 4, device="cuda")
        with self.assertRaisesRegex(
            NotImplementedError,
            "Could not run '_TestAutogenOut::cpu_only_test.out' with arguments from the 'CUDA' backend.",  # noqa: B950
        ):
            torch.ops._TestAutogenOut.cpu_only_test.out(x_cuda, out=out_cuda)


if __name__ == "__main__":
    run_tests()
