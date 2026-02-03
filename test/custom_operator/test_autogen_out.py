# Owner(s): ["module: custom-operators"]
import unittest

import torch
from torch import Tensor
from torch._inductor import config
from torch._inductor.fx_passes.decompose_to_out_variant import decompose_to_out_variant
from torch._inductor.utils import run_and_get_code
from torch._library._autogen_out import to_out_variant
from torch.testing import FileCheck
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

        out_op = to_out_variant(torch.ops._TestAutogenOut.multi_out_test.default)
        self.assertEqual(out_op, torch.ops._TestAutogenOut.multi_out_test.out)

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

        out_op = to_out_variant(torch.ops._TestAutogenOut.overloaded_op.scalar)
        self.assertEqual(out_op, torch.ops._TestAutogenOut.overloaded_op.scalar_out)

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

    @config.patch(decompose_to_out_variant=True)
    def test_compile_single_out(self):
        @torch.library.custom_op(
            "_TestAutogenOut::decompose_test", mutates_args=(), autogen_out=["out"]
        )
        def decompose_test(x: Tensor) -> Tensor:
            return x * 2

        @decompose_test.register_fake
        def _(x):
            return torch.empty_like(x)

        def f(x):
            x = x + x
            a = torch.ops._TestAutogenOut.decompose_test(
                x
            )  # x is used again, cannot replace with out variant
            b = torch.ops._TestAutogenOut.decompose_test(x)
            return a + b

        compiled_f = torch.compile(f)
        inputs = (torch.randn(3),)
        out, codes = run_and_get_code(compiled_f, *inputs)

        self.assertEqual(f(*inputs), out)
        FileCheck().check_count(
            "= torch.ops._TestAutogenOut.decompose_test.default(", 1, exactly=True
        ).run(codes[0])
        FileCheck().check_count(
            "= torch.ops._TestAutogenOut.decompose_test.out(", 1, exactly=True
        ).run(codes[0])

    @config.patch(decompose_to_out_variant=True)
    def test_replace_multiple_out(self):
        @torch.library.custom_op(
            "_TestAutogenOut::multi_out_decompose",
            mutates_args=(),
            autogen_out=["out1", "out2"],
        )
        def multi_out_decompose(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, y * 3

        @multi_out_decompose.register_fake
        def _(x, y):
            return torch.empty_like(x), torch.empty_like(y)

        class M(torch.nn.Module):
            def forward(self, x, y):
                x = x + x
                y = y + y
                a, b = torch.ops._TestAutogenOut.multi_out_decompose(x, y)
                return a + b

        inputs = (torch.randn(3), torch.randn(3))
        ep = torch.export.export(M(), inputs, strict=True)
        ep._graph = decompose_to_out_variant(ep.graph)
        ep.graph_module.recompile()
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, x, y):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    add_1 = torch.ops.aten.add.Tensor(y, y);  y = None
    multi_out_decompose_out = torch.ops._TestAutogenOut.multi_out_decompose.out(add, add_1, out1 = add, out2 = add_1);  add = add_1 = None
    getitem = multi_out_decompose_out[0]
    getitem_1 = multi_out_decompose_out[1];  multi_out_decompose_out = None
    add_2 = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return (add_2,)""",  # noqa: B950
        )

        self.assertEqual(ep.module()(*inputs), M()(*inputs))

    @config.patch(decompose_to_out_variant=True)
    def test_replace_multiple_out_one_in(self):
        @torch.library.custom_op(
            "_TestAutogenOut::multi_result",
            mutates_args=(),
            autogen_out=["out1", "out2"],
        )
        def multi_result(x: Tensor) -> tuple[Tensor, Tensor]:
            return x * 2, x * 3

        @multi_result.register_fake
        def _(x):
            return torch.empty_like(x), torch.empty_like(x)

        class M(torch.nn.Module):
            def forward(self, x):
                x = x + x
                a, b = torch.ops._TestAutogenOut.multi_result(x)
                return a + b

        inputs = (torch.randn(3),)
        ep = torch.export.export(M(), inputs, strict=True)
        decompose_to_out_variant(ep.graph)
        ep.graph_module.recompile()
        # With only one input but two outputs, only one buffer can be reused.
        # The pass still converts to out variant, allocating a fresh buffer for the second output.
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    empty = torch.empty([3], dtype = torch.float32, device = device(type='cpu'))
    multi_result_out = torch.ops._TestAutogenOut.multi_result.out(add, out1 = add, out2 = empty);  add = empty = None
    getitem = multi_result_out[0]
    getitem_1 = multi_result_out[1];  multi_result_out = None
    add_1 = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return (add_1,)""",  # noqa: B950
        )

        self.assertEqual(ep.module()(*inputs), M()(*inputs))


if __name__ == "__main__":
    run_tests()
