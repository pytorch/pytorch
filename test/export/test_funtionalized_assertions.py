# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._export.constraints import constrain_as_size
from torch._export.functionalize_assertions import functionalize
from torch.testing import FileCheck


class TestFuntionalAssertions(TestCase):
    def test_functional_assert_async(self) -> None:
        dep_token = torch.empty((1, 2))
        self.assertEqual(
            torch.ops.aten._functional_assert_async(torch.tensor(1), dep_token),
            dep_token,
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected Tensor with single nonzero value, but got zero"
        ) as cm:
            torch._functional_assert_async(torch.tensor(0), dep_token)

    def test_functional_assert_async_msg(self) -> None:
        dep_token = torch.empty((2, 3))
        self.assertEqual(
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(1), "test msg", dep_token
            ),
            dep_token,
        )
        with self.assertRaisesRegex(RuntimeError, "test msg"):
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(0), "test msg", dep_token
            ),

    def test_functional_sym_constrain_range(self) -> None:
        dep_token = torch.empty((1,))
        self.assertEqual(
            torch.ops.aten.functional_sym_constrain_range(
                3, min=2, max=5, dep_token=dep_token
            ),
            dep_token,
        )


class TestFunctionalization(TestCase):
    def test_functionalize_inline_contraints(self) -> None:
        def f(x):
            a = x.item()
            constrain_as_size(a, 4, 7)
            return torch.empty((a, 4))

        ep = torch._export.export(f, (torch.tensor([7]),))
        gm = ep.graph_module
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default",
            1,
            exactly=True,
        ).run(gm.code)

        gm = functionalize(gm)

        res, dep_token = gm((torch.tensor([5])))
        self.assertEqual(res.shape, torch.Size([5, 4]))
        self.assertEqual(dep_token.shape, torch.Size([0]))

        FileCheck().check_count(
            "torch.ops.aten.functional_sym_constrain_range", 1, exactly=True
        )
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        )

        dep_token_node = next(n for n in gm.graph.nodes if n.name == "dep_token")
        constrain_node = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten.functional_sym_constrain_range
        )
        self.assertEqual(constrain_node.kwargs["dep_token"], dep_token_node)


if __name__ == "__main__":
    run_tests()
