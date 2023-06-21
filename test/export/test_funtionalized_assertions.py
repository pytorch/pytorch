# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._export.constraints import constrain_as_size
from torch._export.functionalize_assertions import _functionalize_side_effectful_ops
from torch._export import dynamic_dim
from torch.testing import FileCheck


class TestFuntionalAssertions(TestCase):
    def test_functional_assert_async_msg(self) -> None:
        dep_token = torch.ops.aten.make_dep_token()
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
        dep_token = torch.ops.aten.make_dep_token()
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

        gm = _functionalize_side_effectful_ops(gm)

        with self.assertRaisesRegex(
            RuntimeError,
            r"_local_scalar_dense_default is outside of inline constraint \[4, 7\]",
        ) as cm:
            gm(torch.tensor([20]))

        res, dep_token = gm((torch.tensor([5])))
        self.assertEqual(res.shape, torch.Size([5, 4]))
        self.assertEqual(dep_token.shape, torch.Size([]))

        FileCheck().check_count(
            "torch.ops.aten.functional_sym_constrain_range", 1, exactly=True
        ).run(gm.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(gm.code)

        dep_token_node = next(n for n in gm.graph.nodes if n.name == "dep_token_4")
        constrain_node = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten.functional_sym_constrain_range
        )
        self.assertEqual(constrain_node.kwargs["dep_token"], dep_token_node)

    def test_functionalize_input_constraints(self) -> None:
        def f(x):
            return x * 2

        inp = torch.zeros(4, 8)
        ep = torch._export.export(
            f,
            (inp,),
            constraints=[
                dynamic_dim(inp, 0) < 10,
                dynamic_dim(inp, 0) >= 3,
            ],
        )
        gm = _functionalize_side_effectful_ops(ep.graph_module)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Input arg0_1.shape\[0\] is outside of specified dynamic range \[3, 9\]",
        ):
            gm(torch.ones(11, 8))

        self.assertEqual(gm(torch.ones(6, 8))[0], torch.full((6, 8), 2.0))
        FileCheck().check_count(
            "torch.ops.aten._functional_assert_async.msg", 3, exactly=True
        ).run(gm.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 0, exactly=True
        ).run(gm.code)


if __name__ == "__main__":
    run_tests()
