"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_native_python_assertion)
"""

# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._export.constraints import constrain_as_size
from torch._export.functionalize_assertions import _functionalize_side_effectful_ops
from torch._export import dynamic_dim
from torch.testing import FileCheck
from torch._export.exported_program import ExportGraphSignature


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

        result = _functionalize_side_effectful_ops(gm)
        self.assertEqual(result.dep_token_output, "dep_token_7")
        self.assertEqual(result.dep_token_output_index, 1)

        gm = result.graph_module
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
        result = _functionalize_side_effectful_ops(ep.graph_module)
        self.assertEqual(result.dep_token_output, "dep_token_4")
        self.assertEqual(result.dep_token_output_index, 1)

        gm = result.graph_module
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

    def test_functionalization(self) -> None:
        def f(x, y):
            a = x.item()
            constrain_as_size(a, 4, 7)
            return x + 4, x + y * 2

        inps = (torch.tensor([5]), torch.zeros((3, 4)))
        ep = torch._export.export(
            f,
            inps,
            constraints=[dynamic_dim(inps[1], 1) < 6],
            _functionalize_runtime_assertions=True,
        )
        FileCheck().check_count(
            "torch.ops.aten.functional_sym_constrain_range", 1, exactly=True
        ).run(ep.graph_module.code)
        inps = (torch.tensor([7]), torch.ones((3, 5)))
        self.assertTrue(torch._dynamo.utils.same(ep(*inps), f(*inps)))

    def test_native_python_assertion(self) -> None:
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        inp = torch.Tensor([3, 4, 5])
        ep = torch._export.export(f, (inp,), _functionalize_runtime_assertions=True)

        # Check native assertion has corresponding functional assertion nodes generated.
        select_int_node = next(
            n
            for n in ep.graph_module.graph.nodes
            if n.target == torch.ops.aten.select.int
        )
        equal_scalar_node = select_int_node.next
        dep_token_node = equal_scalar_node.next
        self.assertIn(
            "call_function[target=torch.ops.aten._functional_assert_async.msg]"
            "(args = (%eq_scalar, assertion error), kwargs = {dep_token: %dep_token_2}",
            dep_token_node.format_node(),
        )

    def test_functionalization_with_mutated_buffer(self) -> None:
        buf = torch.ones(6, 2)
        weight = 0.01
        bias = 0.2
        d_in = 3
        d_out = 4

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buf", buf)

                self.linear = torch.nn.Linear(d_in, d_out)
                self.linear.weight.data.fill_(weight)
                self.linear.bias.data.fill_(bias)

            def forward(self, x):
                self.buf.add_(5)
                return self.linear(x).cos() + self.buf.sum()

        inp = torch.ones(4, 3)
        ep = torch._export.export(
            Foo(),
            (inp,),
            constraints=[dynamic_dim(inp, 0) >= 3],
            _functionalize_runtime_assertions=True,
        )

        self.assertEqual(
            ep.graph_signature,
            ExportGraphSignature(
                parameters=["L__self___linear.weight", "L__self___linear.bias"],
                buffers=["L__self___buf"],
                user_inputs=["arg3_1"],
                user_outputs=["add_tensor_1"],
                inputs_to_parameters={
                    "arg0_1": "L__self___linear.weight",
                    "arg1_1": "L__self___linear.bias",
                },
                inputs_to_buffers={"arg2_1": "L__self___buf"},
                buffers_to_mutate={"add_tensor": "L__self___buf"},
                backward_signature=None,
                assertion_dep_token_output="dep_token_8",
                assertion_dep_token_index=2,
            ),
        )
        output_node = next(n for n in ep.graph.nodes if n.op == "output")
        self.assertEqual(
            [str(arg) for arg in output_node.args[0]],
            ["add_tensor", "add_tensor_1", "dep_token_8"],
        )
        inp = torch.randn(5, 3)
        self.assertTrue(
            torch._dynamo.utils.same(
                # Directly check run output of `ep.graph_module` which is
                # functionalized.
                ep.graph_module(
                    torch.full((d_out, d_in), weight),
                    torch.full((d_out,), bias),
                    buf.clone(),
                    inp,
                ),
                (buf.add(5), Foo()(inp), torch.empty(0)),
            )
        )
        self.assertTrue(torch._dynamo.utils.same(ep(inp), Foo()(inp)))


if __name__ == "__main__":
    run_tests()
