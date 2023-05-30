# Owner(s): ["module: dynamo"]
from typing import List, Type
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx import subgraph_rewriter
from torch.library import impl, Library
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import ExportPassBase, ExportPassBaseError


class TestPassInfra(TestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_export_pass_base(self) -> None:
        def f(x: torch.Tensor) -> List[torch.Tensor]:
            y = torch.cat([x, x])
            return torch.ops.aten.tensor_split.sections(y, 2)

        class NullPass(ExportPassBase):
            pass

        gm, _ = torchdynamo.export(f, *(torch.ones(3, 2),), aten_graph=True)
        new_gm = NullPass()(gm)
        self.assertIsNotNone(new_gm)
        new_nodes = new_gm.graph_module.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        old_nodes = gm.graph.nodes
        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_dialects(self) -> None:
        """
        Test if the dialects are maintained
        """

        lib = Library("DO_NOT_USE_TEST_ONLY", "DEF")
        lib.define("add_relu(Tensor self, Tensor other) -> Tensor")

        @impl(lib, "add_relu", "CompositeExplicitAutograd")
        def add_relu(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            x = torch.ops.aten.add.Tensor(self, other)
            y = torch.ops.aten.relu.default(x)
            return y

        class BackendOp:
            def __init__(self, op):
                self._op = op
                self.__name__ = f"backend.{self._op.__name__}"

            def __call__(self, *args, **kwargs):
                return self._op(*args, **kwargs)

        backend_op = BackendOp(torch.ops.DO_NOT_USE_TEST_ONLY.add_relu.default)

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = x + y
            return torch.ops.aten.relu.default(z)

        gm, _ = torchdynamo.export(
            f,
            *(torch.randn(2, 2), torch.randn(2, 2)),
            aten_graph=True,
        )
        FileCheck().check("torch.ops.aten.add.Tensor").check(
            "torch.ops.aten.relu.default"
        ).run(gm.code)

        class AddReluFusionPass(ExportPassBase):
            def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
                def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    z = torch.ops.aten.add.Tensor(x, y)
                    z = torch.ops.aten.relu.default(z)
                    return z

                def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return backend_op(x, y)

                subgraph_rewriter.replace_pattern(graph_module, pattern, replacement)

        class BackendNullPass(ExportPassBase):
            def get_valid_dialects(self) -> List[Type]:
                return [torch.ops.DO_NOT_USE_TEST_ONLY]

        class BackendViolatePass(ExportPassBase):
            """
            Violates the dialect by inserting torch.ops.aten ops rather than
            torch.ops.DO_NOT_USE_TEST_ONLY ops.
            """
            def get_valid_dialects(self) -> List[Type]:
                return [torch.ops.DO_NOT_USE_TEST_ONLY]

            def call_operator(self, op, args, kwargs, meta):
                if op == torch.ops.DO_NOT_USE_TEST_ONLY.add_relu.default:
                    add_res = super().call_operator(
                        torch.ops.aten.add.default,
                        args,
                        kwargs,
                        meta,
                    )
                    return super().call_operator(
                        torch.ops.aten.relu.default,
                        (add_res,),
                        (),
                        meta,
                    )
                return super().call_operator(op, args, kwargs, meta)

        AddReluFusionPass()(gm)
        FileCheck().check(
            "torch.ops.DO_NOT_USE_TEST_ONLY.add_relu.default"
        ).run(gm.code)

        new_gm = BackendNullPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module

        with self.assertRaisesRegex(ExportPassBaseError, "Expecting op of dialects:"):
            _ = BackendViolatePass()(gm)


if __name__ == '__main__':
    run_tests()
