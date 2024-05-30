# Owner(s): ["oncall: export"]
# flake8: noqa
import unittest

from typing import Dict, List, Tuple

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.wrappers import _mark_strict_experimental

from torch._functorch.aot_autograd import aot_export_module
from torch.export._trace import _convert_ts_to_export_experimental

from torch.testing import FileCheck


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
    def test_with_buffer_as_submodule(self):
        @_mark_strict_experimental
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(3))

            def forward(self, x):
                y = x + 2
                y.add_(4)
                # this doesnt' work today with HOO
                # self.buffer1.add_(6)
                buffer_updated = self.buffer1 + 6
                return x.sum() + y.sum() + buffer_updated.sum()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                x_v2 = x.sin()
                return (self.submodule(x_v2), x + 3)

        inp = torch.randn(3)
        ep = torch.export.export(M(), (inp,), strict=False)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, b_submodule_buffer1, x):
    sin = torch.ops.aten.sin.default(x)
    strict_graph_0 = self.strict_graph_0
    strict_mode = torch.ops.higher_order.strict_mode(strict_graph_0, (sin, b_submodule_buffer1));  strict_graph_0 = sin = b_submodule_buffer1 = None
    getitem_2 = strict_mode[0];  strict_mode = None
    add = torch.ops.aten.add.Tensor(x, 3);  x = None
    return (getitem_2, add)""",
        )

        self.assertExpectedInline(
            str(ep.graph_module.strict_graph_0.code.strip()),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 2)
    add_1 = torch.ops.aten.add.Tensor(add, 4);  add = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, 6);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    sum_2 = torch.ops.aten.sum.default(add_1);  add_1 = None
    add_3 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_2);  add_2 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, sum_3);  add_3 = sum_3 = None
    return (add_4,)""",
        )

        eager_mod = M()
        ep = torch.export.export(eager_mod, (inp,), strict=True)

        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

    def test_mark_strict_with_container_type(self):
        @_mark_strict_experimental
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x0 = x[0][0]
                return x0.sum()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                return self.submodule(x)

        inp = ((torch.randn(3),),)
        with self.assertRaisesRegex(
            RuntimeError, "strict_mode HOO doesn't work unless"
        ):
            ep = torch.export.export(M(), inp, strict=False)

    def test_torchscript_module_export(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.cos() + x.sin()

        model_to_trace = M()
        inps = (torch.randn(4, 4),)
        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

        self.assertTrue(torch.allclose(exported_module(*inps), model_to_trace(*inps)))

    def test_torchscript_module_export_single_input(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.cos() + x.sin()

        model_to_trace = M()
        inps = torch.randn(4, 4)
        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

        self.assertTrue(torch.allclose(exported_module(inps), model_to_trace(inps)))

    def test_torchscript_module_export_various_inputs_with_annotated_input_names(self):
        def _check_equality_and_annotations(m_func, inps):
            # Original module.
            model_to_trace = m_func()

            # ExportedProgram from TorchScript module.
            traced_module_by_torchscript = torch.jit.trace(
                m_func(), example_inputs=inps
            )
            exported_module = _convert_ts_to_export_experimental(
                traced_module_by_torchscript, inps
            )

            # ExportedProgram from original module.
            original_exported_module = torch.export.export(m_func(), inps)

            # Check whether input annotations are the same as tracing the original module.
            orig_ph_name_list = [
                n.name
                for n in original_exported_module.graph.nodes
                if n.op == "placeholder"
            ]
            ph_name_list = [
                n.name for n in exported_module.graph.nodes if n.op == "placeholder"
            ]
            self.assertEqual(orig_ph_name_list, ph_name_list)

            # Check results equality.
            self.assertTrue(
                torch.allclose(exported_module(*inps), model_to_trace(*inps))
            )

        # Tuple
        class MTuple(torch.nn.Module):
            def forward(self, x: Tuple[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MTuple, ((torch.randn(4), torch.randn(4)),))

        # List
        class MList(torch.nn.Module):
            def forward(self, x: List[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MList, ([torch.randn(4), torch.randn(4)],))

        # Dict
        class MDict(torch.nn.Module):
            def forward(self, x: Dict[str, torch.Tensor]):
                return x["0"] + x["1"]

        _check_equality_and_annotations(
            MDict, ({"0": torch.randn(4), "1": torch.randn(4)},)
        )


if __name__ == "__main__":
    run_tests()
