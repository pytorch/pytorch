import logging

import torch.fx as fx
import torch.fx.experimental.fx2trt.passes.remove_duplicate_output_args as dedup
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


_LOGGER = logging.getLogger(__name__)


class TestFx2TrtPasses(TestCase):
    def test_remove_duplicate_output_args(self):
        class Sub(nn.Module):
            def forward(self, x):
                return (x, x)

        class Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Sub()

            def forward(self, x):
                a_res = self.a(x)
                return a_res[0] + a_res[1]

        class Tracer(fx.Tracer):
            def is_leaf_module(self, m, qn):
                if isinstance(m, Sub):  # don't trace into
                    return True
                return False

        top = Top()
        ttop = fx.GraphModule(top, Tracer().trace(top), "top")
        ttop.a = fx.symbolic_trace(ttop.a)

        name_to_processed_subnet = dedup.remove_duplicate_output_args(ttop, ["a"])

        ttop(1)  # run inference should work

        processed_a = name_to_processed_subnet["a"]
        *_, a_output = processed_a.module.graph.nodes
        a_output: fx.Node

        ttop_graph_actual = str(ttop.graph).strip()
        ttop_graph_expected = """
graph():
    %x : [#users=1] = placeholder[target=x]
    %a : [#users=2] = call_module[target=a](args = (%x,), kwargs = {})
    %getitem : [#users=1] = call_function[target=operator.getitem](args = (%a, 0), kwargs = {})
    %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%a, 0), kwargs = {})
    %add : [#users=1] = call_function[target=operator.add](args = (%getitem, %getitem_1), kwargs = {})
    return add
""".strip()
        assert (
            ttop_graph_expected == ttop_graph_actual
        ), f"Unexpected ttop graph: {ttop_graph_actual}"

        ttop_a_graph_actual = str(ttop.a.graph).strip()
        ttop_a_graph_expected = """
graph():
    %x : [#users=1] = placeholder[target=x]
    return (x,)
""".strip()
        assert (
            ttop_a_graph_expected == ttop_a_graph_actual
        ), f"Unexpected ttop.a graph: {ttop_a_graph_actual}"

if __name__ == '__main__':
    run_tests()
