# Owner(s): ["oncall: pt2"]

import operator

import torch
from torch._functorch.partitioners import _extract_fwd_bwd_modules
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges


class TestPartitioners(TestCase):
    def test_saves_unreplaced_input_symbols_for_backward(self):
        shape_env = ShapeEnv()
        s0 = make_symbol(SymT.SIZE, 0, integer=True)
        shape_env.var_to_range[s0] = ValueRanges(2, 10)
        s0_sym = shape_env.create_symintnode(s0, hint=3)

        u0_sym = shape_env.create_unbacked_symint()
        u0 = u0_sym.node._expr
        shape_env._set_replacement(u0, s0, "test")
        derived_sym = shape_env.create_symintnode(u0 + 1, hint=None)

        graph = torch.fx.Graph()
        s0_node = graph.placeholder("primals_1")
        s0_node.meta["val"] = s0_sym
        u0_node = graph.call_function(operator.add, (s0_node, 0))
        u0_node.meta["val"] = u0_sym
        derived_node = graph.call_function(operator.add, (u0_node, 1))
        derived_node.meta["val"] = derived_sym
        output = graph.output((s0_node, derived_node))
        output.meta["desc"] = [None, None]
        joint = torch.fx.GraphModule({}, graph)

        _, bw_graph = _extract_fwd_bwd_modules(
            joint, [], [derived_node], num_fwd_outputs=1
        )
        bw_placeholders = list(bw_graph.graph.find_nodes(op="placeholder"))
        self.assertEqual(
            [node.meta["val"].node._expr for node in bw_placeholders],
            [u0, u0 + 1],
        )

        graph_inputs = [node.meta["val"] for node in bw_placeholders]
        lowering = GraphLowering(
            bw_graph, graph_inputs, shape_env=shape_env, is_backward=True
        )
        with V.set_graph_handler(lowering):
            lowering.run(*graph_inputs)

        self.assertEqual(
            [lowering.graph_inputs[name] for name in lowering.graph_input_names],
            [u0, u0 + 1],
        )
        self.assertEqual(
            [
                lowering.graph_inputs[name].xreplace({u0: s0})
                for name in lowering.graph_input_names
            ],
            [s0, s0 + 1],
        )


if __name__ == "__main__":
    run_tests()
