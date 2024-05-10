from __future__ import annotations

import torch
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass


class RemoveAssertions(_pass.Transform):
    """This pass removes all assertion and check nodes from the FX graph."""

    _ATEN_ASSERTION_TARGETS = {torch.ops.aten.sym_constrain_range_for_size.default}

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        graph_module = self.module
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target in self._ATEN_ASSERTION_TARGETS
            ):
                graph.erase_node(node)
        return graph_module
