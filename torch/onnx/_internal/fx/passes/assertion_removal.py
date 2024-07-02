from __future__ import annotations

import torch
from torch.onnx._internal.fx import _pass


class RemoveAssertions(_pass.Transform):
    """This pass removes all assertion and check nodes from the FX graph."""

    _ATEN_ASSERTION_TARGETS = frozenset(
        {
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch.ops.aten._assert_async.msg,
        }
    )

    def _run(self) -> torch.fx.GraphModule:
        for node in self.module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in self._ATEN_ASSERTION_TARGETS
            ):
                self.module.graph.erase_node(node)
        return self.module
