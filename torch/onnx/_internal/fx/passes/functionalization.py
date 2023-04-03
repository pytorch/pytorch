from __future__ import annotations

import torch
import torch.func
import torch.fx

from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.onnx._internal.fx.passes import utils


class Functionalize(_pass.Transform):
    """Functionalize a GraphModule.

    This pass utilizes ``torch.func.functionalize`` to convert a GraphModule into a
    functional form. The two main functionalities are:

    * ``torch.func.functionalize`` removes (intermediate) mutations and aliasing from a
    function, while preserving the function's semantics.

    * ``torch.func.functionalize`` also removes mutations (and views) that were performed
    on function inputs. However to preserve semantics, functionalize will "fix up" the
    mutations after the transform has finished running, by detecting if any tensor inputs
    "should have" been mutated, and copying the new data back to the inputs if necessary.

    For ONNX inference, it is recommended to run ``RemoveInputMutation`` after this pass.
    ``RemoveInputMutation`` removes the "fix up" nodes that were added by ``Functionalize``,
    which are not needed for ONNX inference.
    """

    @_beartype.beartype
    def __init__(self, module: torch.fx.GraphModule, enable_dynamic_axes: bool):
        super().__init__(module)
        self.enable_dynamic_axes = enable_dynamic_axes

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        # To preserve stack trace info after `make_fx`.
        module = utils.wrap_graph_module_for_node_meta_preservation(self.module)

        functionalized_callable = torch.func.functionalize(module)
        fx_mode = "symbolic" if self.enable_dynamic_axes else "fake"

        graph_module = proxy_tensor.make_fx(
            functionalized_callable,
            decomposition_table={},
            tracing_mode=fx_mode,
            _allow_non_fake_inputs=True,
        )(*args)

        # Rename placeholder targets to match the original module's signature since
        # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
        utils.replace_placeholder_name_and_target(graph_module, self.module)

        return graph_module


class RemoveInputMutation(_pass.Transform):
    """Remove `aten.copy_.default` nodes that mutate module inputs.

    This pass is recommended to be used after ``Functionalization`` pass.
    ``Functionalization`` pass adds `aten.copy_.default` nodes to the graph
    when it detects mutations to inputs. These nodes are not needed for ONNX export
    for inference. They could be useful for training.
    """

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        for node in reversed(self.module.graph.nodes):
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.copy_.default
                and len(node.users) == 0
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].op == "placeholder"
            ):
                self.module.graph.erase_node(node)
        return self.module
