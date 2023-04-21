from __future__ import annotations

import torch
import torch.func
import torch.fx
import torch._ops

from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics
from torch.onnx._internal.fx.passes import _utils

from typing import Dict, Optional


class Functionalize(_pass.Transform):
    """Functionalize a GraphModule.

    This pass utilizes ``torch.func.functionalize`` to convert a GraphModule into a
    functional form. The two main functionalities are (copied from its documentations):

    * ``torch.func.functionalize`` removes (intermediate) mutations and aliasing from a
    function, while preserving the function's semantics.

    * ``torch.func.functionalize`` also removes mutations (and views) that were performed
    on function inputs. However to preserve semantics, functionalize will "fix up" the
    mutations after the transform has finished running, by detecting if any tensor inputs
    "should have" been mutated, and copying the new data back to the inputs if necessary.
    For example, consider::

        def fn(a, b):
            a.add_(b)
            return a

      For a call like `fn(x, y)`, the variable `x` outside is also mutated. Hence just
      functionalizing is not enough for preserving the original semantics. A "special"
      input mutation step needs to be inserted at the end.::

        # After functionalization, without input mutation "fix up".
        # This is not semantically the same. The variable outside the function call that
        # was passed in as `a` is not mutated.
        def fn(a, b):
            new_a = a + b
            return new_a

        # Functionalization with input mutation "fix up" that preserves semantics.
        def fn(a, b):
            new_a = a + b

            # Copying the new data back to the inputs
            a.copy_(new_a)

            return new_a

    For ONNX inference, it is recommended to run ``RemoveInputMutation`` after this pass.
    ``RemoveInputMutation`` removes the "fix up" nodes that were added by ``Functionalize``,
    which are not needed for ONNX inference.

    NOTE: Functionalize must run before decomposition and aten graph lowering.
    https://github.com/pytorch/pytorch/issues/99662
    """

    @_beartype.beartype
    def __init__(self, module: torch.fx.GraphModule, enable_dynamic_axes: bool):
        super().__init__(module)
        self.enable_dynamic_axes = enable_dynamic_axes

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        # To preserve stack trace info after `make_fx`.
        module = _utils.wrap_graph_module_for_node_meta_preservation(self.module)

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
        _utils.replace_placeholder_name_and_target(graph_module, self.module)

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


class ReplaceInplacePostFunctionalization(_pass.Transform):
    """

    NOTE: This pass is not needed, if functionalize can be applied on decomposed graph.
    https://github.com/pytorch/pytorch/issues/99662
    """

    @_beartype.beartype
    def _outplace_target(
        self, inplace_target: torch._ops.OpOverload
    ) -> Optional[torch._ops.OpOverload]:
        assert inplace_target.namespace == "aten"
        outplace_name = inplace_target._schema.name.split("::")[1][:-1]
        overload_name = inplace_target._overloadname

        opoverloadpacket = getattr(torch.ops.aten, outplace_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            return None

        return getattr(opoverloadpacket, overload_name, None)

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to validate that
        # the mutated input value is not used after the mutation.
        node_to_last_use: Dict[torch.fx.Node, torch.fx.Node] = {}

        def register_last_uses(n: torch.fx.Node, user: torch.fx.Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user

        for node in reversed(self.module.graph.nodes):
            torch.fx.node.map_arg(node.args, lambda n: register_last_uses(n, node))
            torch.fx.node.map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        for node in self.module.graph.nodes:
            if node.op != "call_function" or not isinstance(
                node.target, torch._ops.OpOverload
            ):
                continue

            target = node.target
            mutated_input = node.args[0]

            name_without_overload = target._schema.name
            is_inplace = name_without_overload.endswith(
                "_"
            ) and not name_without_overload.endswith("__")
            is_aten = target.namespace == "aten"

            if not is_inplace:
                continue

            if not is_aten:
                # TODO(bowbao): Turn this into individual diagnostic.
                diagnostic = diagnostics.export_context().inflight_diagnostic(
                    rule=diagnostics.rules.fx_pass
                )
                diagnostic.level = diagnostics.levels.WARNING
                diagnostic.with_additional_message(
                    f"Found non-aten op {target} in graph with inplace naming convention. "
                    f"Skip replacing this op with outplace version."
                )
                continue

            assert isinstance(
                mutated_input, torch.fx.Node
            ), f"Expected mutated input to be a torch.fx.Node. Got {type(mutated_input)}"

            if node_to_last_use[mutated_input] != node:
                # TODO(bowbao): Turn this into individual diagnostic.
                raise RuntimeError(
                    f"Found inplace op node {node} that is not the last use of its input. "
                    f"Its mutated input is later used by {node_to_last_use[mutated_input]}. "
                    f"Please run RemoveInputMutation pass before ReplaceInplacePostFunctionalization."
                )

            outplace_target = self._outplace_target(target)

            if outplace_target is None:
                # TODO(bowbao): Turn this into individual diagnostic.
                diagnostic = diagnostics.export_context().inflight_diagnostic(
                    rule=diagnostics.rules.fx_pass
                )
                diagnostic.level = diagnostics.levels.WARNING
                diagnostic.with_additional_message(
                    f"Failed to find outplace version of {target}. Skip replacing this op."
                )
                continue

            node.target = outplace_target

        return self.module
