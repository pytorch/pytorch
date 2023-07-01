from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics


class RestoreParameterAndBufferNames(_pass.Transform):
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        original_module: torch.nn.Module,
    ):
        super().__init__(diagnostic_context, module)
        self.original_module = original_module

    @_beartype.beartype
    def _rename_param_and_buffer(
        self,
        diagnostic: diagnostics.Diagnostic,
        nodes: Sequence[torch.fx.Node],
        new_name: str,
    ) -> None:
        assert len(nodes) > 0, "`nodes` cannot be empty"
        assert (
            len({node.target for node in nodes}) == 1
        ), "`nodes` must all have same `target`"
        old_name = nodes[0].target
        assert isinstance(old_name, str), f"Expected str, got type({old_name})"
        # Parameter/buffer name cannot contain "."
        normalized_name = new_name.replace(".", "_")
        attr_value = getattr(self.module, old_name)
        setattr(self.module, normalized_name, attr_value)
        delattr(self.module, old_name)
        for node in nodes:
            with self.module.graph.inserting_before(node):
                new_node = self.module.graph.get_attr(normalized_name)
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                self.module.graph.erase_node(node)
        diagnostic.with_additional_message(
            f"Renamed 'self.{old_name}' to 'self.{normalized_name}', "
            f"normalized from original parameter name '{new_name}'."
        )

    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert len(args) == 0, "RestoreParameterAndBufferNames does not take any args"
        assert (
            len(kwargs) == 0
        ), "RestoreParameterAndBufferNames does not take any kwargs"
        state_to_readable_name: Dict[Union[torch.nn.Parameter, torch.Tensor], str] = {}
        state_to_readable_name.update(
            {v: k for k, v in self.original_module.named_parameters()}
        )
        state_to_readable_name.update(
            {v: k for k, v in self.original_module.named_buffers()}
        )
        diagnostic = self.diagnostic_context.inflight_diagnostic()

        old_name_to_nodes: Dict[str, Tuple[List[torch.fx.Node], str]] = {}

        for node in self.module.graph.nodes:
            if node.op == "get_attr":
                assert isinstance(
                    node.target, str
                ), f"Expected str, got type({node.target})"
                if node.target in old_name_to_nodes:
                    old_name_to_nodes[node.target][0].append(node)
                    continue
                attr_value = getattr(self.module, node.target)
                if (
                    isinstance(attr_value, (torch.nn.Parameter, torch.Tensor))
                    and attr_value in state_to_readable_name
                ):
                    readable_name = state_to_readable_name[attr_value]
                    old_name_to_nodes[node.target] = ([node], readable_name)
                    continue

                diagnostic.with_additional_message(
                    f"Cannot find readable name for self.{node.target}: {type(attr_value)}. The name is unchanged."
                )
                if isinstance(attr_value, torch.nn.Parameter):
                    # If it is a parameter we treat it more seriously.
                    diagnostic.level = diagnostics.levels.WARNING
                else:
                    diagnostic.level = diagnostics.levels.NONE

        for nodes, new_name in old_name_to_nodes.values():
            self._rename_param_and_buffer(diagnostic, nodes, new_name)

        return self.module
