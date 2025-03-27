# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.onnx._internal.fx import _pass


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class RestoreParameterAndBufferNames(_pass.Transform):
    """Restore parameter and buffer names from original nn.module.

    This pass is useful for readability of the exported ONNX graph. It restores the
    parameter and buffer names from the original nn.module. For example, if the original
    nn.module has a parameter named `root.linear.0.weight`, and the parameter is renamed to
    `_param_constant9` by FX, this pass will rename it back.

    This pass must be run after `Decompose` pass. Because this pass is expected to be called on
    `fx.GraphModule` produced by `proxy_tensor.make_fx`, where all parameters and buffers
    are registered at root level.
    """

    def __init__(
        self,
        fx_module: torch.fx.GraphModule,
        original_nn_module: torch.nn.Module,
    ):
        super().__init__(fx_module)
        self.original_nn_module = original_nn_module

    def _rename_param_and_buffer(
        self,
        nodes: Sequence[torch.fx.Node],
        new_name: str,
    ) -> None:
        """Rename the parameter/buffer and replace corresponding nodes with new nodes of updated target."""
        assert len(nodes) > 0, "`nodes` cannot be empty"
        assert len({node.target for node in nodes}) == 1, (
            "`nodes` must all have same `target`"
        )
        old_name = nodes[0].target
        assert isinstance(old_name, str), f"Expected str, got type({old_name})"
        # Parameter/buffer name cannot contain "."
        normalized_name = new_name.replace(".", "/")
        attr_value = getattr(self.module, old_name)
        setattr(self.module, normalized_name, attr_value)
        delattr(self.module, old_name)
        for node in nodes:
            with self.module.graph.inserting_before(node):
                new_node = self.module.graph.get_attr(normalized_name)
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                self.module.graph.erase_node(node)
        logger.info(
            "Renamed 'self.%s' to 'self.%s', "
            "normalized from original parameter name '%s'.",
            old_name,
            normalized_name,
            new_name,
        )

    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """Restore parameter and buffer names from original module.

        For each `get_attr` node, if the target is a str representing a parameter or buffer
        under `self.module`, we rename the parameter or buffer to its original name.
        The parameters and buffers between `self.module` and `self.original_nn_module` refer
        to the same objects, allowing us to use it as key to retrieve the original name.
        """
        assert len(args) == 0, "RestoreParameterAndBufferNames does not take any args"
        assert len(kwargs) == 0, (
            "RestoreParameterAndBufferNames does not take any kwargs"
        )
        # state_to_readable_name[parameter/buffer] returns the original readable name of
        # the parameter/buffer. E.g., "self.linear.weight".
        state_to_readable_name: dict[torch.nn.Parameter | torch.Tensor, str] = {}
        state_to_readable_name.update(
            {v: k for k, v in self.original_nn_module.named_parameters()}
        )
        state_to_readable_name.update(
            {v: k for k, v in self.original_nn_module.named_buffers()}
        )

        # old_name_to_nodes[old_name] returns a tuple of (nodes, new_name)
        # where `nodes` is a list of `get_attr` nodes with `old_name` as `target` and
        # `new_name` is the new readable name.
        old_name_to_nodes: dict[str, tuple[list[torch.fx.Node], str]] = {}

        for node in self.module.graph.nodes:
            if node.op == "get_attr":
                assert isinstance(node.target, str), (
                    f"Expected str, got type({node.target})"
                )
                if node.target.find(".") != -1:
                    raise RuntimeError(
                        f"Unexpected target {node.target} in get_attr, found '.' in target. "
                        f"All parameters and buffers are expected to be registered at root level, "
                        f"i.e., self.module. "
                    )
                if node.target in old_name_to_nodes:
                    # We have already processed this parameter/buffer.
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

                logger.info(
                    "Cannot find readable name for self.%s: %s. The name is unchanged.",
                    node.target,
                    type(attr_value),
                )

        for nodes, new_name in old_name_to_nodes.values():
            self._rename_param_and_buffer(nodes, new_name)

        return self.module
