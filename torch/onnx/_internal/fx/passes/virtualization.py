# mypy: allow-untyped-defs
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.fx

from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass


class MovePlaceholderToFront(_pass.Transform):
    """This pass move all placeholder nodes to the front of the graph node list.

    In torch.fx.Graph, placeholder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        graph_module = self.module
        graph = graph_module.graph
        placeholders = []
        first_not_placeholder = None
        for node in graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
            if first_not_placeholder is None and node.op != "placeholder":
                first_not_placeholder = node
        if first_not_placeholder is None:
            return graph_module
        for placeholder in placeholders:
            first_not_placeholder.prepend(placeholder)
        return graph_module


class ReplaceGetAttrWithPlaceholder(_pass.Transform):
    """Replace get_attr with placeholder.

    The parameters and buffers accessed by the original get_attr are returned;
    they are useful when creating random inputs for the modified graph_module.
    """

    _replaced_attrs: Optional[Tuple[torch.Tensor, ...]]

    @property
    def replaced_attrs(self) -> Tuple[torch.Tensor, ...]:
        """The list of replaced weight tensors."""
        assert (
            self._replaced_attrs is not None
        ), "Must run ReplaceGetAttrWithPlaceholder first"
        return self._replaced_attrs

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        graph_module = self.module
        graph = graph_module.graph
        replaced_attrs: List[torch.Tensor] = []
        for node in graph.nodes:
            if node.op == "get_attr":
                replaced_attr: Optional[torch.Tensor] = None
                # get_attr could retrieve either parameter or buffer, so
                # we need to try both.
                try:
                    replaced_attr = graph_module.get_parameter(node.target)
                except AttributeError:
                    # It's possible that model author use buffer instead of
                    # parameter to store trainable weights. In this case,
                    # 1. get_parameter will throw something like
                    #    AttributeError: `bias` is not an nn.Parameter.
                    # 2. get_buffer should work.
                    replaced_attr = graph_module.get_buffer(node.target)

                # Reassign op type so that get_attr node becomes placeholder node.
                node.op = "placeholder"
                # The target name in placeholder must be a valid Python identifier.
                # Thus, we replace, e.g., "module.submodule.weight" with
                # "module_submodule_weight".
                node.target = node.target.replace(".", "_")
                # Default value is None. This is needed as long as the "graph_module"
                # has optional inputs. Assume the original forward signature is
                #  def forward(self, x, y=None)
                # and the replaced get_attr node has target "z". Then, the modified
                # signature should be
                #  def forward(self, x, y=None, z=None)
                # Without the following line, the signature will be
                #  def forward(self, x, y=None, z)
                # , which is not valid Python code.
                node.args = (None,)

                replaced_attrs.append(replaced_attr)

        self._replaced_attrs = tuple(replaced_attrs)

        return graph_module
