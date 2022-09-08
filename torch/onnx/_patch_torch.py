"""Importing this patches torch._C classes to add ONNX conveniences."""
from torch import _C
from torch.onnx._internal import torch_graph


# TODO(#78694): Refactor the patching process to make it more transparent to users.
_C.Graph.op = torch_graph.graph_op  # type: ignore[attr-defined]
_C.Graph.at = torch_graph.aten_op  # type: ignore[attr-defined]
_C.Block.op = torch_graph.block_op  # type: ignore[attr-defined]
_C.Graph.constant = torch_graph.graph_constant  # type: ignore[attr-defined]
_C.Node.__getitem__ = torch_graph.node_getitem  # type: ignore[attr-defined, misc, assignment]
