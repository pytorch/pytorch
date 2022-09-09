"""Importing this patches torch._C classes to add ONNX conveniences."""
from torch import _C
from torch.onnx._internal import torchscript


# TODO(#78694): Refactor the patching process to make it more transparent to users.
_C.Graph.op = torchscript.graph_op  # type: ignore[attr-defined]
_C.Graph.at = torchscript.aten_op  # type: ignore[attr-defined]
_C.Block.op = torchscript.block_op  # type: ignore[attr-defined]
_C.Graph.constant = torchscript.graph_constant  # type: ignore[attr-defined]
_C.Node.__getitem__ = torchscript.node_getitem  # type: ignore[attr-defined, misc, assignment]
