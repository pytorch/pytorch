from typing import Any, Tuple, Union
from torch.fx import Node
from ..utils import Pattern  # noqa: F401

NodePattern = Union[Tuple[Node, Node], Tuple[Node, Tuple[Node, Node]], Any]

# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
QuantizerCls = Any
