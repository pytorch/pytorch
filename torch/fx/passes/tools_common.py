from typing import List, Tuple, Union

import torch
import torch.fx


Tensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
TensorOrTensors = Union[torch.Tensor, Tensors]
Nodes = List[torch.fx.Node]
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}
