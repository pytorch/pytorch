from __future__ import annotations

from typing import Optional

import torch.fx
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode

__all__ = ['FakeTensorProp']

@compatibility(is_backward_compatible=False)
class FakeTensorProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
         override (bool): Whether to override the `val` attribute of each Node with the fake tensor result.
             If False, only the Nodes without `val` will be assigned the fake tensor result.
    """
    def __init__(self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None, override: bool = True):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        self._override = override

    def run_node(self, n: Node):
        result = super().run_node(n)
        if self._override or "val" not in n.meta:
            n.meta['val'] = result
        return result

    def propagate(self, *args):
        with self._mode:
            fake_args = [self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
            return super().run(*fake_args)
