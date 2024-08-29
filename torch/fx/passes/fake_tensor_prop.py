# mypy: allow-untyped-defs
from typing import Optional

import torch.fx
from torch.fx import Node
from torch.fx.node import map_aggregate, Argument, Target
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import snapshot_fake, py_sym_types

from typing import Any, Dict, Tuple

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
    """
    def __init__(self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        mode.epoch += 1
        mode.reset_nt_tensor_id_counter()

    def run_node(self, n: Node):
        from torch.fx.experimental.symbolic_shapes import rebind_unbacked, compute_unbacked_bindings

        result = super().run_node(n)
        rebind_unbacked(self._mode.shape_env, n, result)

        def extract_val(obj):
            if isinstance(obj, FakeTensor):
                return snapshot_fake(obj)
            elif isinstance(obj, torch.Tensor):
                # TODO: How is it possible that we get a non fake tensor?  We
                # should be running under the mode...
                return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
            elif isinstance(obj, py_sym_types):
                return obj
            else:
                return None

        meta = map_aggregate(result, extract_val)
        if meta is not None:
            n.meta['val'] = meta
            if (shape_env := self._mode.shape_env) and (symbol_to_path := compute_unbacked_bindings(shape_env, result)):
                n.meta["unbacked_bindings"] = symbol_to_path

        return result

    def propagate(self, *args):
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    def propagate_dont_convert_inputs(self, *args):
        with self._mode:
            return super().run(*args)

    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        devices = [getattr(arg, "device", None) for arg in args if hasattr(arg, "device")]
        if devices and devices[0] and torch.is_autocast_enabled(devices[0].type):
            with torch.amp.autocast(devices[0].type, enabled=False):
                return super().call_function(target, args, kwargs)
        else:
            return super().call_function(target, args, kwargs)
