from typing import Optional

import torch.fx
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import snapshot_fake
from torch.utils._pytree import tree_map

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
    def __init__(self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None, *, check_consistency: bool = True):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        self.check_consistency = check_consistency

    def run_node(self, n: Node):
        from torch.fx.experimental.symbolic_shapes import rename_unbacked_to

        result = super().run_node(n)

        nil = object()
        # TODO: do boolean equality test too, see
        # https://github.com/pytorch/pytorch/issues/124110
        scalar_types = (torch.SymInt, torch.SymFloat, int, float)

        def check_consistent(new, old=nil):
            if isinstance(new, torch.Tensor):
                if old is not nil and self.check_consistency:
                    assert isinstance(old, torch.Tensor)
                    torch._check(old.dim() == new.dim())
                    # Do this manually so that each individual test is irrefutable
                    # (TODO: should be a helper for this, maybe sym_eq?  That
                    # gives us a compound expression and I'm not sure it
                    # simplifies right now)
                    for i, j in zip(old.shape, new.shape):
                        rename_unbacked_to(i, j)
                if isinstance(new, FakeTensor):
                    return snapshot_fake(new)
                else:
                    # TODO: How is it possible that we get a non fake tensor?  We
                    # should be running under the mode...
                    return snapshot_fake(self._mode.from_tensor(new, static_shapes=True))
            elif isinstance(new, scalar_types):
                if old is not nil and self.check_consistency:
                    assert isinstance(old, scalar_types)
                    rename_unbacked_to(old, new)
                return new
            else:
                return None

        meta_arg = []
        if 'val' in n.meta and n.meta['val'] is not None:
            meta_arg = [n.meta['val']]

        meta = tree_map(check_consistent, result, *meta_arg)
        if meta is not None:
            n.meta['val'] = meta
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
