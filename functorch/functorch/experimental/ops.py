from torch.dispatch.dispatcher import dispatcher_singleton, to_flat_tuple, has_torch_function, compute_keyset
from torch._C import DispatchKey, DispatchKeySet
from torch.nn.functional import handle_torch_function

import torch._C as _C

class PyOperator:
    def __init__(self, name):
        self.name = name
        self.table = {}

        # TODO: torch_dispatch expects PyOperator to be an instance of a torch.ops.aten op.
        self.overloadpacket = self
        self.__name__ = name

    def impl(self, dispatch_key, fn):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fn

    def lookup(self, keyset):
        dispatch_key = keyset.highestPriorityTypeId()
        return self.table[dispatch_key]

    def fallthrough(self, dispatch_key):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fallthrough_fn(self, dispatch_key)

    def __call__(self, *args, **kwargs):
        flat_args = to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        return dispatcher_singleton.call(self, args, kwargs)

def fallthrough_fn(operator, dispatch_key):
    def inner(*args, **kwargs):
        all_keys_sans_current = _C._dispatch_keyset_full_after(dispatch_key)
        all_keys_sans_current_masked = all_keys_sans_current & compute_keyset(args, kwargs)
        return dispatcher_singleton.redispatch(operator, all_keys_sans_current_masked, args, kwargs)
    return inner
