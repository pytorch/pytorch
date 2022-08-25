from torch._dispatch._dispatcher import PyDispatcher, to_flat_tuple, compute_keyset
from torch.nn.functional import handle_torch_function
from torch.overrides import has_torch_function
import torch._C as _C

class PyOperator:
    def __init__(self, name):
        self.name = name
        self.table = {}

        self.__name__ = name

    def impl(self, dispatch_key, fn):
        assert dispatch_key not in self.table
        if fn is fallthrough_fn:
            self.table[dispatch_key] = fn(self, dispatch_key)
        else:
            self.table[dispatch_key] = fn

    def lookup(self, keyset):
        dispatch_key = keyset.highestPriorityTypeId()
        return self.table[dispatch_key]

    def __call__(self, *args, **kwargs):
        flat_args = to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        return PyDispatcher.call(self, *args, **kwargs)

def fallthrough_fn(operator, dispatch_key):
    def inner(*args, **kwargs):
        all_keys_after_current = _C._dispatch_keyset_full_after(dispatch_key)
        all_keys_after_current_masked = all_keys_after_current & compute_keyset(args, kwargs)
        return PyDispatcher.redispatch(operator, all_keys_after_current_masked, *args, **kwargs)
    return inner
