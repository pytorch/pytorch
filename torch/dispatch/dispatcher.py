import torch
import torch._C as _C
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch.utils._pytree import tree_flatten
from torch.overrides import handle_torch_function, has_torch_function

SUPPORTED_KEYS = {
    DispatchKey.CPU,
    DispatchKey.BackendSelect,
    DispatchKey.ADInplaceOrView,
    DispatchKey.AutogradCPU,
    DispatchKey.Python,
    DispatchKey.PythonTLSSnapshot,
}

"""
This is a dispatcher (in Python)
- You can define new operations (in Python) without schemas
- It interfaces with the PyTorch dispatcher
"""

class PyDispatcher:
    def __init__(self):
        self.current_dispatching_op = None
        self.already_dispatched_keys = None
        self.current_key = None

    def call(self, operator, args, kwargs):
        try:
            key = compute_dispatch_key(operator, args, kwargs, self.current_key)
            self.record_dispatch(key, operator)
            print(f"PyDispatcher.call {key}")
            return dispatch(key, operator, args, kwargs)
        finally:
            self.reset_dispatch_record()

    def redispatch(self, operator, args, kwargs):
        # Redispatch doesn't go to the top
        assert operator == self.currently_dispatching_op
        key = compute_dispatch_key(operator, args, kwargs, self.current_key, self.already_dispatched_keys)
        self.record_dispatch(key, operator)
        print(f"PyDispatcher.redispatch {key}")
        return dispatch(key, operator, args, kwargs)

    def reset_dispatch_record(self):
        self.current_dispatching_op = None
        self.already_dispatched_keys = None

    def record_dispatch(self, dispatch_key, operator):
        self.currently_dispatching_op = operator
        self.current_key = dispatch_key
        if self.already_dispatched_keys is None:
            self.already_dispatched_keys = DispatchKeySet(dispatch_key)
        else:
            self.already_dispatched_keys = self.already_dispatched_keys | DispatchKeySet(dispatch_key)


dispatcher_singleton = PyDispatcher()


def compute_dispatch_key(operator, args, kwargs, current_key, additional_exclude=None):
    if current_key is not None and operator.entrance_rules[current_key]:
        return current_key

    tensors = get_tensors(args, kwargs)
    dispatch_key = key_extractor(tensors, additional_exclude)
    return dispatch_key


def dispatch(dispatch_key, operator, args, kwargs):
    if dispatch_key not in SUPPORTED_KEYS:
        raise RuntimeError(f'NYI: {dispatch_key} {SUPPORTED_KEYS}')
    assert dispatch_key in operator.table
    kernel = operator.table[dispatch_key]
    return kernel(*args, **kwargs)


def key_extractor(tensors, additional_exclude=None):
    key_set = _C._dispatch_tls_local_include_set()
    for tensor in tensors:
        key_set = key_set | _C._dispatch_keys(tensor)
    key_set = key_set - _C._dispatch_tls_local_exclude_set()
    if additional_exclude is not None:
        key_set = key_set - additional_exclude
    return key_set.highestPriorityTypeId()


def to_flat_tuple(args, kwargs):
    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    flat_all = flat_args + flat_kwargs
    return flat_all

def get_tensors(args, kwargs):
    flat_all = to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)
