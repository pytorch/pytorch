import torch
from torch.utils._pytree import tree_map

schema_check_recorded_ops = []

# This Tensor Subclass is used to verify op schemas
# This Tensor currently:
#  - Records the called ops and appends to schema_check_records_ops

class SchemaCheckTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem):
        # The wrapping tensor (SchemaCheckTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"SchemaCheckTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"SchemaCheckTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e

        global schema_check_recorded_ops
        schema_check_recorded_ops.append(func.__name__)
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        return tree_map(wrap, out)

def reset_cache():
    global schema_check_recorded_ops
    schema_check_recorded_ops.clear()

def display_ops():
    print(*schema_check_recorded_ops, sep=",")
