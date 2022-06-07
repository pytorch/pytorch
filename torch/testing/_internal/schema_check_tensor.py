import torch
from torch.utils._pytree import tree_map
from torch.fx.operator_schemas import get_signature_for_torch_op
from torch.testing._internal.jit_utils import clone_inputs
from torch.testing._internal.common_utils import is_iterable_of_dtype
from typing import Any, Callable, Dict, Tuple

schema_check_recorded_ops = []

# This Tensor Subclass is used to verify op schemas
# This Tensor currently:
#  - Records the called ops and appends to schema_check_records_ops
#  - Checks for mutations on all inputs

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
            if isinstance(e, cls):
                return e.elem
            elif is_iterable_of_dtype(e, cls):
                return [t.elem for t in e]
            else:
                return e

        def wrap(e):
            if isinstance(e, torch.Tensor):
                return cls(e)
            elif is_iterable_of_dtype(e, torch.Tensor):
                return [cls(t) for t in e]
            else:
                return e

        def has_mutated(before, after):
            if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
                return not torch.equal(before, after)
            elif is_iterable_of_dtype(before, torch.Tensor) and is_iterable_of_dtype(after, torch.Tensor):
                before_list = list(before)
                after_list = list(after)
                if (len(before_list) != len(after_list)):
                    return True
                for before_elem, after_elem in zip(before_list, after_list):
                    if (not torch.equal(before_elem, after_elem)):
                        return True
                return False
            else:
                return False

        global schema_check_recorded_ops
        schema_check_recorded_ops.append(func.__name__)
        schema, arguments = find_matching_schema(func, tree_map(unwrap, args), tree_map(unwrap, kwargs))
        cloned_arguments = dict(zip(arguments.keys(), clone_inputs(arguments.values())))
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for argument in schema.arguments:
            name = argument.name if argument.name != "self" else "input"
            if (arguments.get(name) is not None and cloned_arguments.get(name) is not None):
                if (has_mutated(arguments.get(name), cloned_arguments.get(name)) and not argument.is_mutable):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
        return tree_map(wrap, out)

def find_matching_schema(op: Callable, args : Tuple[Any], kwargs : Dict[str, Any]):
    signatures, schemas = get_signature_for_torch_op(op, return_schemas=True)
    if (signatures and schemas):
        matched_schemas = []
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                candidate_bound_arguments = candidate_signature.bind(*args, **kwargs)
                matched_schemas.append((schema, candidate_bound_arguments.arguments))
            except TypeError as e:
                continue

        if (len(matched_schemas) == 0):
            raise RuntimeError('No matching schema found')
            return None
        if (len(matched_schemas) == 1):
            return matched_schemas[0]
        else:
            raise RuntimeError('More than one matching schema found')
            return None

def reset_cache():
    global schema_check_recorded_ops
    schema_check_recorded_ops.clear()

def display_ops():
    print(*schema_check_recorded_ops, sep=",")
