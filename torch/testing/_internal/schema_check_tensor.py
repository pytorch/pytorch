import torch
from torch.utils._pytree import tree_map, tree_flatten
from torch.fx.operator_schemas import get_signature_for_torch_op
from torch.testing._internal.jit_utils import clone_inputs
from typing import Any, Callable, Dict, Tuple

# This Tensor Subclass is used to verify op schemas
# This Tensor currently:
#  - Records the called ops and appends to schema_check_records_ops
#  - Checks for mutations on all inputs

class SchemaCheckTensor(torch.Tensor):
    elem: torch.Tensor

    recorded_ops = []

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

    @staticmethod
    def reset_cache():
        SchemaCheckTensor.recorded_ops.clear()

    @staticmethod
    def display_ops():
        print(*recorded_ops, sep=",")

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

        def has_mutated(before, after):
            return not torch.equal(before, after) if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor) else False

        SchemaCheckTensor.recorded_ops.append(func.__name__)
        schema, arguments = find_matching_schema(func, tree_map(unwrap, args), tree_map(unwrap, kwargs))
        cloned_arguments = dict(zip(arguments.keys(), clone_inputs(arguments.values())))
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for argument in schema.arguments:
            name = argument.name if argument.name != "self" else "input"
            if arguments.get(name) is not None:
                before = tree_flatten(arguments.get(name))[0]
                after = tree_flatten(cloned_arguments.get(name))[0]
                if (any([has_mutated(i, j) for i, j in zip(before, after)]) and not argument.is_mutable):
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
        if (len(matched_schemas) == 1):
            return matched_schemas[0]
        else:
            raise RuntimeError('More than one matching schema found')
