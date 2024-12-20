from typing import List, Tuple

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        todo: List[torch.Tensor] = list(tensors)
        for tp, meta, inner_tensors_attrs in reversed(self.rebuild_stack):
            num_children: int = len(inner_tensors_attrs)
            d = {  # noqa: C416
                a: b for a, b in zip(inner_tensors_attrs, todo[-num_children:])
            }
            todo = todo[:-num_children]
            rebuilt = tp.__tensor_unflatten__(d, meta, None, None)  # type: ignore[attr-defined]
            todo.append(rebuilt)

        assert len(todo) == 1
        return todo[0]

    def right_inverse(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        assert type(tensor) is not torch.Tensor
        rebuild_stack = []
        plain_tensors = []
        todo = [tensor]
        while todo:
            obj = todo.pop()
            inner_tensors_attrnames, metadata = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            inner_tensors_attrnames_stack_order = []
            subclasses_attrnames = []
            for attr_name in inner_tensors_attrnames:
                val = getattr(obj, attr_name)
                if type(val) is torch.Tensor:
                    plain_tensors.append(val)
                    inner_tensors_attrnames_stack_order.append(attr_name)
                else:
                    assert isinstance(val, torch.Tensor)
                    todo.append(val)
                    subclasses_attrnames.append(attr_name)
            inner_tensors_attrnames_stack_order.extend(subclasses_attrnames)
            rebuild_stack.append(
                (type(obj), metadata, inner_tensors_attrnames_stack_order)
            )

        self.rebuild_stack = rebuild_stack
        return plain_tensors


def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    name_param: List[Tuple[str, torch.nn.Parameter]] = list(
        module.named_parameters(recurse=False)
    )
    for name, param in name_param:
        if is_traceable_wrapper_subclass(param):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

    for name, child in module.named_children():
        unwrap_tensor_subclass_parameters(child)

    return module
