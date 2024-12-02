from typing import List, Tuple

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        todo: List[torch.Tensor] = list(tensors)
        for tp, meta, inner_tensors in reversed(self.rebuild_stack):
            nb_tensor: int = len(inner_tensors)
            d = {a: b for a, b in zip(inner_tensors, todo[-nb_tensor:])}  # noqa: C416
            todo = todo[nb_tensor:]
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
            inner_tensors, metadata = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            rebuild_stack.append((type(obj), metadata, inner_tensors))
            for attr_name in inner_tensors:
                val = getattr(obj, attr_name)
                if type(val) is torch.Tensor:
                    plain_tensors.append(val)
                else:
                    assert isinstance(val, torch.Tensor)
                    todo.append(val)

        self.rebuild_stack = rebuild_stack

        return plain_tensors


def unwrap_tensor_subclass_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    name_param: List[Tuple[str, torch.nn.Parameter]] = list(model.named_parameters())
    for name, param in name_param:
        if is_traceable_wrapper_subclass(param):
            torch.nn.utils.parametrize.register_parametrization(
                model, name, UnwrapTensorSubclass()
            )
    return model
