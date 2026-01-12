from typing import Any

import torch
from ..decorators import substitute_in_graph


@substitute_in_graph(  # type: ignore[arg-type]
    torch.Tensor._make_subclass
)
def make_subclass(
    cls: type[Any], data: torch.Tensor, requires_grad: bool = False, **kwargs: Any
) -> Any:
    with torch._C.DisableTorchFunctionSubclass():
        # This is a rough approximation of `THPVariable_make_subclass`. It should
        # suffice for most of Dynamo tracing purposes.
        # https://github.com/pytorch/pytorch/blob/ccfde4dadfa3c342076a1ee387017f84dd4ad2f7/torch/csrc/autograd/python_variable.cpp#L597-L650
        assert len(kwargs) == 0, (
            "_make_subclass only supports requires_grad as keyword arg"
        )
        data = data.detach()

        # Avoid unnecessary `requires_grad` mutation, which isn't supported in Dynamo.
        if data.requires_grad != requires_grad:
            data.requires_grad = requires_grad

        # Dynamo can't yet handle upcasting to base tensor type via `as_subclass`.
        if cls is torch.Tensor:
            return torch.Tensor(data)

        # Calling `as_subclass` because
        # 1. Dynamo knows how to handle it
        # 2. the C impls match at this point -- both `THPVariable_make_subclass` and
        #    `THPVariable_as_subclass` calls `THPVariable_NewWithVar`.
        return data.as_subclass(cls)


__all__ = [
    "make_subclass",
]
