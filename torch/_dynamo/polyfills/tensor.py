from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import torch

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor


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


def dtensor_full_via_local_full(
    fill_value: DTensor, size: Any, **kwargs: Any
) -> DTensor:
    """torch.full for replicated DTensor fill_value via local full + DTensor.from_local."""
    import torch.distributed.tensor as dt

    names = kwargs.pop("names", None)
    if names is not None:
        raise RuntimeError(
            "torch.full(..., names=...) with DTensor fill_value "
            "is not supported under torch.compile."
        )
    dtype = kwargs.pop("dtype", None)
    layout = kwargs.pop("layout", torch.strided)
    requires_grad = kwargs.pop("requires_grad", False)
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    if kwargs:
        raise RuntimeError(
            "torch.full with DTensor fill_value: unsupported keyword arguments "
            f"{sorted(kwargs)!r}"
        )

    local = torch.full(
        cast(Any, size),
        cast(Any, fill_value.to_local()),
        dtype=dtype,
        layout=layout,
        device=fill_value.device,
        requires_grad=requires_grad,
    )
    return dt.DTensor.from_local(
        local,
        device_mesh=fill_value.device_mesh,
        placements=list(fill_value.placements),
        run_check=False,
    )


__all__ = [
    "make_subclass",
]
