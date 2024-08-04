from typing import Any

from torch._parameter import (
    is_lazy,
    Parameter,
    UninitializedBuffer,
    UninitializedParameter,
)


__all__ = ["Parameter", "is_lazy", "UninitializedParameter", "UninitializedBuffer"]


def __getattr__(name: str) -> Any:
    import warnings

    from torch import _parameter

    try:
        member = getattr(_parameter, name)
    except AttributeError as ex:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from ex

    warnings.warn(
        f"torch.nn.parameter.{name} is not a public API of PyTorch.",
        stacklevel=2,
    )
    return member
