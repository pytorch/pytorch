from collections.abc import Sequence
from typing import Any, Optional, Union, Literal
import torch
import typing

from . import _impl


def symbolic(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | torch.Tensor
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool]
        | Sequence[torch.Tensor],
    ]
    | None = None,
    *,
    dtype: torch.dtype,
    shape: Sequence[int | torch.SymInt],
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
    num_outputs: int = 1,
) -> torch.Tensor:
    return _impl._symbolic(...)



def symbolic_multi_out(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor],
    attrs: dict[
        str,
        int
        | float
        | str
        | bool
        | torch.Tensor
        | Sequence[int]
        | Sequence[float]
        | Sequence[str]
        | Sequence[bool]
        | Sequence[torch.Tensor],
    ]
    | None = None,
    *,
    dtype: torch.dtype,
    shape: Sequence[int | torch.SymInt],
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
    num_outputs: int,
) -> Sequence[torch.Tensor]: ...
    return _impl._symbolic_multi_out(...)
