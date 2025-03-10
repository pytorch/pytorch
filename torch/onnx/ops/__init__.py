from collections.abc import Sequence
from typing import Any, Optional, Union, Literal
import torch
import typing

from . import _impl


@typing.overload
def onnx_symbolic(
    op_type: str,
    inputs: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    shape: Sequence[int | torch.SymInt],
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
    domain: str = "",
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
    num_outputs: Literal[1] = 1,
) -> torch.Tensor: ...


@typing.overload
def onnx_symbolic(
    op_type: str,
    inputs: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    shape: Sequence[int | torch.SymInt],
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
    domain: str = "",
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
    num_outputs: int,
) -> Sequence[torch.Tensor]: ...


def onnx_symbolic(
    op_type: str,
    inputs: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype,
    shape: Sequence[int | torch.SymInt],
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
    domain: str = "",
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
    num_outputs: int = 1,
) -> torch.Tensor | Sequence[torch.Tensor]:
    if num_outputs == 1:
        return _impl._onnx_symbolic_single_output(...)
