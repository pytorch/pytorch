from __future__ import annotations

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
    version: int | None = None,
    matadata_props: dict[str, str] | None = None,
) -> torch.Tensor:
    attr_keys, attr_ints, attr_floats, attr_strs, attr_bools, attr_tensors = (
        _impl.encode_onnx_attrs(attrs)
    )
    # TODO: Convert dtype
    return _impl._symbolic(
        inputs,
        domain_op,
        dtype,
        attr_tensors,
        shape=shape,
        attr_keys=attr_keys,
        attr_ints=attr_ints,
        attr_floats=attr_floats,
        attr_strs=attr_strs,
        attr_bools=attr_bools,
        metadata_props_keys=matadata_props.keys() if matadata_props else [],
        metadata_props_values=matadata_props.values() if matadata_props else [],
        domain=domain_op,
        version=version,
    )


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
    num_outputs: int,
    dtypes: torch.dtype,
    shapes: Sequence[Sequence[Union[int, torch.SymInt]]],
    version: int | None = None,
    matadata_props: dict[str, str] | None = None,
) -> Sequence[torch.Tensor]:
    attr_keys, attr_ints, attr_floats, attr_strs, attr_bools, attr_tensors = (
        _impl.encode_onnx_attrs(attrs)
    )
    return _impl._symbolic_multi_out(
        inputs,
        domain_op,
        dtypes,
        attr_tensors,
        shape=shapes,
        attr_keys=attr_keys,
        attr_ints=attr_ints,
        attr_floats=attr_floats,
        attr_strs=attr_strs,
        attr_bools=attr_bools,
        metadata_props_keys=matadata_props.keys() if matadata_props else [],
        metadata_props_values=matadata_props.values() if matadata_props else [],
        domain=domain_op,
        version=version,
    )
