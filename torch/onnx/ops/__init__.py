from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx.ops import _impl


if TYPE_CHECKING:
    from collections.abc import Sequence


_TORCH_DTYPE_TO_ONNX_DTYPE = {
    torch.float32: 1,  # FLOAT
    torch.uint8: 2,  # UINT8
    torch.int8: 3,  # INT8
    torch.uint16: 4,  # UINT16
    torch.int16: 5,  # INT16
    torch.int32: 6,  # INT32
    torch.int64: 7,  # INT64
    str: 8,  # STRING
    torch.bool: 9,  # BOOL
    torch.float16: 10,  # FLOAT16
    torch.double: 11,  # DOUBLE
    torch.uint32: 12,  # UINT32
    torch.uint64: 13,  # UINT64
    torch.complex64: 14,  # COMPLEX64
    torch.complex128: 15,  # COMPLEX128
    torch.bfloat16: 16,  # BFLOAT16
    torch.float8_e4m3fn: 17,  # FLOAT8E4M3FN
    torch.float8_e4m3fnuz: 18,  # FLOAT8E4M3FNUZ
    torch.float8_e5m2: 19,  # FLOAT8E5M2
    torch.float8_e5m2fnuz: 20,  # FLOAT8E5M2FNUZ
}


def _parse_domain_op_type(domain_op: str) -> tuple[str, str]:
    splitted = domain_op.split("::", 1)
    if len(splitted) == 1:
        domain = ""
        op_type = splitted[0]
    else:
        domain = splitted[0]
        op_type = splitted[1]
    return domain, op_type


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
    dtype: torch.dtype | int,
    shape: Sequence[int | torch.SymInt],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> torch.Tensor:
    if not isinstance(dtype, int):
        if dtype not in _TORCH_DTYPE_TO_ONNX_DTYPE:
            raise ValueError(f"Unsupported dtype: {dtype}")
        dtype = _TORCH_DTYPE_TO_ONNX_DTYPE[dtype]
    domain, op_type = _parse_domain_op_type(domain_op)
    encoded_attrs = _impl.encode_onnx_attrs(attrs)
    # TODO: Parse domain
    return _impl._symbolic(
        inputs,
        op_type,
        dtype,
        encoded_attrs.attr_tensors,
        shape=shape,
        attr_keys=encoded_attrs.attr_keys,
        attr_types=encoded_attrs.attr_types,
        attr_pos=encoded_attrs.attr_pos,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
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
    dtypes: Sequence[torch.dtype | int],
    shapes: Sequence[Sequence[int | torch.SymInt]],
    version: int | None = None,
    metadata_props: dict[str, str] | None = None,
) -> Sequence[torch.Tensor]:
    onnx_dtypes = []
    for dtype in dtypes:
        if not isinstance(dtype, int):
            if dtype not in _TORCH_DTYPE_TO_ONNX_DTYPE:
                raise ValueError(f"Unsupported dtype: {dtype}")
            onnx_dtypes.append(_TORCH_DTYPE_TO_ONNX_DTYPE[dtype])
        else:
            onnx_dtypes.append(dtype)
    domain, op_type = _parse_domain_op_type(domain_op)
    encoded_attrs = _impl.encode_onnx_attrs(attrs)
    # Use the size of dtypes to determine the number of outputs
    return _impl._symbolic_multi_out(
        inputs,
        op_type,
        onnx_dtypes,
        encoded_attrs.attr_tensors,
        shape=shapes,
        attr_keys=encoded_attrs.attr_keys,
        attr_ints=encoded_attrs.attr_ints,
        attr_floats=encoded_attrs.attr_floats,
        attr_strs=encoded_attrs.attr_strs,
        metadata_props_keys=metadata_props.keys() if metadata_props else [],
        metadata_props_values=metadata_props.values() if metadata_props else [],
        domain=domain,
        version=version,
    )
