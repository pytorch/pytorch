from collections.abc import Sequence
from typing import Any, Optional, Union
import torch


_ONNX_DTYPE_TO_TORCH_DTYPE = {
    1: torch.float32,  # FLOAT
    2: torch.uint8,  # UINT8
    3: torch.int8,  # INT8
    4: torch.uint16,  # UINT16
    5: torch.int16,  # INT16
    6: torch.int32,  # INT32
    7: torch.int64,  # INT64
    8: str,  # STRING
    9: torch.bool,  # BOOL
    10: torch.float16,  # FLOAT16
    11: torch.double,  # DOUBLE
    12: torch.uint32,  # UINT32
    13: torch.uint64,  # UINT64
    14: torch.complex64,  # COMPLEX64
    15: torch.complex128,  # COMPLEX128
    16: torch.bfloat16,  # BFLOAT16
    17: torch.float8_e4m3fn,  # FLOAT8E4M3FN
    18: torch.float8_e4m3fnuz,  # FLOAT8E4M3FNUZ
    19: torch.float8_e5m2,  # FLOAT8E5M2
    20: torch.float8_e5m2fnuz,  # FLOAT8E5M2FNUZ
    21: torch.uint8,  # UINT4
    22: torch.uint8,  # INT4
    23: torch.uint8,  # FLOAT4E2M1
}

@torch.library.custom_op(
    "onnx_symbolic::_symbolic",
    mutates_args=(),
    schema="(Tensor[] inputs, str op_type, int onnx_dtype, Tensor[] attr_tensors, *, SymInt[] shape, str[] attr_keys, int[] attr_ints, float[] attr_floats, str[] attr_strs, bool[] attr_bools, str domain='', int? version=None) -> Tensor",
)
def _symbolic(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
    attr_tensors: Sequence[torch.Tensor],
    *,
    shape: Sequence[int | torch.SymInt],
    attr_keys: Sequence[str],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    attr_bools: Sequence[bool],
    domain: str = "",
    version: Optional[int] = None,
) -> torch.Tensor:
    # TODO: Verify that shape supports SymInt
    return torch.zeros(shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype])


@_symbolic.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
    attr_tensors: Sequence[torch.Tensor]=(),
    *,
    shape: Sequence[int | torch.SymInt],
    attr_keys: Sequence[str],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    attr_bools: Sequence[bool],
    domain: str = "",
    version: Optional[int] = None,
) -> torch.Tensor:
    return torch.empty(*shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype])


torch.library.opcheck(
    _symbolic,
    (
        [torch.tensor(1)],
        "Add",
        1,
        [torch.tensor(42)]
    ),
    dict(
        shape=[
            1,
        ],
        attr_keys=["key"],
        attr_ints=[1],
        attr_floats=[1.0],
        attr_strs=["attr"],
        attr_bools=[True],
        domain="",
        version=1,
    )
)


# @torch.library.custom_op("onnx_symbolic::_symbolic", mutates_args=())
# def _symbolic_multi_out(
#     inputs: Sequence[torch.Tensor],
#     *,
#     op_type: str,
#     # dtype: Sequence[torch.dtype],
#     shape: Sequence[int],
#     num_outputs: int,
#     attr_keys: Optional[Sequence[str]],
#     attr_ints: Optional[Sequence[int]],
#     attr_floats: Optional[Sequence[float]],
#     attr_strs: Optional[Sequence[str]],
#     attr_bools: Optional[Sequence[bool]],
#     attr_tensors: Optional[Sequence[torch.Tensor]],
#     domain: str = "",
#     version: Optional[int] = None,
#     matadata_props: Optional[dict[str, str]] = None,
# ) -> torch.Tensor:
#     # TODO: Verify that shape supports SymInt
#     return inputs


def _encode_onnx_attr_key(name: str, attr_type, positions) -> str: ...


def _encode_onnx_attrs(
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
    ],
) -> tuple[
    Optional[Sequence[str]],
    Optional[Sequence[int]],
    Optional[Sequence[float]],
    Optional[Sequence[str]],
    Optional[Sequence[bool]],
    Optional[Sequence[torch.Tensor]],
]:
    attr_keys: Optional[Sequence[str]] = []
    attr_ints: Optional[Sequence[int]] = []
    attr_floats: Optional[Sequence[float]] = []
    attr_strs: Optional[Sequence[str]] = []
    attr_bools: Optional[Sequence[bool]] = []
    attr_tensors: Optional[Sequence[torch.Tensor]] = []

    for i, (k, v) in enumerate(attrs.items()):
        if isinstance(v, int):
            attr_ints.append(v)
