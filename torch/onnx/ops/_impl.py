from collections.abc import Sequence
from typing import Any, Optional, Union
import torch


@torch.library.custom_op("onnx_symbolic::_onnx_symbolic_single_output", mutates_args=())
def _onnx_symbolic_single_output(
    op_type: str,
    inputs: Sequence[torch.Tensor],
    dtype: torch.dtype,
    shape: Sequence[int],
    attr_keys: Optional[Sequence[str]],
    attr_ints: Optional[Sequence[int]],
    attr_floats: Optional[Sequence[float]],
    attr_strs: Optional[Sequence[str]],
    attr_bools: Optional[Sequence[bool]],
    attr_tensors: Optional[Sequence[torch.Tensor]],
    domain: str = "",
    version: Optional[int] = None,
    matadata_props: Optional[dict[str, str]] = None,
) -> torch.Tensor:
    # TODO: Verify that shape supports SymInt
    return inputs



def _encode_onnx_attr_key(name:str, attr_type, positions) -> str:
    ...


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
