import dataclasses
from collections.abc import Sequence
from typing import Optional, Union

import torch


_ONNX_DTYPE_TO_TORCH_DTYPE: dict[int, torch.dtype] = {
    1: torch.float32,  # FLOAT
    2: torch.uint8,  # UINT8
    3: torch.int8,  # INT8
    4: torch.uint16,  # UINT16
    5: torch.int16,  # INT16
    6: torch.int32,  # INT32
    7: torch.int64,  # INT64
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


@dataclasses.dataclass
class EncodedAttrs:
    """Class to encode attributes from dictionary into lists of FX compatible attributes."""

    attr_keys: list[str]
    attr_types: list[str]
    attr_pos: list[tuple[int, int]]
    attr_ints: list[int]
    attr_floats: list[float]
    attr_strs: list[str]
    attr_tensors: list[torch.Tensor]

    @classmethod
    def from_dict(
        cls,
        attrs: dict[
            str,
            Union[
                int,
                float,
                str,
                bool,
                torch.Tensor,
                Sequence[int],
                Sequence[float],
                Sequence[str],
                Sequence[bool],
                Sequence[torch.Tensor],
            ],
        ],
    ) -> "EncodedAttrs":
        encoded = cls(
            attr_keys=[],
            attr_types=[],
            attr_pos=[],
            attr_ints=[],
            attr_floats=[],
            attr_strs=[],
            attr_tensors=[],
        )
        for i, (k, v) in enumerate(attrs.items()):
            encoded.attr_keys.append(k)
            if isinstance(v, int):
                start_pos = len(encoded.attr_ints)
                encoded.attr_ints.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append("i")
            elif isinstance(v, float):
                start_pos = len(encoded.attr_floats)
                encoded.attr_floats.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append("f")
            elif isinstance(v, str):
                start_pos = len(encoded.attr_strs)
                encoded.attr_strs.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append("s")
            elif isinstance(v, torch.Tensor):
                start_pos = len(encoded.attr_tensors)
                encoded.attr_tensors.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append("t")
            elif isinstance(v, Sequence):
                if len(v) == 0:
                    raise ValueError(f"Empty sequence for attribute {k}")
                if any(isinstance(elem, float) for elem in v):
                    start_pos = len(encoded.attr_floats)
                    encoded.attr_floats.extend([float(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append("fs")
                elif isinstance(v[0], int):
                    start_pos = len(encoded.attr_ints)
                    encoded.attr_ints.extend([int(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append("is")
                elif isinstance(v[0], str):
                    start_pos = len(encoded.attr_strs)
                    encoded.attr_strs.extend([str(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append("ss")
                elif isinstance(v[0], torch.Tensor):
                    start_pos = len(encoded.attr_tensors)
                    encoded.attr_tensors.extend([torch.tensor(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append("ts")
                else:
                    raise ValueError(f"Unsupported sequence type for attribute {k}")
            else:
                raise ValueError(f"Unsupported attribute type for {k}: {type(v)}")
        assert len(encoded.attr_keys) == len(encoded.attr_types), (
            f"Mismatch between number of attribute keys and types: {len(encoded.attr_keys)} != {len(encoded.attr_types)}"
        )
        assert len(encoded.attr_keys) == len(encoded.attr_pos), (
            f"Mismatch between number of attribute keys and positions: {len(encoded.attr_keys)} != {len(encoded.attr_pos)}"
        )
        return encoded

    def to_dict(
        self,
    ) -> dict[
        str,
        Union[
            int,
            float,
            str,
            torch.Tensor,
            list[int],
            list[float],
            list[str],
            list[torch.Tensor],
        ],
    ]:
        """Convert the encoded attributes back to a dictionary for creating an ONNX node."""
        attrs: dict[
            str,
            Union[
                int,
                float,
                str,
                torch.Tensor,
                list[int],
                list[float],
                list[str],
                list[torch.Tensor],
            ],
        ] = {}
        for i, key in enumerate(self.attr_keys):
            attr_type = self.attr_types[i]
            if attr_type == "i":
                attrs[key] = self.attr_ints[self.attr_pos[i][0]]
            elif attr_type == "f":
                attrs[key] = self.attr_floats[self.attr_pos[i][0]]
            elif attr_type == "s":
                attrs[key] = self.attr_strs[self.attr_pos[i][0]]
            elif attr_type == "t":
                attrs[key] = self.attr_tensors[self.attr_pos[i][0]]
            elif attr_type == "fs":
                attrs[key] = self.attr_floats[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == "is":
                attrs[key] = self.attr_ints[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == "ss":
                attrs[key] = self.attr_strs[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == "ts":
                attrs[key] = self.attr_tensors[
                    self.attr_pos[i][0] : self.attr_pos[i][1]
                ]
            else:
                raise ValueError(f"Unsupported attribute type: {attr_type}")
        return attrs


@torch.library.custom_op(
    "onnx_symbolic::_symbolic",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int onnx_dtype, Tensor[] attr_tensors, *,"
        " SymInt[] shape, str[] attr_keys, str[] attr_types, int[][] attr_pos,"
        " int[] attr_ints, float[] attr_floats, str[] attr_strs, str[] metadata_props_keys,"
        " str[] metadata_props_values, str domain='', int? version=None"
        ") -> Tensor"
    ),
)
def _symbolic(
    inputs: Sequence[Optional[torch.Tensor]],
    op_type: str,
    onnx_dtype: int,
    attr_tensors: Sequence[torch.Tensor],
    *,
    shape: Sequence[Union[int, torch.SymInt]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: Optional[int] = None,
) -> torch.Tensor:
    # TODO: Verify that shape supports SymInt
    torch._check(
        onnx_dtype in _ONNX_DTYPE_TO_TORCH_DTYPE,
        f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    return torch.zeros(shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype])


@_symbolic.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
    attr_tensors: Sequence[torch.Tensor] = (),
    *,
    shape: Sequence[Union[int, torch.SymInt]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: Optional[int] = None,
) -> torch.Tensor:
    torch._check(
        onnx_dtype in _ONNX_DTYPE_TO_TORCH_DTYPE,
        f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    return torch.zeros(shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype])


torch.library.opcheck(
    _symbolic,
    ([torch.tensor(1)], "Add", 1, [torch.tensor(42)]),
    dict(
        shape=[
            1,
        ],
        attr_keys=["key"],
        attr_types=["i"],
        attr_pos=[(0, 1)],
        attr_ints=[1],
        attr_floats=[1.0],
        attr_strs=["attr"],
        metadata_props_keys=["meta_key"],
        metadata_props_values=["meta_value"],
        domain="",
        version=1,
    ),
)


@torch.library.custom_op(
    "onnx_symbolic::_symbolic_multi_out",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int[] onnx_dtypes, Tensor[] attr_tensors, *,"
        " SymInt[][] shapes, str[] attr_keys, str[] attr_types, int[][] attr_pos,"
        " int[] attr_ints, float[] attr_floats, str[] attr_strs, str[] metadata_props_keys,"
        " str[] metadata_props_values, str domain='', int? version=None"
        ") -> Tensor[]"
    ),
)
def _symbolic_multi_out(
    inputs: Sequence[Optional[torch.Tensor]],
    op_type: str,
    onnx_dtypes: Sequence[int],
    attr_tensors: Sequence[torch.Tensor],
    *,
    shapes: Sequence[Sequence[Union[int, torch.SymInt]]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: Optional[int] = None,
) -> list[torch.Tensor]:
    outputs = []
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _ONNX_DTYPE_TO_TORCH_DTYPE,
            f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        outputs.append(torch.zeros(shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]))
    return outputs


@_symbolic_multi_out.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtypes: Sequence[int],
    attr_tensors: Sequence[torch.Tensor],
    *,
    shapes: Sequence[Sequence[Union[int, torch.SymInt]]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: Optional[int] = None,
) -> list[torch.Tensor]:
    outputs = []
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _ONNX_DTYPE_TO_TORCH_DTYPE,
            f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        outputs.append(torch.zeros(shape, dtype=_ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]))
    return outputs


torch.library.opcheck(
    _symbolic_multi_out,
    ([torch.tensor(1)], "Add", [1], [torch.tensor(42)]),
    dict(
        shapes=[
            [
                1,
            ]
        ],
        attr_keys=["key"],
        attr_types=["i"],
        attr_pos=[(0, 1)],
        attr_ints=[1],
        attr_floats=[1.0],
        attr_strs=["attr"],
        metadata_props_keys=["meta_key"],
        metadata_props_values=["meta_value"],
        domain="",
        version=1,
    ),
)
