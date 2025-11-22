"""Implementation of symbolic FX ops to represent arbitrary ONNX ops.

This module provides a way to create symbolic FX operators that can represent
arbitrary ONNX operators.

The operators are called "symbolic" because they don't do any actual computation
but instead serve as placeholders in the computation graph.

Each implementation contains two parts: A "real" implementation that produce all
zeros based on the input shape and dtype, and a "fake" implementation that does more
or less the same thing but is required by the `torch.library.custom_op` interface.
"""

# flake8: noqa: B950
import dataclasses
from collections.abc import Sequence
from typing import Optional, Union

import torch
from torch.onnx.ops import _dtype_mappings


_INT_TYPE = "i"
_FLOAT_TYPE = "f"
_STRING_TYPE = "s"
_INT_SEQ_TYPE = "is"
_FLOAT_SEQ_TYPE = "fs"
_STRING_SEQ_TYPE = "ss"


@dataclasses.dataclass
class EncodedAttrs:
    """Class to encode attributes from dictionary into lists of FX compatible attributes.

    Since FX does not support dictionaries, we need to encode the attributes into
    lists. This class provides a way to encode and decode the attributes.

    Attributes:
        attr_keys: List of attribute keys.
        attr_types: List of attribute types. Values can be "i" (int), "f" (float),
            "s" (string), "is" (int sequence), "fs" (float sequence), or "ss" (string sequence).
        attr_pos: List of tuples representing the start and end positions of each
            attribute in the corresponding list.
        attr_ints: List of integer attributes.
        attr_floats: List of float attributes.
        attr_strs: List of string attributes.
    """

    attr_keys: list[str]
    attr_types: list[str]
    attr_pos: list[tuple[int, int]]
    attr_ints: list[int]
    attr_floats: list[float]
    attr_strs: list[str]

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
                Sequence[int],
                Sequence[float],
                Sequence[str],
                Sequence[bool],
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
        )
        for k, v in attrs.items():
            encoded.attr_keys.append(k)
            if isinstance(v, int):
                start_pos = len(encoded.attr_ints)
                encoded.attr_ints.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_INT_TYPE)
            elif isinstance(v, float):
                start_pos = len(encoded.attr_floats)
                encoded.attr_floats.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_FLOAT_TYPE)
            elif isinstance(v, str):
                start_pos = len(encoded.attr_strs)
                encoded.attr_strs.append(v)
                encoded.attr_pos.append((start_pos, start_pos + 1))
                encoded.attr_types.append(_STRING_TYPE)
            elif isinstance(v, Sequence):
                if len(v) == 0:
                    raise ValueError(f"Empty sequence for attribute {k}")
                if any(isinstance(elem, float) for elem in v):
                    start_pos = len(encoded.attr_floats)
                    encoded.attr_floats.extend([float(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_FLOAT_SEQ_TYPE)
                elif isinstance(v[0], int):
                    start_pos = len(encoded.attr_ints)
                    encoded.attr_ints.extend([int(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_INT_SEQ_TYPE)
                elif isinstance(v[0], str):
                    start_pos = len(encoded.attr_strs)
                    encoded.attr_strs.extend([str(elem) for elem in v])
                    encoded.attr_pos.append((start_pos, start_pos + len(v)))
                    encoded.attr_types.append(_STRING_SEQ_TYPE)
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
            list[int],
            list[float],
            list[str],
        ],
    ]:
        """Convert the encoded attributes back to a dictionary for creating an ONNX node."""
        attrs: dict[
            str,
            Union[
                int,
                float,
                str,
                list[int],
                list[float],
                list[str],
            ],
        ] = {}
        for i, key in enumerate(self.attr_keys):
            attr_type = self.attr_types[i]
            if attr_type == _INT_TYPE:
                attrs[key] = self.attr_ints[self.attr_pos[i][0]]
            elif attr_type == _FLOAT_TYPE:
                attrs[key] = self.attr_floats[self.attr_pos[i][0]]
            elif attr_type == _STRING_TYPE:
                attrs[key] = self.attr_strs[self.attr_pos[i][0]]
            elif attr_type == _FLOAT_SEQ_TYPE:
                attrs[key] = self.attr_floats[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == _INT_SEQ_TYPE:
                attrs[key] = self.attr_ints[self.attr_pos[i][0] : self.attr_pos[i][1]]
            elif attr_type == _STRING_SEQ_TYPE:
                attrs[key] = self.attr_strs[self.attr_pos[i][0] : self.attr_pos[i][1]]
            else:
                raise ValueError(f"Unsupported attribute type: {attr_type}")
        return attrs


@torch.library.custom_op(
    "onnx_symbolic::_symbolic",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int onnx_dtype, *,"
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
        onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
        lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    return torch.zeros(
        shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
    )


@_symbolic.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtype: int,
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
        onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
        lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
    )
    # NOTE(justinchuby): Use zeros instead of torch.empty because I haven't figured
    # out how it can handle empty shapes
    return torch.zeros(
        shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
    )


@torch.library.custom_op(
    "onnx_symbolic::_symbolic_multi_out",
    mutates_args=(),
    schema=(
        "(Tensor?[] inputs, str op_type, int[] onnx_dtypes, *,"
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
    torch._check(
        len(shapes) == len(onnx_dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of ONNX dtypes ({len(onnx_dtypes)})",
    )
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
            lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        outputs.append(
            torch.zeros(
                shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
            )
        )
    return outputs


@_symbolic_multi_out.register_fake
def _(
    inputs: Sequence[torch.Tensor],
    op_type: str,
    onnx_dtypes: Sequence[int],
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
    torch._check(
        len(shapes) == len(onnx_dtypes),
        lambda: f"Number of shapes ({len(shapes)}) must match number of ONNX dtypes ({len(onnx_dtypes)})",
    )
    for shape, onnx_dtype in zip(shapes, onnx_dtypes):
        torch._check(
            onnx_dtype in _dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE,
            lambda: f"{onnx_dtype} is invalid as an ONNX data type. Valid values are {list(_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE.keys())}",
        )
        # NOTE(justinchuby): Use zeros instead of torch.empty because I haven't figured
        # out how it can handle empty shapes
        outputs.append(
            torch.zeros(
                shape, dtype=_dtype_mappings.ONNX_DTYPE_TO_TORCH_DTYPE[onnx_dtype]
            )
        )
    return outputs
