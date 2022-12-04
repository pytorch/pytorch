"""This file exports ONNX ops for opset 16.

Note [ONNX Operators that are added/updated in opset 16]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-16-of-the-default-onnx-operator-set
New operators:
    GridSample https://github.com/onnx/onnx/pull/3557

Updated operators:
    Identity
    If
    LeakyRelu
    Loop
    PRelu
    RoiAlign
    Scan
    ScatterElements
    ScatterND
    Where
    GreaterOrEqual
    LessOrEqual
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import _type_utils, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=16)


# note (mkozuki): Why `grid_sampler` instead of `grid_sample`?
# Because `torch.nn.functional.grid_sample` calls `torch.grid_sampler`.
@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
@_beartype.beartype
def grid_sampler(
    g: jit_utils.GraphContext,
    input,
    grid,
    mode_enum,
    padding_mode_enum,
    align_corners,
):
    mode_s = {v: k for k, v in GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg]
    padding_mode_s = {v: k for k, v in GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]  # type: ignore[call-arg]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("scatter", self, dim, index, src, overload_name="src")

    src_type = _type_utils.JitScalarType.from_value(
        src, _type_utils.JitScalarType.UNDEFINED
    )
    src_sizes = symbolic_helper._get_tensor_sizes(src)
    index_sizes = symbolic_helper._get_tensor_sizes(index)

    if len(src_sizes) != len(index_sizes):
        return symbolic_helper._unimplemented(
            "scatter_add",
            f"`index` ({index_sizes}) should have the same dimensionality as `src` ({src_sizes})",
        )

    if src_sizes != index_sizes:
        # In ONNX, src and index are required to be the same rank and shape
        # However, in PyTorch, src is only required to have the same rank as index,
        # and shape would be accomodated. In static shape, converter can apply Slice op
        # to accommodate. We use Slice to adjust to shape of src if it's not the same
        # as index.
        # More detail on: https://github.com/onnx/onnx/issues/4672
        axes = list()
        ends = list()
        # Align the dynamic sizes of src and index
        # NOTE: Even if users set src and index with different dynamic axes, they are
        # still expected to have the same shape in runtime in terms of ONNX spec.
        # So the usage of different shape of src and index on dynamic size is not
        # supported.
        # More detail on: https://github.com/onnx/onnx/issues/4672
        for idx, d in enumerate(index_sizes):
            if d is None or src_sizes[idx] == d:
                # 1. the axe with dynamic shape is ignored, and will be aligned by
                # setType later
                # 2. if the shape are the same, we don't need to slice
                continue
            if src_sizes[idx] < d:
                return symbolic_helper._unimplemented(
                    "scatter_add",
                    f"`index` ({index_sizes}) should have smaller or equal (<=) size at any dimension than `src` ({src_sizes})",
                )
            axes.append(idx)
            ends.append(d)
        starts = [0] * len(ends)
        if axes and starts and ends:
            src = symbolic_helper._slice_helper(
                g, src, axes=axes, starts=starts, ends=ends
            )

    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s="add")
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if _type_utils.JitScalarType.from_value(self) != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )

        return g.op(
            "ScatterElements",
            self,
            index,
            src,
            axis_i=dim,
            reduction_s="add",
        )
