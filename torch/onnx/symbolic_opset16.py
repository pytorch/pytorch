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

import torch
from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import _type_utils, errors, symbolic_helper, utils
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
    # Check the input and grid tensor rank beforehand.
    if symbolic_helper._get_tensor_rank(input) == 5:
        return symbolic_helper._onnx_unsupported("GridSample with 5D volumetric input")
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

    # PyTorch only allows index shape <= src shape, so we can only consider
    # taking index as subset size to src, like PyTorch does. When sizes for src
    # and index are not matched or there are dynamic axes, we take index shape to
    # slice src to accommodate.
    if src_sizes != index_sizes or None in index_sizes:
        adjusted_shape = g.op("Shape", index)
        starts = g.op("Constant", value_t=torch.tensor([0] * len(index_sizes)))
        src = g.op("Slice", src, starts, adjusted_shape)

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


@_onnx_symbolic("aten::scatter_reduce")
@symbolic_helper.parse_args("v", "i", "v", "v", "s", "b")
@_beartype.beartype
def scatter_reduce(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    dim: int,
    index: torch._C.Value,
    src: torch._C.Value,
    reduce: str,
    include_self: bool,
):
    if reduce == "mean":
        raise errors.OnnxExporterError(
            "ONNX does not support mean reduction for scatter_reduce"
        )
    if not include_self:
        raise errors.OnnxExporterError(
            "ONNX does not support include_self=False for scatter_reduce"
        )

    reduce_mode = {  # convert torch string name to onnx string name
        "mean": "none",  # 'mean' doesn't support in ONNX 1.14 definition
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    onnx_reduce = reduce_mode[reduce]

    self_rank = g.op("Size", g.op("Shape", self))

    # if self_rank == 0:  # assert (index_rank == 0 and rank_src == 0)
    self_rank_is_zero = g.op(
        "Equal", self_rank, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
    )
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=3
    )
    neg_1 = if_context.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))

    self_reshape = if_context.op("Reshape", self, neg_1)
    utils._add_output_to_block(if_context.block, self_reshape)
    index_reshape = if_context.op("Reshape", index, neg_1)
    utils._add_output_to_block(if_context.block, index_reshape)
    src_reshape = if_context.op("Reshape", src, neg_1)
    utils._add_output_to_block(if_context.block, src_reshape)

    self_identity = else_context.op("Identity", self)
    utils._add_output_to_block(else_context.block, self_identity)
    index_identitye = else_context.op("Identity", index)
    utils._add_output_to_block(else_context.block, index_identitye)
    src_identity = else_context.op("Identity", src)
    utils._add_output_to_block(else_context.block, src_identity)

    result = g.op("ScatterElements", *if_op, axis_i=dim, reduction_s=onnx_reduce)

    # if self_rank == 0:
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=1
    )
    result_squeezed = if_context.op("Squeeze", result)
    utils._add_output_to_block(if_context.block, result_squeezed)
    result_identity = else_context.op("Identity", result)
    utils._add_output_to_block(else_context.block, result_identity)
    result_final = if_op.node().output()

    return result_final
