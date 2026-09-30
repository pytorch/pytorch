# mypy: allow-untyped-defs
"""Shared FlexGEMM local-reduce geometry, constants, and validation helpers."""

import dataclasses
from collections.abc import Sequence
from typing import Final

from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_FRAGMENT_WIDTH,
    GemmReductionGeometry,
)
from torch._inductor.kernel.gemm_epilogue_utils import statically_known
from torch._inductor.utils import _IntLike
from torch.types import IntLikeType


LOCAL_REDUCE_FEED_MAIN_ARG_NAME: Final = "local_reduce0"
LOCAL_REDUCE_PREPASS_FN_SUFFIX: Final = "_local_reduce_prepass"
LOCAL_REDUCE_STORE_ARG_NAME: Final = "local_reduce_store"


# Axis-0 feed-main reduces within one lane-layout M group. Axis-1 groups that
# fit one logical TensorSSA fragment use QuACK's in-kernel accumulator prepass;
# larger groups remain unsupported.
LOCAL_REDUCE_FRAGMENT_WIDTH = GEMM_REDUCTION_FRAGMENT_WIDTH
LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only same-warp axis-0 "
    f"groups <= {LOCAL_REDUCE_FRAGMENT_WIDTH}"
)
LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR = (
    "FlexGEMM local-reduce feed-main for axis-1 groups larger than one "
    "TensorSSA fragment is not supported yet"
)
LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR = (
    "local_reduce_group must divide the selected FlexGEMM output dimension"
)
LOCAL_REDUCE_GROUP_POSITIVE_ERROR = "local_reduce_group must be positive"
LOCAL_REDUCE_AXIS_ERROR = "local_reduce_axis must be 0 or 1"
LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR = (
    "FlexGEMM local reductions require group size greater than 1"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR = (
    "FlexGEMM local reductions larger than TensorSSA fragment width 32 "
    "require group size to be a multiple of 32"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR = (
    "FlexGEMM local reductions require group size to divide TensorSSA fragment width 32"
)
LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR = (
    "FlexGEMM local reductions do not support mixing grouped TensorSSA "
    "values with different grouped layouts"
)
LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR = (
    "FlexGEMM local reductions currently support only aten.mm"
)
LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR = (
    "FlexGEMM local-reduce broadcast values support one generated physical reduction"
)
LOCAL_REDUCE_FINALIZE_CAPTURE_ERROR = (
    "FlexGEMM captured tensors cannot be read while finalizing a compressed grouped "
    "reduction; apply them to the returned partials outside flex_gemm or to the main output"
)
LOCAL_REDUCE_UNPLANNED_ERROR = (
    "FlexGEMM grouped reductions must be returned as an auxiliary output or feed the "
    "main output through a recognized broadcast (x - r, x * r, x / r, ...); other "
    "consumers of the reduced value are not supported"
)
LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR = (
    "FlexGEMM physical local-reduce feed-main source expressions require "
    "two-phase local-reduce source lowering"
)
LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR = (
    "FlexGEMM does not support explicit reduction dtype yet"
)
LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR = (
    "FlexGEMM local reductions currently support only reductions over the "
    "innermost grouped dimension"
)
LOCAL_REDUCE_GROUPED_RESHAPE_ERROR = (
    "FlexGEMM local-reduce grouped reshape must split exactly one GEMM output dimension"
)
LOCAL_REDUCE_MIXED_MATCH_ERROR = (
    "FlexGEMM local reductions do not support mixing different grouped layouts"
)
LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR = (
    "FlexGEMM local-reduce broadcast values must share one grouped layout"
)
FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR = "FlexGEMM output plans require tensor output nodes"
FLEX_GEMM_OUTPUT_TENSOR_ERROR = "FlexGEMM expects tensor outputs"
FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR = (
    "FlexGEMM grouped main outputs do not yet compose with auxiliary outputs or "
    "reductions"
)
FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: contracted output shape must equal the "
    "physical GEMM output shape with N divided by the contraction group"
)
FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: main output shape must equal the physical "
    "GEMM output shape"
)
LOCAL_REDUCE_MATCH_NODE_ERROR = "local-reduce matches require tensor nodes"
LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR = "local-reduce output plans require tensor nodes"
LOCAL_REDUCE_RUNTIME_OUT_ERROR = "compressed local reductions require local_reduce_out"


def statically_known_multiple(value: _IntLike | IntLikeType, divisor: _IntLike) -> bool:
    """Return whether a symbolic shape value is known divisible without guards.

    ``value`` spans both worlds: inductor sizes reach it as ``int``/``sympy.Expr``,
    while the local-reduce validators below pass ``torch.Size``-derived dims whose
    dynamic entries are ``SymInt``.
    """
    return statically_known(value % divisor == 0)


def aux_output_shape_error(
    aux_size: Sequence[_IntLike], output_size: Sequence[_IntLike]
) -> NotImplementedError:
    """Reject auxiliary outputs that are neither same-shape nor a known contract."""
    return NotImplementedError(
        f"unsupported FlexGEMM aux output shape {list(aux_size)}: auxiliary outputs "
        f"must match the GEMM output shape {list(output_size)}, be the partials of "
        "one grouped reduction, or be one indexed row gather"
    )


def ungrouped_reduction_error(op_name: str) -> NotImplementedError:
    """Explain that FlexGEMM reduces the GEMM output only through one grouped view."""
    return NotImplementedError(
        f"unsupported FlexGEMM epilogue reduction {op_name}: it reduces a GEMM "
        "output dimension that no grouped view splits. Reduce through exactly one "
        "acc.view(m, -1, group) (N axis) or acc.view(-1, group, n) (M axis), "
        "return the grouped partials as an auxiliary output, and finalize them "
        "outside flex_gemm"
    )


def unsupported_reduction_op_error(op_name: str) -> NotImplementedError:
    """Name a reduction op that no FlexGEMM local-reduction path implements."""
    return NotImplementedError(
        f"unsupported FlexGEMM reduction op: {op_name}; grouped local reductions "
        "support sum, mean, prod, amax, amin, and softmax/logsumexp state"
    )


def validate_local_reduce_group_axis(group: int, axis: int) -> None:
    """Keep local-reduce specs inside the GEMM tile's M/N grouping model."""
    if group <= 0:
        raise RuntimeError(LOCAL_REDUCE_GROUP_POSITIVE_ERROR)
    if axis not in (0, 1):
        raise RuntimeError(LOCAL_REDUCE_AXIS_ERROR)


def validate_local_reduce_selected_dim_divisible(
    shape: Sequence[IntLikeType], group: int, axis: int
) -> None:
    """Reject selected M/N dimensions known not to have an integral compressed shape."""
    validate_local_reduce_group_axis(group, axis)
    selected_dim = shape[axis - 2]
    if statically_known_multiple(selected_dim, group):
        return
    if statically_known(selected_dim % group != 0):
        raise RuntimeError(LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR)


def validate_local_reduce_tensorssa_group_size(axis: int, group: int) -> None:
    """Mirror the TensorSSA fragment tiling constraints used by QuACK.

    Groups within one fragment must divide the 32-lane TensorSSA width. Larger
    groups are handled as 32-lane TensorSSA partials plus physical combine, so
    they must be exact multiples of that fragment width.
    """
    if group <= 1:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR)
    validate_local_reduce_group_axis(group, axis)
    if group > LOCAL_REDUCE_FRAGMENT_WIDTH and group % LOCAL_REDUCE_FRAGMENT_WIDTH != 0:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR)
    if (
        group <= LOCAL_REDUCE_FRAGMENT_WIDTH
        and LOCAL_REDUCE_FRAGMENT_WIDTH % group != 0
    ):
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR)


def local_reduce_compressed_shape(
    shape: Sequence[IntLikeType], group: int, axis: int
) -> tuple[IntLikeType, ...]:
    """Compute the explicit aux shape that mirrors QuACK's grouped store."""
    validate_local_reduce_selected_dim_divisible(shape, group, axis)
    result = list(shape)
    result[axis - 2] //= group
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputContraction:
    """Describe the contraction in NOTE [Non-shape-preserving FlexGEMM outputs].

    Attributes:
        group: Number of physical N values contracted into each logical output.
        chunked: Whether group values are contiguous N chunks rather than interleaved.
    """

    group: int
    chunked: bool = False

    def __post_init__(self) -> None:
        if self.group <= 1:
            raise ValueError("grouped main-output group must be greater than one")

    @property
    def concat_layout(self) -> tuple[str, ...]:
        """Return QuACK inputs whose contiguous chunks must be interleaved."""
        return ("B",) if self.chunked else ()


FlexGemmLocalReduceGeometry = GemmReductionGeometry
