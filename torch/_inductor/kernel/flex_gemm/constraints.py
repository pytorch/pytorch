# mypy: allow-untyped-defs
"""Shared FlexGEMM local-reduce geometry, constants, and validation helpers."""

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

import sympy

from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    statically_known_true as fx_statically_known_true,
)


LOCAL_REDUCE_FEED_MAIN_ARG_NAME: Final = "local_reduce0"
LOCAL_REDUCE_COMBINE_FN_SUFFIX: Final = "_local_reduce_combine_fn"
LOCAL_REDUCE_FINALIZE_FN_SUFFIX: Final = "_local_reduce_finalize_fn"
LOCAL_REDUCE_COMBINE_KEY_SUFFIX: Final = ":local_reduce_combine"
LOCAL_REDUCE_FINALIZE_KEY_SUFFIX: Final = ":local_reduce_finalize"


# The physical feed-main path currently reduces only within one lane-layout M
# group; cross-warp M stitching needs the two-phase/replay path used by
# compressed aux reductions. Axis-1 feeds whose groups fit in one TensorSSA
# fragment lower as plain generated TensorSSA without a feed plan.
LOCAL_REDUCE_FRAGMENT_WIDTH = 32
LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only axis 0"
)
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
LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR = (
    "unsupported FlexGEMM epilogue partial-output contract: FlexGEMM does not "
    "support this local-reduce output contract yet. Please file an issue with "
    "the FlexGEMM epilogue expression."
)
LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR = (
    "FlexGEMM local reductions do not support mixing grouped TensorSSA "
    "values with different grouped layouts"
)
LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR = (
    "FlexGEMM local reductions currently support only aten.mm"
)
LOCAL_REDUCE_C_ALPHA_BETA_ERROR = (
    "FlexGEMM local reductions cannot be combined with C/alpha/beta yet"
)
LOCAL_REDUCE_SWAP_AB_ERROR = (
    "FlexGEMM local reductions do not support swap_ab configs yet"
)
LOCAL_REDUCE_AUX_TENSORSSA_ERROR = (
    "FlexGEMM local-reduce aux output must be produced by a grouped TensorSSA reduction"
)
LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR = (
    "FlexGEMM does not support this aux output shape yet. Please file an issue "
    "with the FlexGEMM epilogue expression."
)
LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR = (
    "FlexGEMM local-reduce broadcast values support one generated physical reduction"
)
LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR = (
    "FlexGEMM physical finalize expressions support a single physical local reduction"
)
LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR = (
    "FlexGEMM physical local reductions do not support post-reduction "
    "pointwise transforms yet. Please file an issue with the FlexGEMM epilogue "
    "expression."
)
LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR = (
    "FlexGEMM physical local reductions require finalize expressions to depend "
    "only on the reduced value and scalar constants"
)
LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR = (
    "FlexGEMM physical local-reduce feed-main source expressions require "
    "two-phase local-reduce source lowering"
)
LOCAL_REDUCE_CONFIG_ERROR = (
    "FlexGEMM local-reduce aux outputs require a non-swap_ab config whose CTA "
    "tile axis is divisible by group"
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
LOCAL_REDUCE_MATCH_NODE_ERROR = "local-reduce matches require tensor nodes"
LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR = "local-reduce output plans require tensor nodes"
LOCAL_REDUCE_RUNTIME_OUT_ERROR = "compressed local reductions require local_reduce_out"
LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR = (
    "FlexGEMM local reductions currently support only 2-D aten.mm"
)
LOCAL_REDUCE_OUT_SHAPE_ERROR = "local_reduce_out shape must be {expected}, got {actual}"
LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR = (
    "physical local reductions require generated local-reduce callbacks"
)


def statically_known(expr: Any) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_multiple(value: Any, divisor: int) -> bool:
    """Return whether a symbolic shape value is known divisible without guards."""
    return statically_known(value % divisor == 0)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )


def is_flex_gemm_partial_reduction_shape(
    aux_size: Sequence[Any], output_size: Sequence[Any]
) -> bool:
    """Recognize aux shapes that imply a final PyTorch reduction, not local reduce.

    FlexGEMM's generic aux-output path supports one same-shape aux tensor beside
    the main output. Reduced shapes such as ``[]``, ``[M]``, ``[N]``, ``[M, 1]``,
    ``[1, N]``, or exact 2-D divisors of ``[M, N]`` mean the epilogue tried to
    return a final PyTorch reduction/block reduction. Those are different from
    QuACK local-reduce aux outputs, which are only accepted after the epilogue
    exposes an explicit grouped view such as ``acc.view(M, -1, group).sum(-1)``.
    """
    if len(output_size) != 2:
        return False
    aux_shape = tuple(aux_size)
    m, n = output_size
    if any(
        statically_known_shape_equal(aux_shape, candidate)
        for candidate in ((), (m,), (n,), (1, 1), (m, 1), (1, n))
    ):
        return True
    if len(aux_shape) != 2:
        return False
    aux_m, aux_n = aux_shape
    return (
        statically_known(aux_m > 0)
        and statically_known(aux_n > 0)
        and statically_known(aux_m <= m)
        and statically_known(aux_n <= n)
        and (statically_known(aux_m < m) or statically_known(aux_n < n))
        and statically_known_multiple(m, aux_m)
        and statically_known_multiple(n, aux_n)
    )


def local_reduce_unsupported_tensorssa_error(
    reduction: Any, *, value_only: bool = False
) -> NotImplementedError:
    """Explain why a grouped reduction is outside the current TensorSSA subset."""
    suffix = " value-only reduction" if value_only else ""
    return NotImplementedError(
        "FlexGEMM does not support this grouped local reduction yet: "
        f"{reduction} does not map to a CuTe TensorSSA{suffix}. Please file "
        "an issue with the FlexGEMM epilogue expression."
    )


def validate_local_reduce_group_axis(group: int, axis: int) -> None:
    """Keep local-reduce specs inside the GEMM tile's M/N grouping model."""
    if group <= 0:
        raise RuntimeError(LOCAL_REDUCE_GROUP_POSITIVE_ERROR)
    if axis not in (0, 1):
        raise RuntimeError(LOCAL_REDUCE_AXIS_ERROR)


def validate_local_reduce_selected_dim_divisible(
    shape: Sequence[Any], group: int, axis: int
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


def local_reduce_needs_physical_callbacks(axis: int, group: int) -> bool:
    """Return whether QuACK must merge TensorSSA partials outside the fragment path."""
    return axis == 0 or group > LOCAL_REDUCE_FRAGMENT_WIDTH


def validate_local_reduce_runtime_dense_mm(ndim: int) -> None:
    """Keep runtime wrappers on the only layout QuACK currently supports.

    Local-reduce group/axis semantics are defined relative to dense ``mm`` output
    dimensions. Batched or vectorized matmul layouts would need separate shape
    compression and epilogue argument mapping rules before the same ABI is valid.
    """
    if ndim != 2:
        raise NotImplementedError(LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR)


def validate_local_reduce_out_shape(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> None:
    """Ensure caller-provided aux storage matches the structural reduce plan.

    Runtime cannot reinterpret an arbitrary aux tensor as the compressed local-
    reduce domain: QuACK writes exactly the shape produced by dividing the chosen
    GEMM output dimension by the group size, so mismatches would corrupt memory
    or silently expose the wrong logical tensor.
    """
    actual = tuple(actual_shape)
    expected = tuple(expected_shape)
    if not statically_known_shape_equal(actual, expected):
        raise RuntimeError(
            LOCAL_REDUCE_OUT_SHAPE_ERROR.format(expected=expected, actual=actual)
        )


def validate_local_reduce_feed_main_capability(axis: int, group: int) -> None:
    """Limit feed-main reducers to the physical path QuACK can re-inject today.

    Feeding a reduction back into the main epilogue needs the physical row-lane
    combine result to be available as a scalar value for each output element.
    That is currently implemented only for same-warp M-axis groups.
    """
    if axis != 0:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR)
    if group > LOCAL_REDUCE_FRAGMENT_WIDTH:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR)


def local_reduce_compressed_shape(
    shape: Sequence[Any], group: int, axis: int
) -> tuple[Any, ...]:
    """Compute the explicit aux shape that mirrors QuACK's grouped store."""
    validate_local_reduce_selected_dim_divisible(shape, group, axis)
    result = list(shape)
    result[axis - 2] //= group
    return tuple(result)


def validate_local_reduce_no_c_alpha_beta(
    effective_C: Any | None, alpha: float, beta: float
) -> None:
    """Reject C/alpha/beta composition until local-reduce ordering is explicit."""
    if effective_C is not None or alpha != 1.0 or beta != 1.0:
        raise NotImplementedError(LOCAL_REDUCE_C_ALPHA_BETA_ERROR)


def validate_flex_gemm_local_reduce_config(config: Any, group: int, axis: int) -> bool:
    """Return whether a QuACK config can keep grouped reductions inside one CTA.

    This host gate covers tile and cluster fields available on ``GemmConfig``.
    SM100 128x128 two-CTA configs expose only 16 contiguous N values per
    epilogue fragment; other accepted configs expose the full 32-wide fragment.
    Lane/warp ownership is derived later from QuACK's tiled-copy layout, where
    ``GroupedLocalReduce`` asserts the remaining lane-count, divisibility, and
    stride invariants. Forced-config tests cover the accepted SM100 extremes.
    """
    match axis:
        case 0:
            tile = config.tile_m
        case 1:
            tile = config.tile_n
        case _:
            return False
    if group <= 0 or config.swap_ab:
        return False
    if config.tile_n < 128 or config.tile_n % 64 != 0:
        return False
    if tile % group != 0:
        return False
    fragment_width = LOCAL_REDUCE_FRAGMENT_WIDTH
    if (
        axis == 1
        and config.tile_m == 128
        and config.tile_n == 128
        and config.cluster_m > 1
    ):
        fragment_width //= 2
    match group:
        case _ if group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
            return fragment_width % group == 0 and group < tile
        case _:
            return (
                group % LOCAL_REDUCE_FRAGMENT_WIDTH == 0
                and group <= tile
                and config.tile_m == 128
                and config.cluster_m == 1
                and config.cluster_n == 1
            )


def flex_gemm_local_reduce_candidate_groups(config: Any, axis: int) -> tuple[int, ...]:
    """Enumerate group sizes worth checking against the config capability gate."""
    match axis:
        case 0:
            tile = config.tile_m
        case 1:
            tile = config.tile_n
        case _:
            return ()
    return (2, 4, 8, 16, 32, *range(64, tile + 1, 32))


def max_flex_gemm_local_reduce_group_for_configs(
    configs: Sequence[Any], axis: int
) -> int | None:
    """Return the largest group accepted by the current local-reduce config gate."""
    candidates = [
        group
        for config in configs
        for group in flex_gemm_local_reduce_candidate_groups(config, axis)
        if validate_flex_gemm_local_reduce_config(config, group, axis)
    ]
    return max(candidates) if candidates else None


def flex_gemm_local_reduce_config_error(
    configs: Sequence[Any], group: int, axis: int
) -> str:
    """Explain the current config-filter frontier for local-reduce groups."""
    max_group = max_flex_gemm_local_reduce_group_for_configs(configs, axis)
    if max_group is None:
        return LOCAL_REDUCE_CONFIG_ERROR
    return (
        f"{LOCAL_REDUCE_CONFIG_ERROR}; requested group={group}, "
        f"max supported group={max_group} for axis={axis}"
    )


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceGeometry:
    """Describe the grouped output axis shared by local-reduce consumers.

    Attributes:
        group: Number of contiguous M or N elements in each local group.
        axis: GEMM output axis being grouped: 0 for M, 1 for N.
    """

    group: int
    axis: int

    def __post_init__(self) -> None:
        """Reject geometry outside the GEMM tile's M/N grouping model."""
        validate_local_reduce_group_axis(self.group, self.axis)

    @property
    def needs_physical_callbacks(self) -> bool:
        return local_reduce_needs_physical_callbacks(self.axis, self.group)


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceCallbacks:
    """Carry generated physical combine/finalize functions."""

    combine_fn: Any
    finalize_fn: Any

    def __post_init__(self) -> None:
        """Keep physical reducers from existing without their generated code."""
        if self.combine_fn is None or self.finalize_fn is None:
            raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)
