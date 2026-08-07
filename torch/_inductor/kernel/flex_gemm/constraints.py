# mypy: allow-untyped-defs
"""Shared FlexGEMM local-reduce geometry, constants, and validation helpers."""

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

from torch._inductor.kernel.gemm_epilogue import GemmReductionGeometry
from torch._inductor.kernel.gemm_epilogue_utils import (
    statically_known,
    statically_known_shape_equal,
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
FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR = (
    "FlexGEMM output contractions do not compose with aux outputs, local "
    "reductions, C, alpha/beta, or batched GEMMs yet"
)
FLEX_GEMM_OUTPUT_CONTRACTION_CAPTURE_ERROR = (
    "FlexGEMM output contractions currently support only numeric [1, 1] and "
    "[M, 1] captured tensors"
)
FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR = (
    "FlexGEMM concat-layout output contractions do not compose with grouped "
    "reductions because concat layout permutes accumulator columns"
)
FLEX_GEMM_CHUNKED_CONTIGUOUS_B_ERROR = (
    "FlexGEMM concat-layout output contractions require B's output dimension to "
    "be non-contiguous, as in linear weight.t()"
)
FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: contracted output shape must equal the "
    "physical GEMM output shape with N divided by the contraction group"
)
FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR = (
    "unsupported FlexGEMM epilogue: main output shape must equal the physical "
    "GEMM output shape"
)


def statically_known_multiple(value: Any, divisor: int) -> bool:
    """Return whether a symbolic shape value is known divisible without guards."""
    return statically_known(value % divisor == 0)


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
    """Return whether a QuACK config has a validated grouped-reduction layout.

    This matches ``GemmConfig`` fields against layout families covered by forced
    kernel tests; tile divisibility alone is not sufficient. Axis-1 groups within
    one 32-value epilogue fragment need no cross-fragment combine. Some SM100
    two-CTA layouts expose only 16 contiguous N values, reducing that local limit.

    Axis-0 groups and larger axis-1 groups use ``GroupedLocalReduce``'s physical
    callback path, which combines epilogue fragments inside one CTA and directly
    stores one value per ``(row, group)``. For two-CTA ``tile_m=128`` kernels, each
    CTA has a 64-row epilogue tile whose warps are split 2x2 across M and N. An
    axis-1 group spanning the full N tile therefore crosses N-warp ownership that
    the temporal fragment combine does not stitch; strict subgroups remain valid.

    Two-CTA ``tile_m=256`` kernels instead have a 128-row epilogue tile with a 4x1
    warp layout, so one N-warp partition owns each row's full N tile. Their axis-1
    temporal fragment combine supports a full-tile group without cross-CTA state.
    Axis-0 full groups still exceed the per-CTA M tile and remain unsupported.
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
    if config.tile_n % LOCAL_REDUCE_FRAGMENT_WIDTH != 0 or tile % group != 0:
        return False

    fragment_width = LOCAL_REDUCE_FRAGMENT_WIDTH
    has_half_n_fragment = (
        axis == 1
        and config.tile_m == 128
        and config.tile_n in (128, 160, 224)
        and config.cluster_m > 1
    )
    if has_half_n_fragment:
        fragment_width //= 2
    if group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
        return fragment_width % group == 0 and group < tile

    if group % LOCAL_REDUCE_FRAGMENT_WIDTH != 0 or config.cluster_n != 1:
        return False

    is_single_cta_layout = config.tile_m == 128 and config.cluster_m == 1
    is_wide_m_two_cta_layout = config.tile_m == 256 and config.cluster_m == 2
    is_split_n_warp_two_cta_layout = (
        axis == 1
        and config.tile_m == 128
        and config.tile_n == 256
        and config.cluster_m == 2
    )
    if is_single_cta_layout:
        return True
    if is_wide_m_two_cta_layout:
        return axis == 1 or group < tile
    return is_split_n_warp_two_cta_layout and group < tile


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


# NOTE [Non-shape-preserving FlexGEMM outputs]
# FlexGEMM normally returns one value per physical GEMM accumulator. Output
# contraction is the current exception: it exposes grouped physical N values to
# the ordinary FX epilogue, requires the main expression to consume the complete
# group, and stores one logical value per group. Interleaved groups use
# ``view(M, logical_N, group)``; chunked groups use
# ``view(M, group, logical_N)`` or ``split(logical_N, dim=-1)``. This covers
# SwiGLU-like pointwise combinations without claiming to support arbitrary slices,
# permutations, expansions, or M-axis contraction.
#
# Numeric ``[1, 1]`` and ``[M, 1]`` captures are N-invariant. QuACK loads their
# one value per epilogue thread and lets generated pointwise code broadcast it in
# either the physical or contracted layout. N-varying ``[1, N]`` and ``[M, N]``
# captures need the same concat-to-interleave mapping as B for chunked outputs.
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
        if self.group <= 0:
            raise ValueError("output-contraction group must be positive")

    @property
    def concat_layout(self) -> tuple[str, ...]:
        """Return the QuACK ABI tag that interleaves chunked B columns."""
        return ("B",) if self.chunked else ()

    def validate_quack(self, device_capacity: int) -> None:
        if device_capacity == 12:
            raise NotImplementedError(
                "FlexGEMM output contractions are not yet supported on SM120"
            )
        if device_capacity not in (10, 11):
            raise NotImplementedError(
                "FlexGEMM output contractions are currently validated only on "
                "SM100 and SM110"
            )
        if self.group == 2 or (
            self.group == 4 and not self.chunked and device_capacity == 10
        ):
            return
        raise NotImplementedError(
            "FlexGEMM output-contraction stores support group 2 on SM100 and "
            "SM110, plus interleaved group 4 on SM100"
        )


def output_contraction_capture_supported(kind: str, is_boolean: bool) -> bool:
    """Return whether output-contraction codegen can broadcast a captured tensor."""
    return kind in ("scalar", "col") and not is_boolean


def output_contraction_config_supported(config: Any, n: Any) -> bool:
    """Return whether a config has validated output-contraction store ownership.

    Keep the physical M/N orientation, one CTA per cluster along N, and require
    the physical N tile not to exceed the problem. Admit only M-cluster families
    whose row ownership has been validated: a single M CTA or the wide-M two-CTA
    layout. Multiple N tiles and partial final tiles are supported by the ordinary
    tile scheduler and store predicates.
    """
    supported_m_cluster = config.cluster_m == 1 or (
        config.tile_m == 256 and config.cluster_m == 2
    )
    return (
        not config.swap_ab
        and supported_m_cluster
        and config.cluster_n == 1
        and statically_known(config.tile_n <= n)
    )


FlexGemmLocalReduceGeometry = GemmReductionGeometry


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceCallbacks:
    """Carry generated physical combine/finalize functions."""

    combine_fn: Any
    finalize_fn: Any

    def __post_init__(self) -> None:
        """Keep physical reducers from existing without their generated code."""
        if self.combine_fn is None or self.finalize_fn is None:
            raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)
