# mypy: allow-untyped-defs
"""Lower FlexGEMM HOP bodies and connect epilogue analysis to backend templates.

``flex_gemm_lowering`` is the main entry point. Non-QUACK dense requests execute
through ordinary Inductor lowering. Scaled-mm requires the QUACK fused path.
See ``lower_quack_flex_gemm`` for the template flow.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import logging
from typing import Any, cast, Literal, TYPE_CHECKING

import sympy

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.flex_gemm import (
    _SUPPORTED_FLEX_GEMM_OP_NAMES,
    flex_gemm_body_gemm_op,
    flex_gemm_hop,
    FLEX_GEMM_OP_SPECS,
    validate_scaled_mm,
)
from torch._inductor import config
from torch._logging import warning_once
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from ... import ir
from ...heuristics.template.flex_gemm import (
    flex_gemm_default_config,
    flex_gemm_search_space,
    QuackConfigKey,
)
from ...ir import IRNode, TensorBox
from ...lowering import (
    constant_pad_nd,
    empty_strided,
    expand,
    process_subgraph_nodes,
    register_lowering,
    view,
)
from ...utils import _IntLike, ceildiv, is_bf16x9_matmul
from ..gemm_epilogue_utils import statically_known_equal, statically_known_shape_equal
from .constraints import (
    aux_output_shape_error,
    FLEX_GEMM_CAPTURE_SHAPE_ERROR,
    FlexGemmOutputContraction,
    FlexGemmOutputExpansion,
    LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR,
)
from .debug import (
    format_flex_gemm_analysis,
    format_flex_gemm_analysis_details,
    format_flex_gemm_config_candidates,
    format_flex_gemm_lowering_plan,
    format_flex_gemm_problem,
    log_flex_gemm_artifact,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..gemm_epilogue_analysis import GemmLocalReduceAnalysis
    from .template import FlexGemmEpilogueConfig


log = logging.getLogger(__name__)


def decompose_nvgemm_additive_gemm(graph_module: torch.fx.GraphModule) -> None:
    graph = graph_module.graph
    changed = False
    for node in list(graph.nodes):
        if node.target not in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        ):
            continue
        bias, mat1, mat2 = node.args[:3]
        alpha = node.kwargs.get("alpha", 1.0)
        beta = node.kwargs.get("beta", 1.0)
        gemm_target = (
            torch.ops.aten.mm.default
            if node.target is torch.ops.aten.addmm.default
            else torch.ops.aten.bmm.default
        )
        with graph.inserting_before(node):
            result = graph.call_function(gemm_target, (mat1, mat2))
            if alpha != 1:
                result = graph.call_function(torch.ops.aten.mul.Tensor, (result, alpha))
            if beta != 0:
                if beta != 1:
                    bias = graph.call_function(torch.ops.aten.mul.Tensor, (bias, beta))
                result = graph.call_function(torch.ops.aten.add.Tensor, (result, bias))
        result.meta = node.meta
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()


def check_quack_fp32_operand(gemm_arg: TensorBox) -> None:
    """Reject CUDA float32 GEMMs unless the matmul policy permits TF32."""
    precision = torch.backends.cuda.matmul.fp32_precision
    if (
        gemm_arg.get_dtype() is not torch.float32
        or gemm_arg.get_device_or_error().type != "cuda"
        or precision == "tf32"
    ):
        return
    raise NotImplementedError(
        "FlexGEMM QUACK computes float32 GEMM operands in TF32, but "
        f"torch.backends.cuda.matmul.fp32_precision is {precision!r}; "
        "opt in with torch.set_float32_matmul_precision('high') or use bfloat16/float16 operands"
    )


def check_quack_sm90_operands(gemm_arg: TensorBox, blockscaled: bool) -> None:
    """Reject operands QuACK's SM90 (WGMMA) mainloop cannot compute."""
    device = gemm_arg.get_device_or_error()
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        return
    if blockscaled:
        raise NotImplementedError(
            "FlexGEMM QUACK block-scaled GEMMs require SM100 or SM120; "
            "SM90 has no block-scaled MMA"
        )
    if gemm_arg.get_dtype() is torch.float32:
        raise NotImplementedError(
            "FlexGEMM QUACK float32 (TF32) GEMMs are not supported on SM90; "
            "use bfloat16/float16 operands"
        )


def has_flex_gemm_quack() -> bool:
    """Whether the vendored QuACK backend can import its CuTeDSL dependency."""
    return importlib.util.find_spec("cutlass") is not None


# QuACK epilogue captures, aux outputs and local reductions are validated for
# these 2-D, bias-free GEMMs only; grouped_mm additionally rejects the
# features QuACK's varlen-M path lacks.
QUACK_EPILOGUE_FEATURE_OPS = frozenset(
    (
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_mm_v2.default,
        torch.ops.aten._grouped_mm.default,
    )
)


@dataclasses.dataclass(frozen=True)
class QuackBlockScaledContract:
    """One traced aten._scaled_mm_v2 call resolved to QuACK's block-scaled main loop.

    ``gemm_inputs`` is (A, B, SFA, SFB) in template-input order;
    ``tensorwise_scales`` are the optional NVFP4 global scales folded into the
    epilogue as scalar operands.
    """

    format: str
    gemm_inputs: tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]
    tensorwise_scales: tuple[torch.fx.Node, ...]


def quack_blockscaled_contract(gemm_fx_node: torch.fx.Node) -> QuackBlockScaledContract:
    """Resolve the traced native call to the fused block-scaled input contract."""
    normalized = normalize_function(
        torch.ops.aten._scaled_mm_v2.default,
        gemm_fx_node.args,
        gemm_fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("aten._scaled_mm_v2 arguments must bind to its schema")
    call = normalized.kwargs
    if call["bias"] is not None or call["contraction_dim"] or call["use_fast_accum"]:
        raise NotImplementedError(
            "FlexGEMM scaled-mm does not support bias, custom contraction, or fast accumulation"
        )
    scale_a, scale_b = tuple(call["scale_a"]), tuple(call["scale_b"])
    format_name = validate_scaled_mm(
        call["input"].meta["val"],
        call["mat2"].meta["val"],
        [node.meta["val"] for node in scale_a],
        [node.meta["val"] for node in scale_b],
        call["recipe_a"],
        call["recipe_b"],
        call["swizzle_a"],
        call["swizzle_b"],
    )
    return QuackBlockScaledContract(
        format_name,
        (call["input"], call["mat2"], scale_a[0], scale_b[0]),
        (*scale_a[1:], *scale_b[1:]),
    )


def quack_grouped_mm_contract(
    gemm_fx_node: torch.fx.Node,
) -> tuple[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node], Literal["m", "k"]]:
    """Resolve grouped_mm operands and the axis segmented by ``offs[E]`` end offsets.

    Varlen-M uses k-major A ``[total_m, K]`` and B ``[E, K, N]``. Varlen-K
    uses m-major A ``[M, total_k]`` and n-major B ``[total_k, N]`` for wgrad.
    """
    normalized = normalize_function(
        torch.ops.aten._grouped_mm.default,
        gemm_fx_node.args,
        gemm_fx_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("aten._grouped_mm arguments must bind to its schema")
    call = normalized.kwargs
    mat_a, mat_b, offs = call["input"], call["mat2"], call["offs"]
    if offs is None or call["bias"] is not None or call["out_dtype"] is not None:
        raise NotImplementedError(
            "FlexGEMM QUACK grouped_mm requires offs and no bias or out_dtype"
        )
    # aten._grouped_mm's meta already requires contiguous 1-D int32 offs.
    a_meta, b_meta = (node.meta["val"] for node in (mat_a, mat_b))
    if (
        a_meta.ndim != 2
        or b_meta.ndim not in (2, 3)
        or a_meta.dtype not in (torch.bfloat16, torch.float16)
        or b_meta.dtype is not a_meta.dtype
    ):
        raise NotImplementedError(
            "FlexGEMM QUACK grouped_mm supports only bf16/fp16 2-D A and 2-D or 3-D B"
        )
    if b_meta.ndim == 3:
        if a_meta.stride(-1) != 1:
            raise NotImplementedError("FlexGEMM varlen-M requires k-major A")
        return (mat_a, mat_b, offs), "m"
    if a_meta.stride(-2) != 1 or b_meta.stride(-1) != 1:
        raise NotImplementedError(
            "FlexGEMM varlen-K requires m-major A [M, total_k] and n-major B [total_k, N]"
        )
    if a_meta.shape[-1] != b_meta.shape[-2]:
        raise NotImplementedError("FlexGEMM varlen-K requires equal reduction extents")
    return (mat_a, mat_b, offs), "k"


def flex_gemm_cu_seqlens_benchmark_input(
    node: IRNode, total_length: _IntLike
) -> torch.Tensor:
    """Build evenly spaced M/K boundaries for autotune benchmarks."""
    from torch._inductor.virtualized import V

    sizevars = V.graph.sizevars
    return torch.linspace(
        0,
        sizevars.optimization_hint(total_length),
        sizevars.optimization_hint(node.get_size()[0]),
        dtype=torch.int32,
        device=node.get_device_or_error(),
    )


def flex_gemm_tensor_placeholders(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    """Return placeholders QuACK can bind as tensor epilogue arguments.

    FlexGEMM identifies the GEMM A/B inputs from the mm node, then treats the
    remaining tensor-valued placeholders as closed-over epilogue tensors. Scalar
    SymInt placeholders are shape values, not tensor arguments; the current QuACK
    FlexGEMM entrypoint has no scalar epilogue-argument slots for them.
    """
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]


def flex_gemm_epilogue_arg_placeholders(
    graph_module: torch.fx.GraphModule, gemm_fx_node: torch.fx.Node
) -> tuple[torch.fx.Node, ...]:
    """Find tensor inputs captured by epilogue loads, excluding GEMM operands."""
    gemm_placeholders = OrderedSet(
        arg
        for arg in pytree.tree_leaves((gemm_fx_node.args, gemm_fx_node.kwargs))
        if isinstance(arg, torch.fx.Node)
    )
    return tuple(
        node
        for node in flex_gemm_tensor_placeholders(graph_module)
        if node not in gemm_placeholders
    )


def infer_flex_gemm_epilogue_arg_kinds(
    gemm_op: torch._ops.OpOverload,
    epilogue_args: Sequence[IRNode],
    output_size: list[Any],
) -> tuple[str, ...]:
    """Classify realized captured epilogue tensors for static wrapper kwargs.

    bmm captures may carry a leading batch dim: ``[B, 1, N]`` rows and
    ``[B, M, 1]`` columns are read per batch, ``[1, 1, N]`` / ``[1, M, 1]`` /
    ``[1, 1, 1]`` are shared by every batch, and tiles must be ``[B, M, N]``.
    """
    if not epilogue_args:
        return ()
    batched = gemm_op is torch.ops.aten.bmm.default
    if gemm_op not in QUACK_EPILOGUE_FEATURE_OPS and not batched:
        raise NotImplementedError(
            "FlexGEMM generated epilogues with captured tensor reads currently "
            "support only aten.mm, aten.bmm, aten._scaled_mm_v2 and aten._grouped_mm"
        )
    m, n = output_size[-2], output_size[-1]
    shared_leads = ([], [1]) if batched else ([],)
    vector_leads = (*shared_leads, output_size[:-2]) if batched else shared_leads

    def matches(
        size: Sequence[Any], leads: Sequence[Sequence[Any]], inner: Sequence[Any]
    ) -> bool:
        return any(
            statically_known_shape_equal(size, [*lead, *inner]) for lead in leads
        )

    epilogue_arg_kinds = []
    for epilogue_arg in epilogue_args:
        epilogue_arg_size = epilogue_arg.get_size()
        if matches(epilogue_arg_size, shared_leads, [1, 1]):
            epilogue_arg_kinds.append("scalar")
        elif statically_known_shape_equal(epilogue_arg_size, output_size):
            epilogue_arg_kinds.append("tile")
        elif not batched and statically_known_shape_equal(
            epilogue_arg_size, [*output_size[:-1], 2 * n]
        ):
            # Two lanes interleaved along N, read as ``h.view(..., N, 2)[..., j]``.
            epilogue_arg_kinds.append("interleaved")
        elif matches(epilogue_arg_size, vector_leads, [1, n]):
            epilogue_arg_kinds.append("row")
        elif matches(epilogue_arg_size, vector_leads, [m, 1]):
            epilogue_arg_kinds.append("col")
        else:
            raise NotImplementedError(
                f"{FLEX_GEMM_CAPTURE_SHAPE_ERROR}; got {list(epilogue_arg_size)} "
                f"against GEMM output {list(output_size)}"
            )
    return tuple(epilogue_arg_kinds)


def widen_row_capture(arg: TensorBox, group: int) -> TensorBox:
    """Repeat a ``[1, n]`` row capture ``group`` times per column: ``[1, n * group]``."""
    n = arg.get_size()[-1]
    return view(expand(view(arg, [n, 1]), [n, group]), [1, n * group])


def require_lane_domain_row_captures(
    analysis: GemmLocalReduceAnalysis,
    gemm: torch.fx.Node,
    rows: Sequence[torch.fx.Node],
) -> None:
    """Reject row captures an output contraction would read at the wrong column.

    ``widen_row_capture`` maps physical column ``c`` to ``row[c // group]``, which
    matches eager only where the row meets selected lanes (logical columns).
    Capture-only pointwise math and views keeping a ``[n]`` / ``[1, n]`` / ``[m, n]``
    shape stay in logical columns; any other view or a use against physical GEMM
    columns (such as ``acc.view(m, 2, n) * row``) is rejected.
    """
    from .quack_reductions import (
        is_pointwise_node,
        squeeze_source_node,
        tensor_meta_shape,
    )

    m = gemm.meta["val"].shape[0]
    for row in rows:
        n = row.meta["val"].shape[-1]
        pending, seen = [row], OrderedSet[torch.fx.Node]()
        while pending:
            for user in pending.pop().users:
                if user in seen or user.op == "output":
                    continue
                seen.add(user)
                if analysis.graph.depends_on(user, gemm):
                    if user not in analysis.lane_domains:
                        raise NotImplementedError(
                            f"FlexGEMM output contractions read row capture {row.name} "
                            "only against selected lanes, not physical GEMM columns"
                        )
                    continue
                shape = tensor_meta_shape(user)
                if not (
                    is_pointwise_node(user)
                    or user.target in CAPTURE_RESHAPE_TARGETS
                    or squeeze_source_node(user) is not None
                ) or not any(
                    shape is not None and statically_known_shape_equal(shape, allowed)
                    for allowed in ((n,), (1, n), (m, n))
                ):
                    raise NotImplementedError(
                        f"FlexGEMM output contractions read row capture {row.name} "
                        f"only through pointwise ops and [n] / [1, n] / [m, n] views; got {user.format_node()}"
                    )
                pending.append(user)


def validate_flex_gemm_matrix_layout(name: str, layout: ir.Layout) -> None:
    """Reject matrix layouts that cannot satisfy QuACK's TMA argument ABI."""
    from torch._inductor.virtualized import V

    if any(statically_known_equal(size, 0) for size in layout.size):
        return
    stride = layout.stride
    unit = len(stride) - (1 if statically_known_equal(stride[-1], 1) else 2)
    alignment = max(16 // layout.dtype.itemsize, 1)
    aligned = sympy.And(
        sympy.Eq(stride[unit], 1),
        *(sympy.Eq(s % alignment, 0) for dim, s in enumerate(stride) if dim != unit),
    )
    if not V.graph.sizevars.expect_true(aligned):
        raise NotImplementedError(
            f"FlexGEMM QUACK {name} requires a unit-stride matrix dimension and "
            f"16-byte aligned outer strides; got {stride} for {layout.dtype}"
        )


def require_tma_loadable_tile_captures(
    epilogue_args: list[TensorBox], epilogue_arg_kinds: Sequence[str]
) -> list[TensorBox]:
    """Give tile and interleaved captures a layout QuACK's TMA load can read in place.

    A tile load needs a 16-byte aligned base, one unit-stride matrix dim, and aligned outer strides;
    an interleaved load pairs adjacent N columns, so it must be N-major. Views
    such as ``h.view(m, n, 2)[..., 0]`` lack a unit-stride matrix dim and are copied to a
    contiguous layout here instead of failing QuACK's launch-time stride check.
    The copy's row stride is padded to the alignment, so an unaligned N (a bf16
    ``[m, 12]`` tile) stays loadable.
    Row-strided views such as ``h[:, :n]`` load in place when their base and row stride
    are aligned. Unknown producer layouts are constrained to padded strides; symbolic
    layouts can specialize on stride divisibility.
    """
    from torch._inductor.virtualized import V

    sizevars = V.graph.sizevars

    def loadable(arg: TensorBox, unit_dims: tuple[int, ...], alignment: int) -> bool:
        stride = arg.maybe_get_stride()
        if stride is None:
            return False
        for unit in unit_dims:
            if statically_known_equal(stride[unit], 1) and all(
                sizevars.statically_known_multiple_of(s, alignment)
                for dim, s in enumerate(stride)
                if dim != len(stride) + unit
            ):
                return True
        return False

    unit_dims = {"tile": (-1, -2), "interleaved": (-1,)}
    out = []
    for arg, kind in zip(epilogue_args, epilogue_arg_kinds, strict=True):
        if kind in ("row", "col"):
            if ir.is_unaligned(arg):
                arg = ir.ExternKernel.copy_input(arg)
            if len(arg.get_size()) == 3:
                # QuACK's batched vector ABI requires unit vector stride and
                # a batch stride divisible by 4 elements after squeezing.
                vector_dim = -1 if kind == "row" else -2
                stride = arg.maybe_get_stride()
                if (
                    stride is None
                    or not statically_known_equal(stride[vector_dim], 1)
                    or not sizevars.statically_known_multiple_of(stride[0], 4)
                ):
                    padded = list(arg.get_size())
                    padded[vector_dim] = ceildiv(padded[vector_dim], 4) * 4
                    arg = cast(
                        TensorBox,
                        ir.ExternKernel.require_exact_strides(
                            arg, ir.FlexibleLayout.contiguous_strides(padded)
                        ),
                    )
            else:
                arg = cast(TensorBox, ir.ExternKernel.require_contiguous(arg))
        alignment = max(16 // arg.get_dtype().itemsize, 1)
        if kind in unit_dims and ir.is_unaligned(arg):
            arg = ir.ExternKernel.copy_input(arg)
        if kind in unit_dims and not loadable(arg, unit_dims[kind], alignment):
            *outer, n = arg.get_size()
            padded = [*outer, ceildiv(n, alignment) * alignment]
            arg = cast(
                TensorBox,
                ir.ExternKernel.require_exact_strides(
                    arg, ir.FlexibleLayout.contiguous_strides(padded)
                ),
            )
        out.append(arg)
    return out


CAPTURE_RESHAPE_TARGETS = frozenset(
    (
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    )
)


def capture_reshape_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return the shape a body reshape gives a captured tensor, else None."""
    meta = node.meta.get("val")
    if node.target not in CAPTURE_RESHAPE_TARGETS or not isinstance(meta, torch.Tensor):
        return None
    return tuple(meta.shape)


def flex_gemm_1d_capture_shape(
    node: torch.fx.Node, m: Any, n: Any
) -> tuple[Any, Any] | None:
    """Return the [1, N] / [M, 1] / [1, 1] reading of one 0-D or 1-D capture.

    ``w[:, None]`` reads a length-M capture as a column; direct broadcasting and
    ``w[None, :]`` read a length-N capture as a row, so an [N] capture with
    M == N follows ``acc + w`` semantics. Return ``None`` when the length fits
    neither reading, leaving the capture for the later shape check.
    """
    meta = node.meta["val"]
    if meta.ndim > 1:
        return None
    length = meta.numel()
    if statically_known_equal(length, 1):
        return (1, 1)
    column_uses = [
        (shape := capture_reshape_shape(user)) is not None
        and statically_known_shape_equal(shape, (length, 1))
        for user in node.users
    ]
    if all(column_uses):
        return (m, 1) if statically_known_equal(length, m) else None
    if any(column_uses):
        raise NotImplementedError(
            f"FlexGEMM capture {node.name} is used as both a row and a column"
        )
    return (1, n) if statically_known_equal(length, n) else None


def normalize_flex_gemm_1d_captures(
    graph_module: torch.fx.GraphModule,
    placeholders: Sequence[torch.fx.Node],
    epilogue_args: list[TensorBox],
    gemm_shape: Sequence[Any],
) -> list[TensorBox]:
    """Rewrite 0-D/1-D captures to the 2-D broadcast views QuACK binds.

    The placeholder metadata and the realized template input become the 2-D
    view, and body reshapes producing that view fold into the placeholder so
    ``w[None, :]`` inside the epilogue lowers like a hoisted ``[1, N]`` capture.
    """
    m, n = gemm_shape[-2:]
    normalized = list(epilogue_args)
    changed = False
    for index, (node, arg) in enumerate(zip(placeholders, epilogue_args, strict=True)):
        shape = flex_gemm_1d_capture_shape(node, m, n)
        if shape is None:
            continue
        for user in tuple(node.users):
            user_shape = capture_reshape_shape(user)
            if user_shape is not None and statically_known_shape_equal(
                user_shape, shape
            ):
                user.replace_all_uses_with(node)
                graph_module.graph.erase_node(user)
        node.meta["val"] = node.meta["val"].view(shape)
        normalized[index] = view(arg, ir.convert_shape_to_inductor(shape))
        changed = True
    if changed:
        graph_module.graph.lint()
        graph_module.recompile()
    return normalized


def validate_bmm_capture_views(graph_module: torch.fx.GraphModule) -> None:
    """Views lower as aliases, so they must preserve broadcast axes and dtype."""
    from ..gemm_epilogue_codegen import _cute_op_name

    for node in graph_module.graph.nodes:
        if node.op != "call_function" or _cute_op_name(node.target) not in (
            "view",
            "reshape",
            "squeeze",
            "unsqueeze",
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        source_meta, result_meta = source.meta["val"], node.meta["val"]
        shapes = [list(meta.shape) for meta in (source_meta, result_meta)]
        for shape in shapes:
            while shape and statically_known_equal(shape[0], 1):
                shape.pop(0)
        if source_meta.dtype != result_meta.dtype or not statically_known_shape_equal(
            shapes[0], shapes[1]
        ):
            raise NotImplementedError(
                "FlexGEMM bmm epilogue views must preserve batch, row, and column axes "
                "and dtype; reshape or reinterpret captures before the FlexGEMM call"
            )


def validate_flex_gemm_aux_outputs(
    gemm_op: torch._ops.OpOverload,
    aux_outputs: tuple[torch.fx.Node, ...],
    output_size: Sequence[_IntLike],
) -> tuple[Any, ...]:
    """Validate QUACK aux-output support and return fake tensor metadata."""
    if not aux_outputs:
        return ()
    if gemm_op not in QUACK_EPILOGUE_FEATURE_OPS:
        raise NotImplementedError(
            "FlexGEMM generic aux tuple epilogues currently support only "
            "aten.mm, aten._scaled_mm_v2 and aten._grouped_mm"
        )
    aux_metas = []
    for aux_output in aux_outputs:
        aux_meta = aux_output.meta.get("val")
        if aux_meta is None:
            raise NotImplementedError(
                "FlexGEMM generic aux tuple epilogues require aux output metadata"
            )
        aux_size = ir.convert_shape_to_inductor(aux_meta.shape)
        if not statically_known_shape_equal(aux_size, output_size):
            raise aux_output_shape_error(aux_size, output_size)
        aux_metas.append(aux_meta)
    return tuple(aux_metas)


def allocate_flex_gemm_aux_outs(
    aux_metas: tuple[Any, ...], mat1: TensorBox
) -> tuple[TensorBox, ...]:
    """Allocate auxiliary buffers with their requested dense strides."""
    return tuple(
        empty_strided(
            ir.convert_shape_to_inductor(aux_meta.shape),
            ir.convert_shape_to_inductor(aux_meta.stride()),
            dtype=aux_meta.dtype,
            device=mat1.get_device_or_error(),
        )
        for aux_meta in aux_metas
    )


def flex_gemm_local_reduce_metas(local_reduce) -> tuple[Any, ...]:
    """Return metadata for the optional compressed local-reduce output."""
    if local_reduce is None or local_reduce.store is None:
        return ()
    return (local_reduce.store.node.meta["val"],)


def flex_gemm_quack_configs(
    template_config: FlexGemmEpilogueConfig,
    gemm_input_nodes: list[IRNode],
    epilogue_input_nodes: list[IRNode],
) -> tuple[QuackConfigKey, ...]:
    """Ask QuACK which GemmConfigs this call may pin, default first.

    EpiOps decide epilogue legality; lowering also checks MMA operand layouts.
    The runtime's EpiMod is built from metadata and pruned against GEMM shape hints;
    ``FlexGemmEpilogueCaller.output_node`` guards the shape-dependent rules.
    """
    from torch._inductor.kernel.flex_gemm.runtime import (
        flex_gemm_preferred_config,
        flex_gemm_problem,
        selection_callback,
    )
    from torch._inductor.virtualized import V
    from torch._vendor.quack.gemm_config import cta_tile_shape_m, get_all_configs
    from torch._vendor.quack.gemm_runtime.autotune import legal_mod_configs

    output_transform = template_config.output_transform
    epimod = template_config.epimod(
        selection_callback,
        [node.get_dtype() for node in (*gemm_input_nodes, *epilogue_input_nodes)],
        lambda name: selection_callback,
    )
    sizevars = V.graph.sizevars
    mat1 = gemm_input_nodes[template_config.gemm_op.mat1_index]
    mat2 = gemm_input_nodes[template_config.gemm_op.mat2_index]
    device = mat1.get_device_or_error()
    problem = flex_gemm_problem(
        device,
        sizevars.optimization_hint(mat1.get_size()[-2]),
        sizevars.optimization_hint(mat2.get_size()[-1]),
        None if output_transform is None else output_transform.concat_layout,
        blockscaled=template_config.blockscaled_format is not None,
        varlen_m=template_config.varlen_axis == "m",
        varlen_k=template_config.varlen_axis == "k",
    )
    if isinstance(output_transform, FlexGemmOutputContraction) and (
        torch.cuda.get_device_capability(device)[0] == 9
    ):
        # Grouped-main stores need a full tile along N, and SM90 tiles are wide.
        min_tile_n = min(c.tile_n for c in get_all_configs() if c.device_capacity == 9)
        if problem.n < min_tile_n:
            raise NotImplementedError(
                f"FlexGEMM grouped main outputs on SM90 need GEMM N >= {min_tile_n} "
                f"(the smallest SM90 tile_n), got {problem.n}"
            )
    legal = legal_mod_configs(
        epimod, device, problem, preferred_config=flex_gemm_preferred_config(problem)
    )
    if mat1.get_dtype() is torch.float32:
        a_k_major = statically_known_equal(mat1.get_stride()[-1], 1)
        b_n_major = statically_known_equal(mat2.get_stride()[-1], 1)
        # MN-major TF32 needs a 32-element SW128_32B atom per CTA. Two-CTA MMA
        # halves B's N extent; smaller swizzles can silently produce zeros.
        tf32_legal = []
        for c in legal:
            b_mn_major = not a_k_major if c.swap_ab else b_n_major
            if c.device_capacity in (10, 11) and b_mn_major:
                cta_m = cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity)
                mma_ctas = c.tile_m // cta_m
                if c.tile_n % (32 * mma_ctas):
                    continue
            tf32_legal.append(c)
        legal = tuple(tf32_legal)
    return tuple(
        tuple(sorted(dataclasses.asdict(quack_config).items()))
        for quack_config in legal
    )


def flex_gemm_autotune_view_input(node: ir.ReinterpretView) -> torch.Tensor:
    """Rebuild a logical view for Python-backed template benchmarks."""
    from torch._inductor.select_algorithm import (
        AlgorithmSelectorCache,
        get_strides_with_layout_constraints,
    )
    from torch._inductor.virtualized import V

    value = AlgorithmSelectorCache.benchmark_example_value(node)
    base = value if value._base is None else value._base
    sizevars = V.graph.sizevars
    sizes = sizevars.optimization_hints(node.get_size())
    strides = sizevars.optimization_hints(get_strides_with_layout_constraints(node))
    offset = sizevars.optimization_hint(node.get_layout().offset)
    return torch.as_strided(base, sizes, strides, offset)


def lower_quack_flex_gemm(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Lower FlexGEMM through the generated QUACK CuTeDSL template.

    The current pipeline is:

    ::

        FlexGEMM HOP body (FX GraphModule)
                         |
                         v
          find GEMM operands and captured epilogue args
                         |
                         v
             analyze_flex_gemm_epilogue()
                         |
                         +--> output plan + buffer ABI
                         |        `--> derive layout + allocate aux
                         |
                         +--> grouped/reduction geometry
                         |        `--> flex_gemm_quack_configs()
                         |
                         `--> materialize_flex_gemm_epilogue()
                                  `--> CuTeDSL epilogue + callbacks
                         |
                         v
             combine into template choices -> autotune
                         |
                         v
              restore captured output order
    """
    if gemm_op not in FLEX_GEMM_OP_SPECS:
        raise NotImplementedError(
            f"FlexGEMM QUACK backend currently supports only aten.{_SUPPORTED_FLEX_GEMM_OP_NAMES}"
        )
    autotune = kernel_options.get("autotune", False)
    fast_math = kernel_options.get("fast_math", False)
    explicit_config = kernel_options.get("config")
    unsupported_options = OrderedSet(kernel_options) - OrderedSet(
        ["backend", "autotune", "fast_math", "config"]
    )
    if unsupported_options:
        raise NotImplementedError(
            f"unsupported FlexGEMM kernel options: {sorted(unsupported_options)}"
        )
    if not isinstance(fast_math, bool):
        raise NotImplementedError("FlexGEMM fast_math kernel option must be bool")
    if "config" in kernel_options and not isinstance(explicit_config, dict):
        raise NotImplementedError("FlexGEMM config kernel option must be a dict")
    config_constraints = {} if explicit_config is None else dict(explicit_config)
    if config_constraints:
        from torch._vendor.quack.gemm_config import GemmConfig

        config_fields = OrderedSet(
            field.name for field in dataclasses.fields(GemmConfig)
        )
        unknown_fields = OrderedSet(config_constraints) - config_fields
        if unknown_fields:
            raise NotImplementedError(
                f"unknown GemmConfig constraint {sorted(unknown_fields)}; "
                f"choose one of {', '.join(config_fields)}"
            )

    explicit_swap_ab = config_constraints.get("swap_ab") is True

    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        analyze_flex_gemm_epilogue,
        epimod_local_reduce_spec,
        expand_epimod_prepare_softmax_online,
        gemm_node as flex_gemm_node,
        materialize_flex_gemm_epilogue,
    )
    from torch._inductor.kernel.flex_gemm.template import (
        flex_gemm_epilogue_template,
        FlexGemmEpilogueConfig,
        FlexGemmEpilogueLocalReduceConfig,
    )
    from torch._inductor.select_algorithm import autotune_select_algorithm

    op_spec = FLEX_GEMM_OP_SPECS[gemm_op]
    mat1_index = op_spec.mat1_index
    gemm_fx_node = flex_gemm_node(subgraph.graph_module, gemm_op)
    scaled_mm = gemm_op is torch.ops.aten._scaled_mm_v2.default
    grouped_mm = gemm_op is torch.ops.aten._grouped_mm.default
    placeholders = [
        node for node in subgraph.graph_module.graph.nodes if node.op == "placeholder"
    ]
    placeholder_args = dict(zip(placeholders, args, strict=True))
    blockscaled = None
    varlen_axis: Literal["m", "k"] | None = None
    mainloop_scale_nodes: tuple[torch.fx.Node, ...] = ()
    if scaled_mm:
        blockscaled = quack_blockscaled_contract(gemm_fx_node)
        gemm_operand_nodes = blockscaled.gemm_inputs
        mainloop_scale_nodes = blockscaled.tensorwise_scales
        alpha, beta = 1.0, 0.0
    elif grouped_mm:
        gemm_operand_nodes, varlen_axis = quack_grouped_mm_contract(gemm_fx_node)
        gemm_fx_node.args = gemm_operand_nodes
        alpha, beta = 1.0, 0.0
    else:
        gemm_operand_nodes = gemm_fx_node.args
        unsupported_gemm_kwargs = OrderedSet(gemm_kwargs) - OrderedSet(
            ["alpha", "beta"]
        )
        if unsupported_gemm_kwargs:
            raise NotImplementedError(
                f"unsupported FlexGEMM GEMM kwargs: {sorted(unsupported_gemm_kwargs)}"
            )
        alpha = gemm_fx_node.kwargs.get("alpha", gemm_kwargs.get("alpha", 1.0))
        beta = gemm_fx_node.kwargs.get("beta", gemm_kwargs.get("beta", 1.0))
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            raise NotImplementedError("FlexGEMM alpha/beta must be static scalars")

    gemm_args: list[TensorBox] = []
    for arg in gemm_operand_nodes:
        gemm_arg = placeholder_args[arg] if isinstance(arg, torch.fx.Node) else arg
        if not isinstance(gemm_arg, TensorBox):
            raise NotImplementedError("FlexGEMM lowering expects tensor GEMM operands")
        gemm_args.append(gemm_arg)
    check_quack_fp32_operand(gemm_args[mat1_index])
    check_quack_sm90_operands(gemm_args[mat1_index], blockscaled is not None)
    if grouped_mm:
        # QuACK's varlen path reads [0, *offs]; pad in-graph so Inductor owns
        # the buffer instead of the runtime allocating during CUDA graph capture.
        gemm_args[2] = constant_pad_nd(gemm_args[2], [1, 0])
    if varlen_axis == "k":
        # The contract checked FX metadata; pointwise producers would otherwise
        # realize contiguous, turning m-major A into K-major storage. The cast
        # holds because require_stride_order returns a TensorBox for one.
        for index, order in ((0, [0, 1]), (1, [1, 0])):
            gemm_args[index] = cast(
                TensorBox, ir.ExternKernel.require_stride_order(gemm_args[index], order)
            )
    epilogue_arg_placeholders = (
        *mainloop_scale_nodes,
        *flex_gemm_epilogue_arg_placeholders(subgraph.graph_module, gemm_fx_node),
    )
    epilogue_args: list[TensorBox] = []
    for arg in epilogue_arg_placeholders:
        epilogue_arg = placeholder_args[arg]
        if not isinstance(epilogue_arg, TensorBox):
            raise NotImplementedError(
                "FlexGEMM lowering expects tensor epilogue operands"
            )
        epilogue_args.append(epilogue_arg)
    capture_start = len(mainloop_scale_nodes)
    epilogue_args[capture_start:] = normalize_flex_gemm_1d_captures(
        subgraph.graph_module,
        epilogue_arg_placeholders[capture_start:],
        epilogue_args[capture_start:],
        gemm_fx_node.meta["val"].shape,
    )
    if gemm_op is torch.ops.aten.bmm.default and epilogue_args:
        validate_bmm_capture_views(subgraph.graph_module)
    gemm_input_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else f"gemm_arg{index}"
        for index, arg in enumerate(gemm_operand_nodes)
    )
    log_flex_gemm_artifact(
        "problem",
        lambda: format_flex_gemm_problem(
            subgraph.graph_module,
            gemm_op,
            tuple(zip(gemm_input_names, gemm_args, strict=True)),
            tuple(
                zip(
                    (node.name for node in epilogue_arg_placeholders),
                    epilogue_args,
                    strict=True,
                )
            ),
            alpha=float(alpha),
            beta=float(beta),
            autotune=autotune,
            fast_math=fast_math,
            explicit_config=explicit_config,
        ),
        lowering_name=subgraph.name,
    )
    # Normalize supported online-softmax forms before shared analysis.
    expand_epimod_prepare_softmax_online(subgraph.graph_module)
    epilogue_analysis = analyze_flex_gemm_epilogue(subgraph.graph_module, gemm_fx_node)
    log_flex_gemm_artifact(
        "analysis",
        lambda: format_flex_gemm_analysis(epilogue_analysis),
        lowering_name=subgraph.name,
    )
    log_flex_gemm_artifact(
        "analysis_details",
        lambda: format_flex_gemm_analysis_details(epilogue_analysis),
        lowering_name=subgraph.name,
        verbose=True,
    )
    if (
        epilogue_analysis.required_geometries
        and gemm_op not in QUACK_EPILOGUE_FEATURE_OPS
    ):
        raise NotImplementedError(LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR)
    outputs = epilogue_analysis.outputs
    if grouped_mm and outputs.local_reduce is not None:
        # Varlen-M rows never share a group along N, so a stored N-group reduction
        # is per-row and group-safe; M groups would straddle experts. Feed-main is
        # untested under varlen-M (span-1 feed-main also needs the accumulator
        # prepass QuACK rejects with varlen).
        local_reduce = outputs.local_reduce
        store = local_reduce.store
        if (
            varlen_axis != "m"
            or local_reduce.match.geometry.axis != 1
            or local_reduce.feeds_main
            or store is None
            or store.output_layout is not None
        ):
            raise NotImplementedError(
                "FlexGEMM QUACK grouped_mm (varlen) supports grouped reductions only "
                "as stored N-group partials under varlen-M"
            )
        if (
            epimod_local_reduce_spec(epilogue_analysis, local_reduce).prepass
            is not None
        ):
            raise NotImplementedError(
                "FlexGEMM QUACK grouped_mm (varlen) does not support reduction chains "
                "that need QuACK's accumulator prepass"
            )
    output_transform = outputs.output_transform
    if varlen_axis == "k" and output_transform is not None:
        raise NotImplementedError(
            "FlexGEMM varlen-K does not support output contractions or expansions"
        )
    if (
        blockscaled is not None
        and isinstance(output_transform, FlexGemmOutputContraction)
        and output_transform.chunked
    ):
        # TODO: Upstream block-scaled concat_layout support to QuACK if needed.
        raise NotImplementedError(
            "FlexGEMM block-scaled GEMMs do not yet support chunked output contractions"
        )
    local_reduce_store = (
        None if outputs.local_reduce is None else outputs.local_reduce.store
    )
    output_meta = outputs.output.meta.get("val")
    if output_meta is None:
        raise NotImplementedError(
            "FlexGEMM generated epilogues require output metadata"
        )
    logical_output_size = ir.convert_shape_to_inductor(output_meta.shape)
    gemm_size = ir.convert_shape_to_inductor(gemm_fx_node.meta["val"].shape)
    # Captures broadcast against the epilogue's lane shape: the GEMM shape when
    # the main output interleaves two lanes, else the (contracted) output shape.
    epilogue_lane_size = (
        gemm_size
        if isinstance(output_transform, FlexGemmOutputExpansion)
        else logical_output_size
    )
    # Auxiliary buffers keep the GEMM shape, including saved chunked pre-activations.
    aux_metas = validate_flex_gemm_aux_outputs(gemm_op, outputs.aux_outputs, gemm_size)
    if (
        isinstance(output_transform, FlexGemmOutputContraction)
        and output_transform.chunked
        and aux_metas
    ):
        alignment = max(16 // aux_metas[0].dtype.itemsize, 1)
        if aux_metas[0].shape[-1] // 2 % alignment:
            raise NotImplementedError(
                "FlexGEMM chunked accumulator auxiliary requires 16-byte aligned halves"
            )
    if varlen_axis == "k":
        for meta in (output_meta, *aux_metas):
            alignment = max(16 // meta.dtype.itemsize, 1)
            if meta.numel() != 0 and any(
                stride % alignment != 0 for stride in meta.stride()[:-1]
            ):
                raise NotImplementedError(
                    "FlexGEMM varlen-K requires 16-byte aligned output strides"
                )
    if not has_flex_gemm_quack():
        raise NotImplementedError("FlexGEMM QUACK backend requires CuTeDSL")
    # A terminal ``view(dtype)`` reinterprets bits: the kernel stores the source
    # dtype and the result is re-viewed below.
    output_storage = outputs.output_storage
    output_storage_dtype = (
        output_meta.dtype
        if output_storage is None
        else output_storage.meta["val"].dtype
    )
    if (
        isinstance(output_transform, FlexGemmOutputExpansion)
        and output_transform.chunked
    ):
        alignment = max(16 // output_storage_dtype.itemsize, 1)
        if gemm_fx_node.meta["val"].shape[-1] % alignment:
            raise NotImplementedError(
                "FlexGEMM chunked output expansion requires 16-byte aligned halves"
            )
    packed_uint8_main = (
        isinstance(output_transform, FlexGemmOutputContraction)
        and output_storage_dtype is torch.uint8
    )
    if (
        not output_storage_dtype.is_floating_point
        and output_storage_dtype is not torch.bool
        and not packed_uint8_main
    ):
        raise NotImplementedError(
            "FlexGEMM generic main outputs support only floating-point and bool dtypes"
        )
    local_reduce_metas = flex_gemm_local_reduce_metas(outputs.local_reduce)
    if isinstance(output_transform, FlexGemmOutputContraction):
        # Grouped-main kernel stores use compact N rows through TMA, even
        # when the epilogue returns a strided lane view. Keep outer strides
        # 16-byte aligned when the contracted N extent is not aligned.
        # TODO: Consider relaxing exact output-stride preservation for epilogue
        # views of acc, so bare lane selections can return this compact layout
        # without Inductor adding a layout-restoration copy.
        output_stride = ir.FlexibleLayout.contiguous_strides(logical_output_size)
        output_alignment = max(16 // output_storage_dtype.itemsize, 1)
        output_stride[-2] = (
            ceildiv(logical_output_size[-1], output_alignment) * output_alignment
        )
    else:
        output_stride = ir.convert_shape_to_inductor(output_meta.stride())
    layout = ir.FixedLayout(
        gemm_args[mat1_index].get_device_or_error(),
        output_storage_dtype,
        logical_output_size,
        output_stride,
    )
    mainloop_scale_count = len(mainloop_scale_nodes)
    epilogue_arg_kinds = (
        *("scalar" for _ in range(mainloop_scale_count)),
        *infer_flex_gemm_epilogue_arg_kinds(
            gemm_op, epilogue_args[mainloop_scale_count:], epilogue_lane_size
        ),
    )
    if gemm_op is torch.ops.aten.bmm.default and epilogue_args:
        if output_transform is not None:
            raise NotImplementedError(
                "FlexGEMM bmm captured tensor reads require a shape-preserving main output"
            )
        # Batch-shared captures become the 2-D forms QuACK broadcasts over every
        # batch; the runtime then passes rows and columns rank-1.
        epilogue_args = [
            view(arg, arg.get_size()[1:])
            if kind != "tile"
            and len(arg.get_size()) == 3
            and statically_known_equal(arg.get_size()[0], 1)
            else arg
            for arg, kind in zip(epilogue_args, epilogue_arg_kinds, strict=True)
        ]
    gemm_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in gemm_args
    ]
    if blockscaled is None:
        for name, index in (("mat1", op_spec.mat1_index), ("mat2", op_spec.mat2_index)):
            operand = gemm_input_nodes[index]
            if ir.is_unaligned(operand):
                raise NotImplementedError(
                    f"FlexGEMM QUACK {name} requires a 16-byte aligned base"
                )
            validate_flex_gemm_matrix_layout(name, operand.get_layout())
        validate_flex_gemm_matrix_layout("output", layout)
        for meta in aux_metas:
            validate_flex_gemm_matrix_layout(
                "auxiliary output",
                ir.FixedLayout(
                    layout.device,
                    meta.dtype,
                    ir.convert_shape_to_inductor(meta.shape),
                    ir.convert_shape_to_inductor(meta.stride()),
                ),
            )
    if isinstance(output_transform, FlexGemmOutputContraction):
        if output_transform.chunked and statically_known_equal(
            gemm_input_nodes[op_spec.mat2_index].get_stride()[-1], 1
        ):
            raise NotImplementedError(
                "FlexGEMM chunked grouped main output requires column-major B storage"
            )
        if "tile" in epilogue_arg_kinds or "interleaved" in epilogue_arg_kinds:
            raise NotImplementedError(
                "FlexGEMM grouped main outputs support only scalar, row, and column "
                "captured tensors"
            )
        # Row captures index logical columns; the kernel reads physical columns,
        # so repeat each value across its lane group (the epilogue reads lane 0).
        require_lane_domain_row_captures(
            epilogue_analysis.local_reduce,
            gemm_fx_node,
            [
                node
                for node, kind in zip(
                    epilogue_arg_placeholders, epilogue_arg_kinds, strict=True
                )
                if kind == "row"
            ],
        )
        epilogue_args = [
            widen_row_capture(arg, output_transform.group) if kind == "row" else arg
            for arg, kind in zip(epilogue_args, epilogue_arg_kinds, strict=True)
        ]
    epilogue_args = require_tma_loadable_tile_captures(
        epilogue_args, epilogue_arg_kinds
    )
    epilogue_input_nodes = [
        ir.TemplateBuffer.realize_template_input(arg) for arg in epilogue_args
    ]
    aux_outs = allocate_flex_gemm_aux_outs(aux_metas, gemm_args[mat1_index])
    local_reduce_outs = allocate_flex_gemm_aux_outs(
        local_reduce_metas, gemm_args[mat1_index]
    )
    aux_input_nodes = [
        ir.TemplateBuffer.realize_template_input(aux_out) for aux_out in aux_outs
    ]
    local_reduce_input_nodes = [
        ir.TemplateBuffer.realize_template_input(local_reduce_out)
        for local_reduce_out in local_reduce_outs
    ]
    input_nodes = [
        *gemm_input_nodes,
        *epilogue_input_nodes,
        *aux_input_nodes,
        *local_reduce_input_nodes,
    ]
    mutated_input_nodes = aux_input_nodes + local_reduce_input_nodes
    aux_out_start = len(gemm_input_nodes) + len(epilogue_input_nodes)
    aux_out_indices = tuple(range(aux_out_start, aux_out_start + len(aux_input_nodes)))
    gemm_input_indices = tuple(range(len(gemm_input_nodes)))
    local_reduce_out_index = (
        aux_out_start + len(aux_input_nodes) if local_reduce_input_nodes else None
    )
    if gemm_args[mat1_index].get_device_or_error().type != "cuda":
        raise NotImplementedError("FlexGEMM QUACK backend requires CUDA tensors")
    epimod_source = materialize_flex_gemm_epilogue(
        subgraph.graph_module,
        epilogue_analysis,
        epilogue_arg_placeholders,
        float(alpha),
        float(beta),
        epilogue_arg_kinds,
        fast_math=fast_math,
        swap_ab=explicit_swap_ab,
        mainloop_scale_count=mainloop_scale_count,
    )
    log_flex_gemm_artifact(
        "lowering_plan",
        lambda: format_flex_gemm_lowering_plan(
            logical_output_size,
            output_meta.dtype,
            tuple(
                zip(
                    (node.name for node in epilogue_arg_placeholders),
                    epilogue_arg_kinds,
                    strict=True,
                )
            ),
            aux_metas,
            local_reduce_metas,
            local_reduce_layout=(
                None if local_reduce_store is None else local_reduce_store.output_layout
            ),
        ),
        lowering_name=subgraph.name,
    )
    log_flex_gemm_artifact(
        "generated_epilogue",
        lambda: epimod_source.source.strip(),
        lowering_name=subgraph.name,
        verbose=True,
    )
    template_local_reduce = FlexGemmEpilogueLocalReduceConfig.from_plan(
        outputs.local_reduce, local_reduce_out_index, epimod_source
    )
    if epimod_source.local_reduce_fragment_reduced:
        # Fragment partials are lowered for the unswapped accumulator geometry.
        config_constraints["swap_ab"] = False
    epilogue_arg_indices = tuple(
        range(
            len(gemm_input_nodes),
            len(gemm_input_nodes) + len(epilogue_input_nodes),
        )
    )
    template_config = FlexGemmEpilogueConfig(
        epilogue_name=epimod_source.name,
        epilogue_source=epimod_source.source,
        gemm_op=op_spec,
        alpha=float(alpha),
        beta=float(beta),
        blockscaled_format=None if blockscaled is None else blockscaled.format,
        quack_config=None,
        varlen_axis=varlen_axis,
        epilogue_arg_indices=epilogue_arg_indices,
        epilogue_arg_kinds=epilogue_arg_kinds,
        aux_out_indices=aux_out_indices,
        local_reduce=template_local_reduce,
        output_transform=output_transform,
    )
    template_kwargs = dict(
        input_nodes=input_nodes,
        layout=layout,
        mutated_inputs=mutated_input_nodes or None,
    )
    legal_configs = flex_gemm_quack_configs(
        template_config, gemm_input_nodes, epilogue_input_nodes
    )
    if config_constraints:
        legal_configs = tuple(
            config
            for config in legal_configs
            if all(
                type(field) is type(value) and field == value
                for name, value in config_constraints.items()
                for field in (dict(config)[name],)
            )
        )
        if not legal_configs:
            raise NotImplementedError(
                "no supported GemmConfig matches "
                f"config_constraints={config_constraints!r} for this call"
            )
    from torch._inductor.virtualized import V

    sizevars = V.graph.sizevars
    mat1, mat2 = (gemm_input_nodes[i] for i in (op_spec.mat1_index, op_spec.mat2_index))
    m_hint = sizevars.optimization_hint(mat1.get_size()[-2])
    n_hint = sizevars.optimization_hint(mat2.get_size()[-1])
    # Block-scaled calls keep QuACK's shape-aware blockscaled default.
    dense_shape = None if blockscaled is not None else (m_hint, n_hint)
    if autotune:
        quack_configs = flex_gemm_search_space(
            legal_configs,
            varlen=varlen_axis == "m",
            dense_shape=dense_shape if gemm_op is torch.ops.aten.mm.default else None,
        )
    else:
        default = flex_gemm_default_config(
            legal_configs, varlen=varlen_axis == "m", dense_shape=dense_shape
        )
        quack_configs = (default,)
    log_flex_gemm_artifact(
        "config_candidates",
        lambda: format_flex_gemm_config_candidates(quack_configs, autotune=autotune),
        lowering_name=subgraph.name,
    )
    choices: list[Any] = []
    for quack_config in quack_configs:
        error = flex_gemm_epilogue_template.maybe_append_choice(
            choices,
            config=dataclasses.replace(template_config, quack_config=quack_config),
            **template_kwargs,
        )
        if error is not None:
            raise error
    input_gen_fns = {
        index: flex_gemm_autotune_view_input
        for index, input_node in enumerate(input_nodes)
        if isinstance(input_node, ir.ReinterpretView)
    }
    if grouped_mm:
        input_gen_fns[gemm_input_indices[2]] = functools.partial(
            flex_gemm_cu_seqlens_benchmark_input,
            total_length=mat1.get_size()[-1]
            if varlen_axis == "k"
            else logical_output_size[0],
        )
    result, _ = autotune_select_algorithm(
        "flex_gemm_epilogue",
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns or None,
        **({"return_multi_template": False} if mutated_input_nodes else {}),
    )
    if autotune and len(choices) == 1:
        # autotune_select_algorithm skips a lone candidate; compile it now so
        # autotune=True never defers compilation to the first call. Other calls keep
        # lazy first-call compilation and allocate no example tensors.
        choices[0].precompile(use_workers=False)
    structural_outs = {}
    if local_reduce_store is not None:
        structural_outs[local_reduce_store.node] = local_reduce_outs[0]
    aux_iter = iter(aux_outs)
    ordered_aux_outs = [
        structural_outs[node] if node in structural_outs else next(aux_iter)
        for node in outputs.returned_aux_outputs
    ]
    if output_storage_dtype is not output_meta.dtype:
        result = TensorBox(ir.DtypeView.create(result, output_meta.dtype))
    return (result, *ordered_aux_outs)


@register_lowering(flex_gemm_hop, type_promotion_kind=None)
def flex_gemm_lowering(gemm_op, subgraph, args, gemm_kwargs, kernel_options):
    """Dispatch FlexGEMM to ordinary Inductor lowering or the QUACK template."""
    backend = kernel_options.get("backend", "TRITON")
    body_gemm_op = flex_gemm_body_gemm_op(gemm_op, gemm_kwargs)
    if body_gemm_op is torch.ops.aten._scaled_mm_v2.default and backend != "QUACK":
        raise NotImplementedError("FlexGEMM F.scaled_mm requires backend='QUACK'")
    if backend == "NVGEMM" and gemm_op in FLEX_GEMM_OP_SPECS:
        mat1 = args[FLEX_GEMM_OP_SPECS[gemm_op].mat1_index]
        if isinstance(mat1, TensorBox) and is_bf16x9_matmul(
            mat1.get_device_or_error().type, mat1.get_dtype()
        ):
            # See Note [BF16x9 precision] in torch/_inductor/utils.py.
            warning_once(
                log,
                "FlexGEMM NVGEMM does not support bfx9 precision; using ATen/cuBLAS instead.",
            )
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    if backend == "NVGEMM":
        unsupported_options = OrderedSet(kernel_options) - OrderedSet(
            ("backend", "autotune")
        )
        if unsupported_options:
            raise NotImplementedError(
                f"Unsupported NVGEMM FlexGEMM options: {unsupported_options}"
            )
        autotune = kernel_options.get("autotune", False)
        if not isinstance(autotune, bool):
            raise NotImplementedError("NVGEMM FlexGEMM autotune must be a bool")
        decompose_nvgemm_additive_gemm(subgraph.graph_module)
        # NVGEMM templates require max_autotune. autotune=False caps each heuristic
        # family at one config; supplemental wide-tile EFC choices may remain.
        nvgemm_config: dict[str, Any] = {
            "max_autotune": True,
            "max_autotune_gemm_backends": "NVGEMM",
            "nvgemm_max_profiling_configs": 1,
        }
        if autotune:
            nvgemm_config.update(
                nvgemm_max_profiling_configs=None,
                nvgemm_supplement_configs=True,
                nvgemm_swap_ab=True,
            )
        with config.patch(nvgemm_config):
            return process_subgraph_nodes(subgraph.graph_module, list(args))
    if backend == "QUACK":
        return lower_quack_flex_gemm(
            body_gemm_op, subgraph, args, gemm_kwargs, kernel_options
        )
    return process_subgraph_nodes(subgraph.graph_module, list(args))
