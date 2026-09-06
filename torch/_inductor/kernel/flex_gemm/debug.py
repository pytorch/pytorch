"""Format opt-in diagnostics for the simplified FlexGEMM lowering path."""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import LazyString, trace_structured


if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.kernel.flex_gemm.epilogue import FlexGemmEpilogueAnalysis
    from torch._inductor.kernel.flex_gemm.output_layout import FlexGemmOutputLayout


flex_gemm_log = logging.getLogger(__name__)


def log_flex_gemm_artifact(
    name: str,
    payload_fn: Callable[[], str],
    *,
    lowering_name: str | None = None,
    verbose: bool = False,
) -> None:
    """Emit one lazily formatted local and structured lowering phase."""
    heading = "FLEXGEMM LOWERING"
    if lowering_name is not None:
        heading += f" [{lowering_name}]"
    flex_gemm_log.log(
        logging.DEBUG if verbose else logging.INFO,
        "%s\n ===== %s =====\n%s",
        heading,
        name.replace("_", " ").upper(),
        LazyString(payload_fn),
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": f"flex_gemm_{name}",
            "encoding": "string",
            "verbosity": "verbose" if verbose else "concise",
            **({} if lowering_name is None else {"lowering_name": lowering_name}),
        },
        payload_fn=payload_fn,
    )


def append_items(lines: list[str], label: str, items: Iterable[str]) -> None:
    """Append a report section with an explicit empty marker."""
    values = tuple(items)
    lines.append(f"{label}:")
    lines.extend(f"  {value}" for value in values or ("(none)",))


def format_fx_tensor(node: torch.fx.Node) -> str:
    """Format an FX tensor using captured fake metadata."""
    meta = node.meta.get("val")
    if not isinstance(meta, torch.Tensor):
        return node.name
    return (
        f"{node.name}: shape={tuple(meta.shape)}, stride={tuple(meta.stride())}, "
        f"dtype={meta.dtype}"
    )


def format_geometry(geometry: Any) -> str:
    """Format grouped reduction geometry in caller M/N coordinates."""
    return f"axis={'M' if geometry.axis == 0 else 'N'}, group={geometry.group}"


def format_output_layout(layout: "FlexGemmOutputLayout | None") -> str:
    """Format a dense or caller-owned physical output layout."""
    return "dense" if layout is None else layout.name


def format_main_transform(transform: Any | None) -> str:
    """Format an optional grouped-main contraction."""
    if transform is None:
        return "none"
    return (
        f"grouped-N, group={transform.group}, "
        f"layout={'chunked' if transform.chunked else 'interleaved'}"
    )


def format_flex_gemm_analysis(analysis: "FlexGemmEpilogueAnalysis") -> str:
    """Render semantic output, reduction, and config decisions."""
    outputs = analysis.outputs
    lines = [
        "outputs:",
        f"  main: {format_fx_tensor(outputs.output)}",
        f"  main_transform: {format_main_transform(outputs.main_transform)}",
    ]
    append_items(
        lines,
        "auxiliary",
        (format_fx_tensor(output) for output in outputs.aux_outputs),
    )
    if outputs.indexed_output is not None:
        lines.extend(
            (
                "indexed:",
                f"  output: {format_fx_tensor(outputs.indexed_output.node)}",
                f"  indices: {format_fx_tensor(outputs.indexed_output.indices)}",
            )
        )
    lines.append("")
    if outputs.local_reduce is None:
        lines.append("local_reduction: none")
    else:
        local_reduce = outputs.local_reduce
        consumers = []
        if local_reduce.feeds_main:
            consumers.append("main")
        if local_reduce.store is not None:
            consumers.append("returned")
        lines.extend(
            (
                "local_reduction:",
                f"  value: {local_reduce.match.value_node.name}",
                f"  geometry: {format_geometry(local_reduce.match.geometry)}",
                f"  consumers: {' + '.join(consumers)}",
            )
        )
        if local_reduce.store is not None:
            lines.extend(
                (
                    f"  returned_as: {local_reduce.store.node.name}",
                    "  output_layout: "
                    f"{format_output_layout(local_reduce.store.output_layout)}",
                )
            )
    lines.append("")
    append_items(
        lines,
        "config_constraints",
        (format_geometry(geometry) for geometry in analysis.required_geometries),
    )
    return "\n".join(lines)


def format_flex_gemm_analysis_details(
    analysis: "FlexGemmEpilogueAnalysis",
) -> str:
    """Render recognizer maps for detailed debugging."""
    lines: list[str] = []
    append_items(
        lines,
        "fx_nodes",
        (
            f"{node.name}: {node.target}"
            for node in analysis.local_reduce.graph.dependencies
            if node.op == "call_function"
        ),
    )
    lines.append("")
    for label, values in (
        ("grouped_tensors", analysis.local_reduce.grouped_tensors),
        ("local_reduce_matches", analysis.local_reduce.matches),
        ("grouped_select_indices", analysis.grouped_select_indices),
        ("grouped_main_layouts", analysis.grouped_main_layouts),
    ):
        append_items(
            lines,
            label,
            (f"{node.name}: {value!r}" for node, value in values.items()),
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_ir_tensor(name: str, node: "ir.IRNode") -> str:
    """Format one Inductor tensor contract."""
    stride = node.maybe_get_stride()
    return (
        f"{name}: shape={tuple(node.get_size())}, "
        f"stride={'unrealized' if stride is None else tuple(stride)}, "
        f"dtype={node.get_dtype()}"
    )


def format_flex_gemm_problem(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    gemm_inputs: Sequence[tuple[str, "ir.IRNode"]],
    captures: Sequence[tuple[str, "ir.IRNode"]],
    *,
    tuned: bool,
    fast_math: bool,
    explicit_config: dict[str, Any] | None,
) -> str:
    """Render the captured problem entering semantic analysis."""
    lines = [
        f"gemm_op: {gemm_op}",
        f"tuned: {tuned}",
        f"fast_math: {fast_math}",
        f"config: {explicit_config!r}",
    ]
    append_items(
        lines, "gemm_inputs", (format_ir_tensor(*item) for item in gemm_inputs)
    )
    append_items(lines, "captures", (format_ir_tensor(*item) for item in captures))
    lines.extend(("body:", graph_module.print_readable(print_output=False).strip()))
    return "\n".join(lines)


def format_flex_gemm_lowering_plan(
    output_size: Sequence[Any],
    output_dtype: torch.dtype,
    capture_kinds: Sequence[tuple[str, str]],
    aux_metas: Sequence[torch.Tensor],
    indexed_metas: Sequence[torch.Tensor],
    local_reduce_metas: Sequence[torch.Tensor],
    *,
    local_reduce_layout: "FlexGemmOutputLayout | None",
) -> str:
    """Render allocation and runtime-ABI decisions."""
    lines = [f"output: shape={tuple(output_size)}, dtype={output_dtype}"]
    append_items(
        lines, "captures", (f"{name} -> {kind}" for name, kind in capture_kinds)
    )
    append_items(
        lines,
        "auxiliary_storage",
        (f"shape={tuple(meta.shape)}, dtype={meta.dtype}" for meta in aux_metas),
    )
    append_items(
        lines,
        "indexed_storage",
        (f"shape={tuple(meta.shape)}, dtype={meta.dtype}" for meta in indexed_metas),
    )
    append_items(
        lines,
        "local_reduction_storage",
        (
            f"shape={tuple(meta.shape)}, dtype={meta.dtype}"
            for meta in local_reduce_metas
        ),
    )
    lines.append(
        "local_reduction_layout: "
        + (
            "none"
            if not local_reduce_metas
            else format_output_layout(local_reduce_layout)
        )
    )
    return "\n".join(lines)


def format_flex_gemm_config_candidates(configs: Any, *, tuned: bool) -> str:
    """Render the QuACK configs Inductor will benchmark or pin."""
    lines = [
        f"mode: {'autotune' if tuned else 'default'}",
        f"candidates: {len(configs)}",
    ]
    for config in configs:
        fields = dict(config)
        lines.append(
            "  tile=({tile_m}, {tile_n}) cluster=({cluster_m}, {cluster_n}) "
            "swap_ab={swap_ab} dynamic_persistent={is_dynamic_persistent}".format(
                **fields
            )
        )
    return "\n".join(lines)
