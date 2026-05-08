import json
import logging
import math
from collections.abc import Callable

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_all_reduce_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_all_reduce_tensor,
    merge_all_gather,
    merge_all_reduce_bucket,
    merge_reduce_scatter,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_fsdp_all_gather(n: torch.fx.Node) -> bool:
    """Check if an all_gather derives from exactly one placeholder (parameter).

    Uses backward BFS to count placeholder ancestors across all input branches.
    Handles multi-input chains (e.g. cat(param, zeros) for padding) that the old
    single-input-chain walk would miss.
    """
    assert is_all_gather(n)
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = list(n.all_input_nodes)
    placeholders = 0
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op == "placeholder":
            placeholders += 1
            if placeholders > 1:
                return False
        else:
            queue.extend(node.all_input_nodes)
    return placeholders == 1


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)  # type: ignore[arg-type]


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    """Check if a reduce_scatter wait flows only to graph outputs through unary ops.

    Uses forward walk to verify every path from *wait* reaches an output node and
    every intermediate node is unary (single input). This is conservative: it
    rejects multi-input nodes even if all inputs derive from the RS, but that is
    sufficient for current FSDP2 patterns where RS→output is always a unary chain
    (view/reshape/cast).

    Returns False for compiled multi-microbatch gradient accumulation where
    add(existing_grad, rs_result) appears in the chain. Current FSDP2 compile
    patterns don't produce this (each microbatch is compiled separately).
    """
    if not wait.users:
        return False
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = [wait]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for user in node.users:
            if user.op == "output":
                continue
            if len(user.all_input_nodes) != 1:
                return False
            queue.append(user)
    return True


_LINEAR_REDUCE_OPS = OrderedSet(["sum", "avg"])

_dedup_rs_pass: PatternMatcherPass | None = None


def _get_dedup_rs_pass() -> PatternMatcherPass:
    global _dedup_rs_pass

    if _dedup_rs_pass is not None:
        return _dedup_rs_pass

    c10d = torch.ops._c10d_functional
    aten = torch.ops.aten
    dedup_rs_pass = PatternMatcherPass(pass_name="dedup_reduce_scatter")

    def wait_rs(name: str) -> CallFunction:
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                KeywordArg(name),
                KeywordArg("reduce_op"),
                KeywordArg("group_size"),
                KeywordArg("group_name"),
            ),
        )

    def dedup_rs_extra_check(match: Match) -> bool:
        if match.kwargs["reduce_op"] not in _LINEAR_REDUCE_OPS:
            return False
        for node in match.nodes:
            if node.target is aten.add.Tensor:
                continue
            if node.target not in (
                c10d.wait_tensor.default,
                c10d.reduce_scatter_tensor.default,
            ):
                return False
            if len(node.users) != 1:
                return False
        input_a = match.kwargs["input_a"]
        input_b = match.kwargs["input_b"]
        if input_a.meta["val"].dtype != input_b.meta["val"].dtype:
            return False
        return True

    @register_graph_pattern(
        CallFunction(
            aten.add.Tensor,
            wait_rs("input_a"),
            wait_rs("input_b"),
        ),
        extra_check=dedup_rs_extra_check,
        # pyrefly: ignore[bad-argument-type]
        pass_dict=dedup_rs_pass,
    )
    def _(match: Match, input_a, input_b, reduce_op, group_size, group_name):
        def repl(input_a, input_b):
            combined = aten.add.Tensor(input_a, input_b)
            rs = c10d.reduce_scatter_tensor.default(
                combined, reduce_op, group_size, group_name
            )
            return c10d.wait_tensor.default(rs)

        # pyrefly: ignore[bad-argument-type]
        match.replace_by_example(repl, [input_a, input_b])

    _dedup_rs_pass = dedup_rs_pass
    return dedup_rs_pass


def dedup_fsdp_reduce_scatter(gm: torch.fx.GraphModule) -> None:
    """
    Fuse duplicate reduce_scatter ops whose waited results are summed.

    RS is linear, so RS(a) + RS(b) = RS(a + b). This pass rewrites
        rs_a = reduce_scatter(input_a, ...); wait_a = wait(rs_a)
        rs_b = reduce_scatter(input_b, ...); wait_b = wait(rs_b)
        result = add(wait_a, wait_b)
    into
        combined = add(input_a, input_b)
        rs = reduce_scatter(combined, ...)
        result = wait(rs)

    For N-way add trees (N > 2), the pattern is applied repeatedly
    until fixpoint — each iteration fuses one leaf pair.
    """
    dedup_rs_pass = _get_dedup_rs_pass()
    while dedup_rs_pass.apply(gm):
        pass
    gm.graph.lint()
    gm.recompile()


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    assert bucket_cap_mb_by_bucket_idx is not None
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    # reduce_scatter bucketing does not support multidtype mode;
    # resolve None to the default and strip multidtype if present.
    rs_bucket_mode: BucketMode = mode or "default"
    if "multidtype" in rs_bucket_mode:
        rs_bucket_mode = rs_bucket_mode.replace("_multidtype", "")  # type: ignore[assignment]
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
        mode=rs_bucket_mode,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


def bucket_fsdp_all_reduce(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    fsdp_groups: OrderedSet[str] | None = None,
) -> None:
    """Bucketing pass for FSDP all_reduce ops.

    For all_gather and reduce_scatter we use structural heuristics
    (single-placeholder ancestry for AG, unary-chain-to-output for RS)
    to identify FSDP collectives. For all_reduce there is no reliable
    structural pattern, so we identify FSDP groups via AG/RS first and
    filter by group name here.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default

    def is_fsdp_all_reduce_wait(wait: torch.fx.Node) -> bool:
        ar_node = wait.args[0]
        if not is_all_reduce_tensor(ar_node):  # type: ignore[arg-type]
            return False
        if fsdp_groups is None:
            return True
        return _get_group_name(ar_node) in fsdp_groups  # type: ignore[arg-type]

    ar_buckets = bucket_all_reduce_by_mb(
        gm, bucket_cap_mb_by_bucket_idx, filter_wait_node=is_fsdp_all_reduce_wait
    )
    for bucket in ar_buckets:
        merge_all_reduce_bucket(gm.graph, bucket)


def _get_collective_kwargs(n: fx.Node) -> dict[str, object]:
    """Normalize a collective node's args into keyword args."""
    from torch.fx.operator_schemas import normalize_function

    opt = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt is not None
    _, kwargs = opt
    return kwargs


def _get_group_name(n: fx.Node) -> str:
    return _get_collective_kwargs(n)["group_name"]  # type: ignore[return-value]


def _get_group_size_from_node(n: fx.Node) -> int:
    return _get_collective_kwargs(n)["group_size"]  # type: ignore[return-value]


def _find_all_gathers(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all all_gather nodes (both default and _out variants) via O(1) lookup."""
    return [
        *graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        ),
        *graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        ),
    ]


def _find_reduce_scatters(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all reduce_scatter nodes via O(1) lookup."""
    return graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
    )


def _find_all_reduces(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all all_reduce nodes via O(1) lookup."""
    return graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.all_reduce.default,
    )


def identify_fsdp_groups(
    gm: torch.fx.GraphModule,
) -> tuple[OrderedSet[str], int | None]:
    """Identify FSDP process groups and return (group_names, group_size).

    Uses is_fsdp_all_gather heuristic on all_gather nodes to find FSDP groups,
    then returns those group names plus the group_size from the first match.
    All collectives on these groups (AG, RS, AR) are considered FSDP via
    group-name transitivity.
    """
    fsdp_groups: OrderedSet[str] = OrderedSet()
    group_size: int | None = None
    for n in _find_all_gathers(gm.graph):
        if is_fsdp_all_gather(n):
            fsdp_groups.add(_get_group_name(n))
            if group_size is None:
                group_size = _get_group_size_from_node(n)
    return fsdp_groups, group_size


def compute_pre_bucket_cap_mb(
    group_size: int,
    bucket_cap_mb_override: float | None = None,
) -> float:
    """Compute the bucket cap for pre-bucketing based on bandwidth saturation.

    Returns a bucket size in MB that targets saturation of the process group's
    network bandwidth. Uses empirical per-interconnect profiles with auto-detection
    of GPU generation and intra/inter-node topology.

    If bucket_cap_mb_override is set, returns that directly.
    """
    if bucket_cap_mb_override is not None:
        return bucket_cap_mb_override

    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    cal_mult = (
        dist_opts.pre_bucketing_fsdp_collectives_saturation_calibration_multiplier
    )
    floor_mb = dist_opts.pre_bucketing_fsdp_collectives_min_bucket_cap_mb
    ceil_mb = dist_opts.pre_bucketing_fsdp_collectives_max_bucket_cap_mb

    min_bytes = 0
    try:
        from torch._inductor.comm_analysis import (
            compute_min_saturation_bytes,  # pyrefly: ignore [missing-module-attribute]
            NCCL_COLL,
        )

        min_bytes = compute_min_saturation_bytes(
            group_size, NCCL_COLL.ALL_GATHER, target_efficiency=0.9
        )
        cap_mb = cal_mult * min_bytes / (1024 * 1024)
    except ImportError:
        cap_mb = floor_mb
    cap_mb = max(floor_mb, min(ceil_mb, cap_mb))

    if dist_opts.pre_bucketing_fsdp_collectives_verbose:
        try:
            from torch._inductor.comm_analysis import (
                detect_interconnect,  # pyrefly: ignore [missing-module-attribute]
                get_inter_node_bw,  # pyrefly: ignore [missing-module-attribute]
                get_intra_node_bw,  # pyrefly: ignore [missing-module-attribute]
            )

            logger.info(
                "pre_bucket_cap: interconnect=%s intra_bw=%.0f inter_bw=%.0f "
                "saturation_bytes=%d (%.1f MB) cal_mult=%.2f cap_mb=%.1f",
                detect_interconnect(group_size).name,
                get_intra_node_bw(),
                get_inter_node_bw(),
                min_bytes,
                min_bytes / (1024 * 1024),
                cal_mult,
                cap_mb,
            )
        except ImportError:
            logger.info("pre_bucket_cap: cap_mb=%.1f (fallback)", cap_mb)

    return cap_mb


def _tensor_size_mb(val: torch.Tensor) -> float:
    """Return tensor size in MB, using size hints for dynamic shapes."""
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    numel = optimization_hint(val.numel(), fallback=0)
    return numel * val.element_size() / (1024 * 1024)


def _collect_collective_sizes(
    gm: torch.fx.GraphModule, fsdp_groups: OrderedSet[str]
) -> list[dict[str, object]]:
    """Collect per-collective transfer sizes (MB) for FSDP collectives in graph order.

    AG: output tensor size (the gathered result, i.e. bytes on the wire).
    RS: input tensor size (the pre-scatter tensor, i.e. bytes on the wire).
    AR: input tensor size (in-place, same size in and out).
    """
    sizes: list[dict[str, object]] = []
    for n in _find_all_gathers(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.meta["val"])
            sizes.append({"type": "AG", "size_mb": round(size_mb, 3), "name": n.name})
    for n in _find_reduce_scatters(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.all_input_nodes[0].meta["val"])
            sizes.append({"type": "RS", "size_mb": round(size_mb, 3), "name": n.name})
    for n in _find_all_reduces(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.all_input_nodes[0].meta["val"])
            sizes.append({"type": "AR", "size_mb": round(size_mb, 3), "name": n.name})
    return sizes


def pre_bucket_fsdp_collectives(
    gm: torch.fx.GraphModule,
    mode: BucketMode | None = None,
    bucket_cap_mb: float | None = None,
) -> None:
    """Pre-bucket FSDP collectives before overlap scheduling.

    Identifies FSDP process groups via all_gather structural heuristics,
    then merges all_gather, reduce_scatter, and all_reduce ops on  those
    groups into bandwidth-saturating buckets.
    """
    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    verbose = dist_opts.pre_bucketing_fsdp_collectives_verbose

    fsdp_groups, group_size = identify_fsdp_groups(gm)
    if not fsdp_groups:
        return

    def _count_fsdp(nodes: list[fx.Node]) -> int:
        return sum(1 for n in nodes if _get_group_name(n) in fsdp_groups)

    ag_count = _count_fsdp(_find_all_gathers(gm.graph))
    rs_count = _count_fsdp(_find_reduce_scatters(gm.graph))
    ar_count = _count_fsdp(_find_all_reduces(gm.graph))

    if verbose:
        coll_sizes = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives before bucketing, sizes (MB): %s",
            len(coll_sizes),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes),
        )

    if group_size is not None:
        cap_mb = compute_pre_bucket_cap_mb(group_size, bucket_cap_mb)
    else:
        # Reachable when fsdp_groups were identified from all-gathers that
        # were subsequently erased by an earlier pass, or when only RS/AR remain.
        logger.warning("pre_bucket_fsdp: no FSDP all_gather found for group_size")
        cap_mb = bucket_cap_mb if bucket_cap_mb is not None else 500.0

    def bucket_cap_fn(_idx: int) -> float:
        return cap_mb

    resolved_mode: BucketMode = mode or "default"
    bucket_fsdp_all_gather(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_reduce_scatter(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_all_reduce(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, fsdp_groups=fsdp_groups
    )

    ag_count_after = _count_fsdp(_find_all_gathers(gm.graph))
    rs_count_after = _count_fsdp(_find_reduce_scatters(gm.graph))
    ar_count_after = _count_fsdp(_find_all_reduces(gm.graph))

    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    nNodes = math.ceil(group_size / gpus_per_node) if group_size is not None else 1

    # Verbose: log per-collective sizes after bucketing
    if verbose:
        coll_sizes_after = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives after bucketing, sizes (MB): %s",
            len(coll_sizes_after),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes_after[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes_after",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes_after),
        )

    logger.info(
        "pre_bucket_fsdp_collectives: fsdp_groups=%s, group_size=%s, nNodes=%d, "
        "bucket_cap_mb=%.1f, all_gather %d->%d, reduce_scatter %d->%d, "
        "all_reduce %d->%d",
        list(fsdp_groups),
        group_size,
        nNodes,
        cap_mb,
        ag_count,
        ag_count_after,
        rs_count,
        rs_count_after,
        ar_count,
        ar_count_after,
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pre_bucketing_fsdp_collectives",
            "encoding": "string",
        },
        payload_fn=lambda: (
            f"fsdp_groups={list(fsdp_groups)}, group_size={group_size}, "
            f"nNodes={nNodes}, bucket_cap_mb={cap_mb:.1f}, "
            f"all_gather {ag_count}->{ag_count_after}, "
            f"reduce_scatter {rs_count}->{rs_count_after}, "
            f"all_reduce {ar_count}->{ar_count_after}"
        ),
    )
