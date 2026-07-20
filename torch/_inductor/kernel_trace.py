"""Triton Kernel Trace: per-kernel scheduled op structure capture + serialization.

See docs: pytorch/pytorch#189270. Pure-Python; must never break compilation.
"""
from __future__ import annotations
from typing import Any
import sympy

from torch._inductor import config

try:
    from torch._dynamo.utils import signpost_event
except Exception:  # pragma: no cover
    def signpost_event(*a, **k): pass

# addressing/internal nodes that are not part of the emitted value stream
_DROP_TARGETS = {"get_index"}

_COMPUTE_OPS = {"cos","sin","exp","log","sqrt","rsqrt","tanh","sigmoid",
                "reciprocal","pow","erf","reduction"}

_kernel_physical_trace: dict[str, dict] = {}
_last_emitted_trace_json: str | None = None


def _target_name(node: Any) -> str:
    t = node.target
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def _srepr(expr: Any) -> str:
    try:
        return sympy.srepr(expr)
    except Exception:
        return repr(expr)


def _node_identity(nd: Any, indexing_exprs: dict, memo: dict) -> tuple:
    if nd in memo:
        return memo[nd]
    tgt = _target_name(nd)
    if tgt == "load":
        # args: (ops, buffer_name, get_index_node)
        buf = nd.args[1]
        idx_node = nd.args[2]
        idx_name = idx_node.args[0] if hasattr(idx_node, "args") else idx_node
        ident = ("load", str(buf), _srepr(indexing_exprs.get(idx_name)))
    elif tgt == "constant":
        # args: (ops, value, dtype)
        val = nd.args[1] if len(nd.args) > 1 else None
        dt = nd.args[2] if len(nd.args) > 2 else None
        ident = ("constant", repr(val), str(dt))
    elif tgt == "index_expr":
        expr = nd.args[1] if len(nd.args) > 1 else None
        dt = nd.args[2] if len(nd.args) > 2 else None
        ident = ("index_expr", _srepr(expr), str(dt))
    else:
        child = tuple(
            _node_identity(a, indexing_exprs, memo) if hasattr(a, "op") else repr(a)
            for a in nd.args[1:]
        )
        kw = tuple(sorted((k, repr(v)) for k, v in (nd.kwargs or {}).items()))
        ident = (tgt, child, kw)
    memo[nd] = ident
    return ident


def _walk_block(graph: Any, block_id: str, subblocks: dict, order_ref: list, out: list,
                indexing_exprs: dict, memo: dict) -> None:
    """Append ops from one LoopBodyBlock graph in order, recursing masked subblocks."""
    for nd in graph.nodes:
        if nd.op not in ("call_method", "call_function", "call_module"):
            continue
        tgt = _target_name(nd)
        if tgt in _DROP_TARGETS:
            continue
        # a call_module into a named subblock == a masked region; recurse
        if nd.op == "call_module" and isinstance(nd.target, str) and nd.target in subblocks:
            sub = subblocks[nd.target]
            sub_graph = getattr(sub, "graph", None)
            if sub_graph is not None:
                _walk_block(sub_graph, nd.target, subblocks, order_ref, out, indexing_exprs, memo)
            continue
        out.append({
            "order": order_ref[0],
            "target": tgt,
            "block": block_id,
            "args_repr": tuple(_target_name(a) if hasattr(a, "op") else repr(a)
                               for a in nd.args[1:]),  # drop leading `ops` handler arg
            "identity": _node_identity(nd, indexing_exprs, memo),
        })
        order_ref[0] += 1


def extract_leaf_ops(leaf: Any) -> list[dict]:
    body = getattr(leaf, "_body", None)
    if body is None:
        return []
    root = getattr(body, "root_block", None)
    graph = getattr(root, "graph", None)
    if graph is None:
        return []
    subblocks = getattr(body, "subblocks", {}) or {}
    indexing_exprs = dict(getattr(body, "indexing_exprs", {}))
    memo: dict = {}
    out: list[dict] = []
    _walk_block(graph, "root", subblocks, [0], out, indexing_exprs, memo)
    return out


def buffer_roles(leaf: Any) -> dict:
    rw = getattr(leaf, "read_writes", None)
    reads = {d.name for d in rw.reads} if rw else set()
    writes = {d.name for d in rw.writes} if rw else set()
    return {
        "logical_reads": sorted(reads - writes),
        "logical_writes": sorted(writes - reads),
        "in_out": sorted(reads & writes),
    }


def reset_kernel_trace_globals() -> None:
    global _kernel_physical_trace, _last_emitted_trace_json
    _kernel_physical_trace = {}
    _last_emitted_trace_json = None


def _iter_leaves(node_schedule):
    """Yield (leaf, phase). Prefer leaf.is_reduction() when available (robust to marker
    stripping). Fallback: markers are sentinels in schedule; DisableReduction OPENS pointwise,
    EnableReduction CLOSES it (returns to reduction)."""
    from torch._inductor.codegen.simd_kernel_features import DisableReduction, EnableReduction
    for n in node_schedule:
        if n is EnableReduction or n is DisableReduction:
            continue
        # Use leaf's own is_reduction() API if it exists (more robust than markers)
        if hasattr(n, "is_reduction") and callable(n.is_reduction):
            try:
                phase = "reduction" if n.is_reduction() else "pointwise"
            except Exception:
                phase = "pointwise"  # defensive fallback
        else:
            phase = "pointwise"  # non-SchedulerNode leaves (extern/template/foreach)
        yield n, phase


def set_kernel_physical_trace(node_schedule, kernel_name, debug_handle, is_extern=False) -> None:
    if debug_handle is None or config.effective_provenance_tracking_level() == 0:
        return
    try:
        key = f"{kernel_name}:{debug_handle}"
        entry = {"kernel_type": "extern" if is_extern else "triton",
                 "is_extern": is_extern, "leaves": []}
        if is_extern:
            irn = getattr(node_schedule, "node", node_schedule)
            entry["leaves"].append({
                "name": getattr(node_schedule, "get_name", lambda: str(kernel_name))(),
                "scheduler_node_type": type(node_schedule).__name__,
                "extern_target": getattr(getattr(irn, "op_overload", None), "__name__", None)
                                 or type(irn).__name__,
            })
        else:
            for leaf, phase in _iter_leaves(node_schedule):
                if type(leaf).__name__ != "SchedulerNode":
                    # extern/template/foreach-parent leaf: record marker, no ops
                    entry["leaves"].append({
                        "name": leaf.get_name(),
                        "scheduler_node_type": type(leaf).__name__,
                    })
                    continue
                ops = extract_leaf_ops(leaf)
                for op in ops:
                    op["phase"] = phase  # leaf phase from reduction-marker position
                entry["leaves"].append({
                    "name": leaf.get_name(),
                    "scheduler_node_type": "SchedulerNode",
                    "phase": phase,
                    **buffer_roles(leaf),
                    "ops": ops,
                })
        _kernel_physical_trace[key] = entry
    except Exception as e:
        signpost_event("inductor", "provenance_tracking_error",
                       {"function": "set_kernel_physical_trace", "error_msg": str(e)})


def create_triton_kernel_trace_json() -> dict:
    try:
        # pass 1: identity -> set of kernel keys
        ident_kernels: dict = {}
        for key, entry in _kernel_physical_trace.items():
            for lf in entry["leaves"]:
                for op in lf.get("ops", []):
                    if op["target"] in _COMPUTE_OPS:
                        ident_kernels.setdefault(op["identity"], set()).add(key)
        # pass 2: flag + strip internals
        kernels = {}
        for key, entry in _kernel_physical_trace.items():
            leaves = []
            for lf in entry["leaves"]:
                nl = {k: v for k, v in lf.items() if k != "ops"}
                if "ops" in lf:
                    nl["ops"] = [
                        {"order": op["order"], "target": op["target"], "block": op["block"],
                         "phase": op["phase"],
                         "rematerialized": (op["target"] in _COMPUTE_OPS
                                            and len(ident_kernels.get(op["identity"], ())) > 1)}
                        for op in lf["ops"]
                    ]
                leaves.append(nl)
            kernels[key] = {**{k: v for k, v in entry.items() if k != "leaves"}, "leaves": leaves}
        return {"version": 1, "stability": "experimental", "kernels": kernels}
    except Exception as e:
        signpost_event("inductor", "provenance_tracking_error",
                       {"function": "create_triton_kernel_trace_json", "error_msg": str(e)})
        return {}


def set_last_emitted_trace(json_str: str) -> None:
    """Record the last emitted serialized trace for profiler consumption."""
    global _last_emitted_trace_json
    _last_emitted_trace_json = json_str


def get_last_emitted_trace() -> str | None:
    """Retrieve the last emitted serialized trace (set by both cache-miss and cache-hit emit)."""
    return _last_emitted_trace_json


def kernel_ops_summary(trace_json: dict) -> dict:
    """Bare-kernel-name -> compact op summary for profiler event enrichment."""
    out: dict = {}
    for key, entry in trace_json.get("kernels", {}).items():
        bare = key.rsplit(":", 1)[0]
        if bare in out:
            continue  # first-wins on duplicate bare names (documented limitation)
        ops, remat = [], []
        for lf in entry.get("leaves", []):
            for op in lf.get("ops", []):
                target = op.get("target")
                if target is None:
                    continue
                ops.append(target)
                if op.get("rematerialized") and target not in remat:
                    remat.append(target)
        out[bare] = {"ops": ops, "rematerialized": remat}
    return out
