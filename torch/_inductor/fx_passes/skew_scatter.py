# mypy: allow-untyped-defs
"""
Rewrites 2-D row scatter-adds (``index_add``, ``index_put`` with accumulate,
``scatter_add`` with a row-expanded index) on CUDA into kernel/skew_scatter.py.
Only scatters whose ``src`` and index are already materialized are rewritten, so
the pass never un-fuses a producer that Inductor's scatter computes in place.
"""

import contextlib
import itertools
import logging
import math
import operator

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters, detect_fake_mode
from torch._inductor import config
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.virtualized import V
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import optimization_hint


log = logging.getLogger(__name__)
aten = torch.ops.aten


# One autotuning site per rewritten call site in this process.
_sites = itertools.count()


def _scatter_rows(
    index: torch.Tensor, src: torch.Tensor, rows: int, out_dtype: torch.dtype, site: int
) -> torch.Tensor:
    from torch._inductor.kernel.skew_scatter import scatter_rows

    return scatter_rows(index, src, rows, out_dtype, site)


def _index_add(
    base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int
) -> torch.Tensor:
    from torch._inductor.kernel.skew_scatter import skew_scatter_add_

    return skew_scatter_add_(base.clone(), index, src, site)


# For dynamic shapes, where the kernels' launch parameters are not static.
@torch.library.custom_op("inductor::skew_scatter_rows", mutates_args=())
def skew_scatter_rows(
    index: torch.Tensor, src: torch.Tensor, rows: int, out_dtype: torch.dtype, site: int
) -> torch.Tensor:
    return _scatter_rows(index, src, rows, out_dtype, site)


@skew_scatter_rows.register_fake
def _(index, src, rows, out_dtype, site):
    return src.new_empty((rows, src.shape[1]), dtype=out_dtype)


@torch.library.custom_op("inductor::skew_index_add", mutates_args=())
def skew_index_add(
    base: torch.Tensor, index: torch.Tensor, src: torch.Tensor, site: int
) -> torch.Tensor:
    return _index_add(base, index, src, site)


@skew_index_add.register_fake
def _(base, index, src, site):
    return torch.empty_like(base)


def _val(n: object) -> torch.Tensor | None:
    v = n.meta.get("val") if isinstance(n, fx.Node) else None
    return v if isinstance(v, torch.Tensor) else None


def _is_zeros(n: object) -> bool:
    if not isinstance(n, fx.Node) or n.op != "call_function":
        return False
    if n.target is aten.full.default:
        return bool(n.args[1] == 0)
    return n.target in (aten.zeros.default, aten.new_zeros.default)


def _strip_cast(src: fx.Node, out_dtype: torch.dtype) -> fx.Node:
    if (
        src.op == "call_function"
        and src.target is torch.ops.prims.convert_element_type.default
        and src.args[1] == out_dtype
    ):
        inner = _val(src.args[0])
        if inner is not None and inner.dtype in (torch.bfloat16, torch.float16):
            return src.args[0]  # type: ignore[return-value]
    return src


def _index_from_expand(n: object) -> fx.Node | None:
    """``index.unsqueeze(-1).expand(N, D)`` -> the 1-D ``index``."""
    if not isinstance(n, fx.Node) or n.target is not aten.expand.default:
        return None
    u = n.args[0]
    if not isinstance(u, fx.Node) or u.target is not aten.unsqueeze.default:
        return None
    if u.args[1] not in (1, -1):
        return None
    v = _val(u.args[0])
    return u.args[0] if v is not None and v.dim() == 1 else None  # type: ignore[return-value]


def _match(n: fx.Node) -> tuple | None:
    """(base, index, src, embedding_rows) of a supported row scatter-add."""
    if n.op != "call_function":
        return None
    if n.target is aten.index_add.default:
        if n.args[1] != 0 or n.kwargs.get("alpha", 1) != 1:
            return None
        return (n.args[0], n.args[2], n.args[3], None)
    if n.target is aten.index_put.default:
        accumulate = n.args[3] if len(n.args) > 3 else n.kwargs.get("accumulate")
        idx = n.args[1]
        if not accumulate or len(idx) != 1 or idx[0] is None:  # type: ignore[arg-type]
            return None
        return (n.args[0], idx[0], n.args[2], None)  # type: ignore[index]
    if n.target is aten.scatter_add.default and n.args[1] == 0:
        idx = _index_from_expand(n.args[2])
        return None if idx is None else (n.args[0], idx, n.args[3], None)
    if (
        config.skew_scatter_embedding_backward
        and n.target is aten.embedding_dense_backward.default
    ):
        _g, _i, num_weights, padding_idx, scale = n.args
        if padding_idx != -1 or scale:
            return None
        return (None, n.args[1], n.args[0], num_weights)
    return None


def _eligible(out: torch.Tensor, index: torch.Tensor) -> bool:
    if out.dim() != 2 or out.device.type != "cuda":
        return False
    if out.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    from torch._inductor.utils import needs_fallback_due_to_atomic_add_limitations

    if needs_fallback_due_to_atomic_add_limitations(out.dtype):
        return False
    if index.dtype not in (torch.int32, torch.int64):
        return False
    hint = optimization_hint(index.numel(), fallback=0)
    return hint >= config.skew_scatter_min_index_size


def _is_static(*vals: torch.Tensor | None) -> bool:
    return all(
        v is not None and all(isinstance(x, int) for x in (*v.shape, *v.stride()))
        for v in vals
    )


_VIEW_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
    aten.slice.Tensor,
    aten.squeeze.dim,
    aten.unsqueeze.default,
    aten.alias.default,
)
_EXTERN_OPS = (
    aten.mm.default,
    aten.addmm.default,
    aten.bmm.default,
    aten.baddbmm.default,
    aten.convolution.default,
    aten._scaled_mm.default,
)


def _materialized(src: fx.Node) -> bool:
    """``src`` (through views) is a graph input or the output of an extern kernel or
    fallback op, so a user-defined Triton kernel does not have to write it out."""
    from torch._inductor.lowering import fallbacks, lowerings

    n: object = src
    while isinstance(n, fx.Node) and n.op == "call_function" and n.target in _VIEW_OPS:
        n = n.args[0]
    if not isinstance(n, fx.Node):
        return False
    if n.op == "placeholder":
        return True
    if n.op != "call_function":
        return False
    if n.target is operator.getitem:
        n = n.args[0]
        if not isinstance(n, fx.Node) or n.op != "call_function":
            return False
    t = n.target
    return (
        t in _EXTERN_OPS
        or t in fallbacks
        or (isinstance(t, torch._ops.OpOverload) and t not in lowerings)
    )


def _left_to_sorted_segment(n: fx.Node, base: object, index: fx.Node) -> bool:
    """The ``index_put`` nodes that fx_passes/sorted_segment_scatter.py rewrites."""
    if not getattr(config, "sorted_segment_scatter_enabled", False):
        return False
    if n.target is not aten.index_put.default:
        return False
    bv, iv = _val(base), _val(index)
    if bv is None or iv is None or iv.dim() != 1 or bv.dim() < 2:
        return False
    if not all(isinstance(x, int) for x in bv.shape):
        return False
    width = math.prod(bv.shape[1:])
    m = optimization_hint(iv.shape[0], fallback=0)
    min_index = getattr(config, "sorted_segment_scatter_min_index_size", 1_000_000)
    return bv.shape[0] <= 65536 and width >= 64 and m >= min_index


def _inline(graph: fx.Graph, n: fx.Node, fn, args: list, consts: tuple) -> fx.Node:
    """Trace ``fn`` functionally and splice it in before ``n``, so its kernels are
    emitted as user-defined Triton kernels (also with cpp_wrapper and AOTInductor)."""
    fake_args = [a.meta["val"] for a in args]
    fake_mode = detect_fake_mode(fake_args) or contextlib.nullcontext()
    with fake_mode, enable_python_dispatcher():
        gm = make_fx(
            functionalize(lambda *a: fn(*a, *consts), remove="mutations"),
            select_decomp_table(),
            tracing_mode="real",
        )(*fake_args)
    placeholders = gm.graph.find_nodes(op="placeholder")
    with graph.inserting_before(n):
        out = graph.graph_copy(gm.graph, dict(zip(placeholders, args)))
    if not isinstance(out, fx.Node):
        raise AssertionError(f"expected one output node, got {out}")
    return out


def skew_scatter_pass(graph: fx.Graph) -> None:
    if (
        not config.skew_scatter_enabled
        or torch.version.hip is not None
        or torch.are_deterministic_algorithms_enabled()
    ):
        return
    from torch._inductor.kernel.skew_scatter import rewrite_supported

    count = 0
    # Autotuning needs a first call on real data.
    autotune = not (
        V.aot_compilation
        or config.cpp_wrapper
        or config.triton.autotune_at_compile_time
    )
    for n in list(graph.nodes):
        m = _match(n)
        if m is None:
            continue
        base, index, src, emb_rows = m
        out_v, idx_v, src_v = _val(n), _val(index), _val(src)
        if out_v is None or idx_v is None or src_v is None:
            continue
        if not src_v.is_floating_point() or not _eligible(out_v, idx_v):
            continue
        if _left_to_sorted_segment(n, base, index):
            continue
        rows, width = (optimization_hint(x, fallback=0) for x in out_v.shape)
        updates = optimization_hint(idx_v.numel(), fallback=0)
        if not rewrite_supported(
            out_v.device, out_v.dtype, rows, width, updates, autotune
        ):
            continue
        if emb_rows is not None:
            d = out_v.shape[1]
            with graph.inserting_before(n):
                index = graph.call_function(aten.view.default, (index, [-1]))
                src = graph.call_function(aten.view.default, (src, [-1, d]))
            index.meta["val"] = idx_v.view(-1)
            src.meta["val"] = src_v.view(-1, d)
            fn, args, consts = _scatter_rows, [index, src], (emb_rows, out_v.dtype)
        else:
            src = _strip_cast(src, out_v.dtype)
            if not (_materialized(src) and _materialized(index)):
                continue
            if _is_zeros(base):
                fn, args = _scatter_rows, [index, src]
                consts = (out_v.shape[0], out_v.dtype)
            else:
                fn, args, consts = _index_add, [base, index, src], ()
        consts = (*consts, next(_sites) if autotune else -1)
        if _is_static(out_v, *(_val(a) for a in args)):
            new = _inline(graph, n, fn, args, consts)
        elif V.aot_compilation:
            continue
        else:
            op = skew_scatter_rows if fn is _scatter_rows else skew_index_add
            with graph.inserting_before(n):
                new = graph.call_function(op._opoverload, (*args, *consts))
            new.meta.update(n.meta)
        n.replace_all_uses_with(new)
        graph.erase_node(n)
        count += 1
    if count:
        graph.eliminate_dead_code()
        counters["inductor"]["skew_scatter_applied"] += count
        log.info("skew_scatter: %d row scatter-adds rewritten", count)
