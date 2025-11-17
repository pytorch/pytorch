import operator

import sympy

import torch

from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternExpr,
)


def _fn(a: torch.Tensor, b: torch.Tensor, num_chunks: int, orig_out: torch.Tensor):
    a_flat = a.flatten(0, -2)
    a_flat_chunks = a_flat.chunk(num_chunks)
    out = torch.empty_strided(
        size=orig_out.shape,
        stride=orig_out.stride(),
        dtype=orig_out.dtype,
        device=orig_out.device,
    )
    out_flat = out.flatten(0, -2)
    out_flat_chunks = out_flat.chunk(num_chunks)
    for i in range(num_chunks):
        torch.ops.aten.mm.out(a_flat_chunks[i], b, out=out_flat_chunks[i])
    return (out,)


def split_mms(gm: torch.fx.GraphModule, min_m_size: int, num_chunks: int = 2):
    g = gm.graph
    g.owning_module

    def _size_hint(s: sympy.Expr) -> int:
        from torch.fx.experimental.symbolic_shapes import size_hint

        hint = size_hint(s)
        if hint is not None:
            return hint
        return 0

    def _is_contiguous(t) -> bool:
        return t.is_contiguous(memory_format=torch.contiguous_format)

    for n in g.nodes:
        if n.op == "call_function" and n.target != torch.ops.aten.mm.default:
            continue

        mm_n = n
        arg_a = mm_n.args[0]
        arg_b = mm_n.args[1]

        a_t = arg_a.meta["val"]

        M = 1
        for s in a_t.shape[:-1]:
            M *= _size_hint(s)

        if M < min_m_size:
            continue

        arg_a_t = arg_a.meta["val"]
        arg_b_t = arg_b.meta["val"]
        mm_out_t = mm_n.meta["val"]

        # Decompose only to contiguous chunks
        if not (_is_contiguous(arg_a_t) and _is_contiguous(mm_out_t)):
            continue

        from torch._inductor.fx_passes.bucketing import _insert_fn_trace_before_node

        trace_args = (arg_a_t, arg_b_t, num_chunks, mm_out_t)
        _insert_fn_trace_before_node(
            g,
            _fn,
            trace_args,
            mm_n,  # insert before
            [arg_a, arg_b],
            [mm_n],
        )

        g.erase_node(mm_n)


from torch._inductor.fx_passes.micro_pipeline_tp import _ReduceScatterMatch


def split_mm_split_cat_rs(
    gm: torch.fx.GraphModule, min_m_size: int, num_chunks: int = 2
):
    g = gm.graph
    aten = torch.ops.aten
    c10d = torch.ops._c10d_functional

    def reduce_scatter_template(inp: PatternExpr, users: int):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                inp,
                KeywordArg("reduce_op"),
                Ignored(),
                KeywordArg("group_name"),
                _users=users,
            ),
        )

    # Matches funcol.reduce_scatter_tensor with scatter_dim > 0
    non_zero_dim_reduce_scatter_pattern_single_user = reduce_scatter_template(
        CallFunction(
            aten.cat.default,
            ListOf(
                CallFunction(
                    operator.getitem,
                    CallFunction(
                        aten.split.Tensor,
                        KeywordArg("input"),
                        Ignored(),
                        KeywordArg("scatter_dim"),
                        _users=MULTIPLE,
                    ),
                    Ignored(),
                )
            ),
        ),
        users=1,
    )

    reduce_scatters = []
    for node in reversed(g.nodes):
        if node.target == c10d.wait_tensor.default:
            if match := non_zero_dim_reduce_scatter_pattern_single_user.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[-2],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=match.kwargs["scatter_dim"],
                        group_name=match.kwargs["group_name"],
                    )
                )
    reduce_scatters = list(reversed(reduce_scatters))
    for reduce_scatter in reduce_scatters:
        process_matmul_reduce_scatter(reduce_scatter)


def process_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    (
        input_node,
        _reduce_scatter_node,
        rs_wait_tensor_node,
        reduce_op,
        orig_scatter_dim,
        group_name,
    ) = (
        reduce_scatter.input_node,
        reduce_scatter.reduce_scatter_node,
        reduce_scatter.wait_tensor_node,
        reduce_scatter.reduce_op,
        reduce_scatter.scatter_dim,
        reduce_scatter.group_name,
    )
    print(f"XXX process_matmul_reduce_scatter:{reduce_scatter}")
