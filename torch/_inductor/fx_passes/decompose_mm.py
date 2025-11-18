import dataclasses
import operator

import sympy

import torch
from torch._inductor.fx_passes.bucketing import _insert_fn_trace_before_node

from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternExpr,
)


aten = torch.ops.aten
c10d = torch.ops._c10d_functional


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


def _mm_split_cat_rs_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    num_chunks: int,
    orig_out: torch.Tensor,
    reduce_op: str,
    group_size: int,
    group_name: str,
    scatter_dim: int,
    orig_split_num_chunks: int,
    post_mm_ops,
):
    mm_i_a_args = [a] * num_chunks
    mm_i_b_args = [b] * num_chunks
    if scatter_dim == 0:
        b_chunks = b.chunk(num_chunks, dim=-1)
        mm_i_b_args = b_chunks
    else:
        a_flat = a.flatten(0, -2)
        a_flat_chunks = a_flat.chunk(num_chunks)
        mm_i_a_args = a_flat_chunks

    # TODO: add reduce scatter into tensor and remove last cat
    # out = torch.empty_strided(
    #     size=orig_out.shape,
    #     stride=orig_out.stride(),
    #     dtype=orig_out.dtype,
    #     device=orig_out.device,
    # )
    # out_flat = out.flatten(0, -2)
    # out_flat_chunks = out_flat.chunk(num_chunks)

    rs_outs = []
    for a_i, b_i in zip(mm_i_a_args, mm_i_b_args):
        mm_i = torch.ops.aten.mm(a_i, b_i)
        for op in post_mm_ops:
            mm_i = op.apply(mm_i, num_chunks)
        if scatter_dim != 0:
            mm_i_chunks = mm_i.chunk(orig_split_num_chunks)
            cat = torch.cat(mm_i_chunks, dim=scatter_dim)
            mm_i = cat
        w = torch.ops._c10d_functional.reduce_scatter_tensor(
            mm_i, reduce_op, group_size, group_name
        )
        rs_out_i = torch.ops._c10d_functional.wait_tensor(w)
        rs_outs.append(rs_out_i)

    rs_out = torch.cat(rs_outs)
    return (rs_out,)


def _mm_pw_rs_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    num_chunks: int,
    orig_out: torch.Tensor,
    reduce_op: str,
    group_size: int,
    group_name: str,
    scatter_dim: int,
    orig_split_num_chunks: int,
    post_mm_ops,
):
    a_flat = a.flatten(0, -2)
    a_flat_chunks = a_flat.chunk(num_chunks)

    # TODO: add reduce scatter into tensor and remove last cat
    # out = torch.empty_strided(
    #     size=orig_out.shape,
    #     stride=orig_out.stride(),
    #     dtype=orig_out.dtype,
    #     device=orig_out.device,
    # )
    # out_flat = out.flatten(0, -2)
    # out_flat_chunks = out_flat.chunk(num_chunks)

    rs_outs = []
    for i in range(num_chunks):
        mm_i = torch.ops.aten.mm(a_flat_chunks[i], b)
        for op in post_mm_ops:
            mm_i = op.apply(mm_i, num_chunks)
        mm_i_chunks = mm_i.chunk(orig_split_num_chunks)
        cat = torch.cat(mm_i_chunks, dim=scatter_dim)
        w = torch.ops._c10d_functional.reduce_scatter_tensor(
            cat, reduce_op, group_size, group_name
        )
        rs_out_i = torch.ops._c10d_functional.wait_tensor(w)
        rs_outs.append(rs_out_i)

    rs_out = torch.cat(rs_outs)
    return (rs_out,)


def _size_hint(s: sympy.Expr) -> int:
    from torch.fx.experimental.symbolic_shapes import size_hint

    hint = size_hint(s)
    if hint is not None:
        return hint
    return 0


def _is_contiguous(t) -> bool:
    return t.is_contiguous(memory_format=torch.contiguous_format)


def split_mms(gm: torch.fx.GraphModule, min_m_size: int, num_chunks: int = 2):
    g = gm.graph
    g.owning_module

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


@dataclasses.dataclass
class _ReduceScatterMatch:
    match: Match
    input_node: torch.fx.Node
    reduce_scatter_node: torch.fx.Node
    wait_tensor_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_size: int
    group_name: str


def split_mm_rs(gm: torch.fx.GraphModule, min_m_size: int, num_chunks: int = 2):
    g = gm.graph

    def reduce_scatter_template(inp: PatternExpr, users: int):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                inp,
                KeywordArg("reduce_op"),
                KeywordArg("group_size"),
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

    zero_dim_reduce_scatter_pattern_single_user = reduce_scatter_template(
        KeywordArg("input"), users=1
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
                        group_size=match.kwargs["group_size"],
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := zero_dim_reduce_scatter_pattern_single_user.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        reduce_scatter_node=match.nodes[0],
                        wait_tensor_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=0,
                        group_size=match.kwargs["group_size"],
                        group_name=match.kwargs["group_name"],
                    )
                )

    reduce_scatters = list(reversed(reduce_scatters))
    for reduce_scatter in reduce_scatters:
        process_matmul_reduce_scatter(g, reduce_scatter, min_m_size, num_chunks)


@dataclasses.dataclass
class _PostMMOp:
    n: torch.fx.Node

    def apply(self, t, num_chunks):
        return self.n.target(t, *self.n.args[1:], **self.n.kwargs)


@dataclasses.dataclass
class _ViewOp(_PostMMOp):
    chunk_dim: int

    def apply(self, t, num_chunks):
        shape = [s for s in self.n.args[1]]
        if shape[self.chunk_dim] != -1:
            shape[self.chunk_dim] //= num_chunks
        return self.n.target(t, shape)


def process_matmul_reduce_scatter(
    g, reduce_scatter: _ReduceScatterMatch, min_size_after_split, num_chunks
) -> None:
    (
        input_node,
        _reduce_scatter_node,
        rs_wait_tensor_node,
        reduce_op,
        orig_scatter_dim,
        group_size,
        group_name,
    ) = (
        reduce_scatter.input_node,
        reduce_scatter.reduce_scatter_node,
        reduce_scatter.wait_tensor_node,
        reduce_scatter.reduce_op,
        reduce_scatter.scatter_dim,
        reduce_scatter.group_size,
        reduce_scatter.group_name,
    )
    if len(input_node.users) != 1:
        return

    # find matmul node
    mm_n = None
    ns = []
    n = input_node
    while len(n.all_input_nodes) == 1:
        # accumulate line
        ns.append(n)
        n = n.all_input_nodes[0]
        if n.target == aten.mm.default:
            mm_n = n
            break
    if mm_n is None:
        return

    # Support for now pointwise ops that does not change shape
    # or view 2D -> 3D
    post_mm_ops = []

    def _is_pointwise(n):
        return n.target in (torch.ops.prims.convert_element_type.default,)

    for n in reversed(ns):
        if n.target in (aten.view.default, aten.reshape.default):
            in_shape = n.args[0].meta["val"].shape
            out_shape = n.args[1]
            if in_shape[-1] != out_shape[-1]:
                return
            post_mm_ops.append(_ViewOp(n, chunk_dim=1))
            continue

        if not _is_pointwise(n):
            return
        post_mm_ops.append(_PostMMOp(n))

    arg_a = mm_n.args[0]
    arg_b = mm_n.args[1]
    arg_a_t = arg_a.meta["val"]
    arg_b_t = arg_b.meta["val"]

    size_after_split = 1
    cont_check_t = arg_a_t
    if orig_scatter_dim == 0:
        # split B column wise
        size_after_split *= _size_hint(arg_b_t.shape[-1])
        cont_check_t = arg_b_t
    else:
        # split A row wise
        for s in arg_a_t.shape[:-1]:
            size_after_split *= _size_hint(s)

    if size_after_split < min_size_after_split:
        return

    rs_out_t = rs_wait_tensor_node.meta["val"]
    if not (_is_contiguous(cont_check_t) and _is_contiguous(rs_out_t)):
        return

    orig_split_num_chunks = -1
    if orig_scatter_dim != 0:
        orig_split_num_chunks = len(_reduce_scatter_node.args[0].args[0])

    trace_args = (
        arg_a_t,
        arg_b_t,
        num_chunks,
        rs_out_t,
        reduce_op,
        group_size,
        group_name,
        orig_scatter_dim,
        orig_split_num_chunks,
        post_mm_ops,
    )
    _insert_fn_trace_before_node(
        g,
        _mm_split_cat_rs_fn,
        trace_args,
        mm_n,  # insert before
        [arg_a, arg_b],
        [rs_wait_tensor_node],
    )
    # Cleanup previous nodes
    for n in reversed(reduce_scatter.match.nodes):
        if len(n.users) == 0:
            g.erase_node(n)
    for n in reversed(ns):
        assert len(n.users) == 0
        g.erase_node(n)
    assert len(mm_n.users) == 0
    g.erase_node(mm_n)
