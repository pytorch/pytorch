import dataclasses
import operator

import sympy

import torch
from torch._inductor.fx_passes.bucketing import _insert_fn_trace_before_node
from torch._logging import trace_structured
from torch.fx.experimental.symbolic_shapes import size_hint

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


def _trace_fn_split_mm_a(
    a: torch.Tensor, b: torch.Tensor, num_chunks: int, orig_out: torch.Tensor
):
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


def _trace_fn_mm_rs_scatter_dim0(
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
    assert isinstance(num_chunks, int)
    b_chunks = b.chunk(num_chunks, dim=-1)
    rs_outs = []
    for i in range(num_chunks):
        mm_i = torch.ops.aten.mm(a, b_chunks[i])
        for op in post_mm_ops:
            mm_i = op.apply(mm_i, num_chunks)
        w = torch.ops._c10d_functional.reduce_scatter_tensor(
            mm_i, reduce_op, group_size, group_name
        )
        rs_out_i = torch.ops._c10d_functional.wait_tensor(w)
        rs_outs.append(rs_out_i)

    # B column wise split
    rs_out = torch.cat(rs_outs, dim=-1)
    return (rs_out,)


def _trace_fn_mm_rs_scatter_dim_non0(
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
    assert isinstance(num_chunks, int)
    a_flat = a.flatten(0, -2)
    a_flat_chunks = a_flat.chunk(num_chunks)

    out = torch.empty_strided(
        size=orig_out.shape,
        stride=orig_out.stride(),
        dtype=orig_out.dtype,
        device=orig_out.device,
    )

    out_chunks = out.chunk(num_chunks)
    for i in range(num_chunks):
        mm_i = torch.ops.aten.mm(a_flat_chunks[i], b)
        for op in post_mm_ops:
            mm_i = op.apply(mm_i, num_chunks)
        mm_i_chunks = mm_i.chunk(orig_split_num_chunks, dim=scatter_dim)
        cat = torch.cat(mm_i_chunks)
        w = torch.ops._c10d_functional.reduce_scatter_tensor_out(
            cat, reduce_op, group_size, group_name, out=out_chunks[i]
        )
        torch.ops._c10d_functional.wait_tensor(w)

    return (out,)


def _mm_split_cat_rs_fn_view3d(
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
    assert isinstance(num_chunks, int)
    a_flat = a.flatten(0, -2)
    a_flat_chunks = a_flat.chunk(num_chunks)
    out = torch.empty_strided(
        size=orig_out.shape,
        stride=orig_out.stride(),
        dtype=orig_out.dtype,
        device=orig_out.device,
    )
    out_chunks = out.chunk(num_chunks)

    for i in range(num_chunks):
        mm_i = torch.ops.aten.mm(a_flat_chunks[i], b)
        for op in post_mm_ops:
            mm_i = op.apply(mm_i, num_chunks)

        mm_i_chunks = mm_i.chunk(orig_split_num_chunks, dim=scatter_dim)
        cat = torch.cat(mm_i_chunks)

        w = torch.ops._c10d_functional.reduce_scatter_tensor_out(
            cat, reduce_op, group_size, group_name, out=out_chunks[i]
        )
        torch.ops._c10d_functional.wait_tensor(w)

    return (out,)


def _size_hint(s: sympy.Expr) -> int:
    hint = size_hint(s)
    if hint is not None:
        return hint
    return 0


def _is_contiguous(t) -> bool:
    return t.is_contiguous(memory_format=torch.contiguous_format)


def _select_optimal_chunk_count(
    size_before_split: int,
    num_chunks: list[int],
    min_size_after_split: int,
    additional_check=None,
):
    """Pick the largest chunk count that satisfies minimum size constraint.

    Returns the greatest number from num_chunks such that
    size_before_split // chunk_count >= min_size_after_split.

    Args:
        size_before_split: Total size to be split
        num_chunks: List of candidate chunk counts
        min_size_after_split: Minimum size each chunk must have
        additional_check: Optional callable(chunk_count) -> bool for extra validation
    """
    sorted_num_chunks = sorted(num_chunks)
    for n in reversed(sorted_num_chunks):
        if additional_check is not None and not additional_check(n):
            continue
        if size_before_split // n >= min_size_after_split:
            return n
    return None


def _find_matmul_node(input_node):
    """Walk backwards from input_node to find the matmul node.

    Traverses the graph backwards through nodes with a single input,
    collecting intermediate nodes until a matmul operation is found.

    Args:
        input_node: Starting node to walk backwards from

    Returns:
        Tuple of (matmul_node, intermediate_nodes) where:
        - matmul_node: The aten.mm.default node, or None if not found
        - intermediate_nodes: List of nodes between input_node and matmul_node
    """
    intermediate_nodes = []
    node = input_node
    while len(node.all_input_nodes) == 1:
        intermediate_nodes.append(node)
        node = node.all_input_nodes[0]
        if node.target == aten.mm.default:
            return node, intermediate_nodes
    return None, []


def _extract_post_mm_ops(intermediate_nodes):
    """Extract and validate operations between matmul and reduce-scatter.

    Validates that only supported operations appear between the matmul and
    reduce-scatter: pointwise ops (e.g., dtype conversions) and at most one
    2D->3D reshape/view operation.

    Args:
        intermediate_nodes: List of nodes between matmul and reduce-scatter

    Returns:
        Tuple of (post_mm_ops, view3d_node) where:
        - post_mm_ops: List of _PostMMOp or _ViewOp to apply after each matmul chunk
        - view3d_node: The 2D->3D view node if present, else None
        Returns (None, None) if validation fails
    """

    def _is_pointwise(n):
        return n.target in (torch.ops.prims.convert_element_type.default,)

    post_mm_ops = []
    reshape_ops_count = 0
    view3d_node = None

    for node in reversed(intermediate_nodes):
        if node.target in (aten.view.default, aten.reshape.default):
            in_shape = node.args[0].meta["val"].shape
            out_shape = node.args[1]

            # Support only 2D -> 3D view with matching last dimension
            if not (
                len(in_shape) == 2
                and len(out_shape) == 3
                and in_shape[-1] == out_shape[-1]
            ):
                return None, None

            # Support only one 2D -> 3D reshape
            reshape_ops_count += 1
            if reshape_ops_count > 1:
                return None, None

            post_mm_ops.append(_ViewOp(node, chunk_dim=0))
            view3d_node = node
            continue

        if not _is_pointwise(node):
            return None, None

        post_mm_ops.append(_PostMMOp(node))

    return post_mm_ops, view3d_node


def split_mm(
    gm: torch.fx.GraphModule, min_size_after_split: int, num_chunks: list[int]
):
    g = gm.graph
    g.owning_module

    for n in g.nodes:
        if not (n.op == "call_function" and n.target == torch.ops.aten.mm.default):
            continue

        mm_n = n
        arg_a = mm_n.args[0]
        arg_b = mm_n.args[1]

        a_t = arg_a.meta["val"]

        size_before_split = 1
        for s in a_t.shape[:-1]:
            size_before_split *= _size_hint(s)

        n_split = _select_optimal_chunk_count(
            size_before_split, num_chunks, min_size_after_split
        )

        if n_split is None:
            return

        arg_a_t = arg_a.meta["val"]
        arg_b_t = arg_b.meta["val"]
        mm_out_t = mm_n.meta["val"]

        # Decompose only to contiguous chunks
        if not (_is_contiguous(arg_a_t) and _is_contiguous(mm_out_t)):
            continue

        trace_args = (arg_a_t, arg_b_t, n_split, mm_out_t)
        _insert_fn_trace_before_node(
            g,
            _trace_fn_split_mm_a,
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


def _split_mm_rs(
    gm: torch.fx.GraphModule,
    num_chunks: list[int],
    min_size_after_split: int = 2048,
):
    """
    Splits matmul->reduce_scatter into chunks,
    number of chunks will be the greatest value from num_chunks,
    with condition that matmuls dimension after split is larger than min_size_after_split.

    The main motivation usecase is overlap scheduling pass. Splitting large matmuls helps
    more fine grained overlapping.
    """
    g = gm.graph
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pass_split_mm_rs_graph_before",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

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

    for reduce_scatter in reversed(reduce_scatters):
        _process_matmul_reduce_scatter(
            g, reduce_scatter, min_size_after_split, num_chunks
        )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pass_split_mm_rs_graph_after",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )


@dataclasses.dataclass
class _PostMMOp:
    n: torch.fx.Node

    def apply(self, t, num_chunks):
        return self.n.target(  # pyrefly: ignore[not-callable]
            t, *self.n.args[1:], **self.n.kwargs
        )


@dataclasses.dataclass
class _ViewOp(_PostMMOp):
    chunk_dim: int

    def apply(self, t, num_chunks):
        shape = list(self.n.args[1])  # pyrefly: ignore[no-matching-overload]
        if shape[self.chunk_dim] != -1:
            shape[self.chunk_dim] //= num_chunks
        return self.n.target(t, shape)  # pyrefly: ignore[not-callable]


def _process_matmul_reduce_scatter(
    g, reduce_scatter: _ReduceScatterMatch, min_size_after_split, num_chunks: list[int]
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

    # Find matmul node by walking backwards through the graph
    mm_n, ns = _find_matmul_node(input_node)
    if mm_n is None:
        return

    # Validate and extract operations between matmul and reduce-scatter
    post_mm_ops, view3d_n = _extract_post_mm_ops(ns)
    if post_mm_ops is None:
        return

    arg_a = mm_n.args[0]
    arg_b = mm_n.args[1]
    arg_a_t = arg_a.meta["val"]
    arg_b_t = arg_b.meta["val"]

    _fn_to_trace = (
        _trace_fn_mm_rs_scatter_dim0
        if orig_scatter_dim == 0
        else _trace_fn_mm_rs_scatter_dim_non0
    )

    view3d_size_to_split = None
    if view3d_n is not None:
        _fn_to_trace = _mm_split_cat_rs_fn_view3d
        # dim0 of 3d reshape
        view3d_size_to_split = view3d_n.args[1][0]

    size_before_split = 1
    cont_check_t = arg_a_t
    if orig_scatter_dim == 0:
        # split B column wise
        size_before_split *= _size_hint(arg_b_t.shape[-1])
        cont_check_t = arg_b_t
    else:
        # split A row wise
        for s in arg_a_t.shape[:-1]:
            size_before_split *= _size_hint(s)

    # Additional checks: divisibility and view3d constraint
    additional_check = None
    if view3d_size_to_split is not None:

        def _additional_check(n):
            return view3d_size_to_split % n == 0

        additional_check = _additional_check

    n_split = _select_optimal_chunk_count(
        size_before_split, num_chunks, min_size_after_split, additional_check
    )
    if n_split is None:
        return

    rs_out_t = rs_wait_tensor_node.meta["val"]
    if not (_is_contiguous(cont_check_t) and _is_contiguous(rs_out_t)):
        return

    # TODO: just use group_size?
    orig_split_num_chunks = -1
    if orig_scatter_dim != 0:
        orig_split_num_chunks = len(
            _reduce_scatter_node.args[0].args[0]  # pyrefly: ignore[missing-attribute]
        )
    trace_args = (
        arg_a_t,
        arg_b_t,
        n_split,
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
        _fn_to_trace,
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
