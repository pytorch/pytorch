# mypy: allow-untyped-defs
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import torch
import torch.fx.node
from torch._C._dynamo.guards import compute_overlapping_tensors
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters, ReinplaceCounters, ReInplaceTrigger
from torch._guards import detect_fake_mode
from torch._higher_order_ops.user_kernel_hop import (
    get_user_kernel_hop_info,
    is_user_kernel_functional_hop,
)
from torch._inductor import config, inductor_prims
from torch._inductor.fx_utils import (
    _is_unsupported_inductor_lowering,
    _same_size_stride_and_storage_offset,
    FakeTensorUpdater,
    get_mutated_input_nodes,
    get_mutated_storages,
    get_node_storage,
    get_storage,
    is_node_realized,
    same_tensor_meta,
)
from torch._inductor.lowering import (
    inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings,
)
from torch._inductor.virtualized import V
from torch._prims_common import is_contiguous_or_false
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    GuardOnDataDependentSymNode,
    statically_known_true,
)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.reinplace import _is_view_op
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]
    mutated_arg: int | tuple[int, ...]  # Single index or tuple of indices
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True

    @property
    def mutated_args(self) -> tuple[int, ...]:
        """Return mutated_arg as a tuple for uniform handling."""
        if isinstance(self.mutated_arg, int):
            return (self.mutated_arg,)
        return self.mutated_arg


_SCATTER_OP_TO_VIEW = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}
_VIEW_OP_TO_SCATTER = {v: k for k, v in _SCATTER_OP_TO_VIEW.items()}


def graph_call_function(graph: torch.fx.Graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
        (args, kwargs),
    )
    with V.fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)

    node = graph.call_function(fn, args, kwargs)

    node.meta["val"] = fake_result
    if (V.fake_mode.shape_env) and (
        symbol_to_path := compute_unbacked_bindings(V.fake_mode.shape_env, fake_result)
    ):
        node.meta["unbacked_bindings"] = symbol_to_path

    return node


@dataclass
class ViewOp:
    target: torch._ops.OpOverload
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _inplace_generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    tmp = inp
    for view in view_ops:
        fake_args, fake_kwargs = pytree.tree_map(
            lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
            (view.args, view.kwargs),
        )
        # slice and select can allocate new unbacked symints, but those won't be reflected
        # in the output of this function, hence shall be ignored.
        fake_mode = detect_fake_mode(fake_args)
        with (
            fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            if fake_mode and fake_mode.shape_env
            else nullcontext()
        ):
            tmp = view.target(tmp, *fake_args, **fake_kwargs)
    try:
        tmp.copy_(src)
    except RuntimeError as e:
        raise RuntimeError(
            f"shape error in scatter op, can not broadcast {src.shape} to {tmp.shape}"
        ) from e
    return inp


def _generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    out = inp.clone()
    return _inplace_generalized_scatter(out, src, view_ops)


def _decompose_scatter_functional_helper(
    graph: torch.fx.Graph,
    inp: torch.Tensor,
    src: torch.Tensor,
    view_ops: list[ViewOp],
) -> torch.fx.Node:
    view_op, view_ops_tail = view_ops[0], view_ops[1:]

    if view_ops_tail:
        view = graph_call_function(
            graph, view_op.target, inp, *view_op.args, **view_op.kwargs
        )
        src = _decompose_scatter_functional_helper(graph, view, src, view_ops[1:])  # type: ignore[assignment]

    return graph_call_function(
        graph,
        _VIEW_OP_TO_SCATTER[view_op.target],
        inp,
        src,
        *view_op.args,
        **view_op.kwargs,
    )


def _decompose_scatter_functional(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """Decompose _generalized_scatter to a sequence of view_scatter operations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    view = aten.slice(inp, 0, 0, 10)
    view_updated = aten.slice_scatter(view, src, 1, 10, -10)
    inp_updated = aten.slice_scatter(inp, view_updated, 0, 0, 10)
    """
    if node.target is not _generalized_scatter:
        raise AssertionError(
            f"expected node.target to be _generalized_scatter, got {node.target}"
        )
    return _decompose_scatter_functional_helper(graph, *node.args)  # type: ignore[arg-type]


def _decompose_scatter_mutating(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """Decompose _generalized_scatter using mutations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    inp_updated = aten.clone(inp)
    slice1 = aten.slice(inp_updated, 0, 0, 10)
    slice2 = aten.slice(slice1, 1, 10, -10)
    slice2.copy_(src)

    """
    if node.target not in (_generalized_scatter, _inplace_generalized_scatter):
        raise AssertionError(f"unexpected node.target: {node.target}")
    inp, src, view_ops = node.args
    if node.kwargs:
        raise AssertionError(f"expected no kwargs, got {node.kwargs}")

    if node.target is _generalized_scatter:
        inp = graph_call_function(graph, aten.clone, inp)

    tmp = inp
    for view in view_ops:  # type: ignore[union-attr]
        tmp = graph_call_function(graph, view.target, tmp, *view.args, **view.kwargs)  # type: ignore[union-attr]

    graph_call_function(graph, aten.copy_.default, tmp, src)
    return inp  # type: ignore[return-value]


# View ops whose view_scatter op is lowered into mutations anyway,
# so is never a pessimisation to decompose.
_ALWAYS_MUTATING_SCATTER_OPS = OrderedSet(
    [
        aten.as_strided.default,
        aten.diagonal.default,
    ]
)


def scatter_always_uses_mutation(node: torch.fx.Node) -> bool:
    _, _, view_ops = node.args
    view_ops = cast(Sequence[torch.fx.node.Argument], view_ops)
    return any(
        target in _ALWAYS_MUTATING_SCATTER_OPS
        for view in view_ops
        if isinstance(target := getattr(view, "target", None), torch._ops.OpOverload)
    )


def should_reinplace_index_put(node: torch.fx.Node) -> bool:
    """
    index_put_(inp, indices, values) reads indices and values while writing inp,
    so it can't be used if either of them shares inp's storage. For example,
    torch.put(x, x, x) decomposes to index_put(x.flatten(), [x], x), and
    reinplaced into index_put_ the op would read the x it is writing.
    """
    inp = node.args[0]
    if not isinstance(inp, torch.fx.Node):
        return True
    inp_storage = get_node_storage(inp)
    if inp_storage is None:
        return True
    return not any(
        isinstance(arg, torch.fx.Node) and get_node_storage(arg) == inp_storage
        for arg in pytree.tree_leaves((node.args[1:], node.kwargs))
    )


def annotate_scatter_chain_lengths(graph: torch.fx.Graph) -> None:
    """
    Record on each _generalized_scatter node the length of the chain it is in.

    Consider a generalized_scatter chain in the following format:
        gs(gs(gs(base, s0, ...), s1, ...), s2, ...)
    Every generalized_scatter op is given the same length of 3 and is stored in the
    "scatter_chain_length" field in the node's metadata. Note that only single-user
    links in the example above count.

    When should_reinplace_scatter() runs, it will use this field to make decisions on
    whether to preserve the chained scatter (functional form) or reinplace the chain into
    a mutate in-place form that costs more kernel launches but is more memory bw efficient
    per threadblock.
    """
    for head in graph.find_nodes(op="call_function", target=_generalized_scatter):
        if "scatter_chain_length" in head.meta:
            continue
        chain = [head]
        tail = head
        while len(tail.users) == 1:
            (user,) = tail.users
            if user.target is not _generalized_scatter or user.args[0] is not tail:
                break
            chain.append(user)
            tail = user
        for node in chain:
            node.meta["scatter_chain_length"] = len(chain)


# Allocation ops that reach post-grad graphs under torch.compile (empty_like,
# zeros, full_like and friends are decomposed into these). Their result cannot
# alias a graph input.
_FRESH_ALLOCATION_OPS = OrderedSet(
    [
        aten.empty.memory_format,
        aten.empty_strided.default,
        aten.new_empty_strided.default,
        aten.full.default,
    ]
)


# Set on a scatter that should_reinplace_scatter reinplaces only because its
# input is an auto_functionalized_v2 base, so the rewrite can be counted.
_MUTATED_BASE_SCATTER = "reinplace_mutated_base_scatter"


def _updated_base_offset(outputs: Sequence[Any], num_bases: int) -> int:
    """Index of the first updated base among auto_functionalized outputs: the op's
    own outputs come first, then one updated copy per mutated base."""
    return len(outputs) - num_bases


def _auto_functionalized_v2_input_base(node: object) -> torch.fx.Node | None:
    """The _all_bases input that node is the updated copy of, if node is one of
    the updated bases auto_functionalized_v2 returns after its op's own
    outputs, else None."""
    if not (isinstance(node, torch.fx.Node) and node.target is operator.getitem):
        return None
    af, index = node.args
    if not (
        isinstance(af, torch.fx.Node)
        and af.target is torch.ops.higher_order.auto_functionalized_v2
        and isinstance(index, int)
    ):
        return None
    outputs = af.meta.get("val")
    bases = af.kwargs.get("_all_bases", ())
    if not isinstance(outputs, (tuple, list)) or not isinstance(bases, (tuple, list)):
        return None
    base_index = index - _updated_base_offset(outputs, len(bases))
    if base_index < 0:
        return None
    base = bases[base_index]
    return base if isinstance(base, torch.fx.Node) else None


def _is_reinplaced_auto_functionalized_v2_base(node: torch.fx.Node) -> bool:
    """True if an auto_functionalized_v2 that has already been reinplaced writes
    node, one of its _all_bases, in place rather than into a clone."""
    for user in node.users:
        if user.target is not torch.ops.higher_order.auto_functionalized_v2:
            continue
        cloned = user.meta.get("only_clone_these_tensors")
        if cloned is None:
            continue
        bases = user.kwargs.get("_all_bases", ())
        if not isinstance(bases, (tuple, list)):
            continue
        if any(base is node and i not in cloned for i, base in enumerate(bases)):
            return True
    return False


def _is_auto_functionalized_v2_base(node: object) -> bool:
    """True if node is a fresh allocation that auto_functionalized_v2 writes in
    place, seen either as one of its updated bases or as the base itself.

    When reinplacing rewires auto_functionalized_v2 outputs (the default when
    the op is decomposed afterwards), each in-place updated base is replaced
    by its input base before later nodes are visited, so node is the base and
    the op's only_clone_these_tensors metadata shows it is written in place.
    Without rewiring, node is the updated-base getitem, and the base is found
    by following the chain of such getitems and views back to its root.

    Either way the base must be a fresh allocation. If it is a graph input, or
    anything else that may itself be reinplaced into one, the input's epilogue
    copy_ runs after the scatter, and can_inplace() cannot see that.
    """
    if not isinstance(node, torch.fx.Node):
        return False
    base = _auto_functionalized_v2_input_base(node)
    if base is None:
        root = _get_view_base(node)
        return (
            root.op == "call_function"
            and root.target in _FRESH_ALLOCATION_OPS
            and _is_reinplaced_auto_functionalized_v2_base(root)
        )
    root = _get_view_base(base)
    while (base := _auto_functionalized_v2_input_base(root)) is not None:
        root = _get_view_base(base)
    return root.op == "call_function" and root.target in _FRESH_ALLOCATION_OPS


def should_reinplace_scatter(node: torch.fx.Node) -> bool:
    """Choose between mutating and functional scatter decompositions

    Reinplacing view scatter ops can be pessimising as it blocks fusion with the
    input or output tensor computations. However, it is still profitable if the
    input and output would have been realized anyway.
    config.reinplace_whenever_possible overrides profitability, but not safety.
    """
    inp, src, _view_ops = node.args

    # The mutating decomposition is view(inp).copy_(src). If src is itself a
    # view of inp's storage (x[1:] = x[:-1].clone(), once the clone has been
    # removed as a no-op), that copy reads what it is overwriting.
    if isinstance(inp, torch.fx.Node) and isinstance(src, torch.fx.Node):
        inp_storage = get_node_storage(inp)
        if inp_storage is not None and inp_storage == get_node_storage(src):
            return False

    # Mutating scatter ops unconditionally realize input and output
    if scatter_always_uses_mutation(node):
        return True

    if is_node_realized(inp) and is_node_realized(node):  # type: ignore[arg-type]
        return True

    # is_node_realized() does not count the updated bases of
    # auto_functionalized_v2 as buffers, but the op has written them in place,
    # so a functional scatter would copy the whole base to update one region.
    if is_node_realized(node) and _is_auto_functionalized_v2_base(inp):
        node.meta[_MUTATED_BASE_SCATTER] = True
        return True

    # If the output is copied back into the input, this forces both to be
    # realized as the output is a user of the input
    if inp.op in ("placeholder", "get_attr") and any(  # type: ignore[union-attr]
        user.target is aten.copy_.default and user.args[0] is inp for user in node.users
    ):
        return True

    # TODO This can and should be moved to a fxgraph pass upstream
    # A long chain of scatters should NOT be kept in functional variant, as each thread block
    # in functional variant is required to load all input pointers instead of just the one it
    # needs. Therefore, we should reinplace for longer scatter chains.
    chain_length = node.meta.get("scatter_chain_length", 1)
    if (
        config.reinplace_scatter_chain_threshold > 0
        and chain_length >= config.reinplace_scatter_chain_threshold
    ):
        return True

    # Otherwise, prefer functional variants for fusion unless explicitly overridden.
    return config.reinplace_whenever_possible


def should_reinplace_all_reduce(node: torch.fx.Node) -> bool:
    """
    The inplace lowering requires contiguous inputs; other layouts produce a copy.
    The symmetric-memory one-shot all-reduce lowering is out-of-place, so
    reinplacing an all_reduce it may pick only adds a copy back into the input.
    Keep the symmetric-memory gate: the in-place one-shot path marks the reused
    buffer as skip-wait, which can suppress a later collective's wait.
    """
    args = (
        node.args[2:]
        if node.target is torch.ops.higher_order.with_effects
        else node.args
    )
    if not cast(torch.fx.Node, args[0]).meta["val"].is_contiguous():
        return False
    if not config._collective.auto_select:
        return True
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    group_name = cast("torch.distributed.distributed_c10d.GroupName", args[2])
    return not is_symm_mem_enabled_for_group(group_name)


def decompose_generalized_scatter(graph: torch.fx.Graph) -> None:
    """Replace _generalized_scatter with normal aten ops"""
    for node in itertools.chain(
        graph.find_nodes(op="call_function", target=_generalized_scatter),
        graph.find_nodes(op="call_function", target=_inplace_generalized_scatter),
    ):
        use_mutation = (
            node.target is _inplace_generalized_scatter
            or scatter_always_uses_mutation(node)
        )

        prev = node.prev
        with graph.inserting_before(node):
            if use_mutation:
                new_node = _decompose_scatter_mutating(graph, node)
            else:
                new_node = _decompose_scatter_functional(graph, node)

        # Keep user stream/mempool annotations on the decomposed nodes, so they
        # run in the same context as the scatter they replace.
        if custom := node.meta.get("custom"):
            inserted = node.prev
            while inserted is not prev:
                inserted.meta["custom"] = custom.copy()
                inserted = inserted.prev

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def canonicalize_view_scatter_ops(graph: torch.fx.Graph) -> None:
    """
    This canonicalizes view scatter ops into a generalized form, defined as:
      def scatter(inp, src, views):
        tmp = inp.clone()
        for view in views:
          tmp = view(tmp)
        tmp.copy_(src)

    We also fuse consecutive view scatter ops of the form
        a = scatter(view2(self), src, [view1])
        b = scatter(self, a, [view2])
    which can be rewritten as
        b = scatter(self, src, [view2, view1])
        a = view2(b)

    This is both more efficient as we only do a single scatter, and also
    easier to reinplace since there is only one use of `self`
    """

    node_to_view_base: dict[torch.fx.Node, torch.fx.Node] = {}
    node_to_view_op: dict[torch.fx.Node, list[ViewOp]] = defaultdict(list)

    def handle_views(node: torch.fx.Node):
        inp = node.args[0]
        node_to_view_base[node] = node_to_view_base.get(inp, inp)  # type: ignore[arg-type, assignment]
        node_to_view_op[node] = [
            *node_to_view_op[inp],  # type: ignore[index]
            ViewOp(
                node.target,  # type: ignore[arg-type]
                args=node.args[1:],
                kwargs=node.kwargs,
            ),
        ]

    def handle_view_scatter(node: torch.fx.Node):
        if len(node.args) < 2:
            raise AssertionError(f"expected at least 2 args, got {len(node.args)}")
        inp, src = node.args[:2]

        if not isinstance(node.target, torch._ops.OpOverload):
            raise AssertionError(
                f"expected node.target to be an OpOverload, got {type(node.target)}"
            )
        scatter_view_op = ViewOp(
            _SCATTER_OP_TO_VIEW[node.target],
            args=node.args[2:],
            kwargs=node.kwargs,
        )

        def can_fuse():
            if src.target is not _generalized_scatter:  # type: ignore[union-attr]
                return False
            # Fusing would move one scatter into the other's stream/mempool.
            if src.meta.get("custom") != node.meta.get("custom"):  # type: ignore[union-attr]
                return False
            src_inp, _src_src, _src_scatter_view_op = src.args  # type: ignore[union-attr]

            inp_base = node_to_view_base.get(inp, inp)  # type: ignore[arg-type]
            src_base = node_to_view_base.get(src_inp, src_inp)  # type: ignore[arg-type]
            return inp_base is src_base and node_to_view_op[src_inp] == [  # type: ignore[index]
                *node_to_view_op[inp],  # type: ignore[index]
                scatter_view_op,
            ]

        custom = node.meta.get("custom")
        if not can_fuse():
            with graph.inserting_before(node):
                new_node = graph_call_function(
                    graph,
                    _generalized_scatter,
                    inp,
                    src,
                    [scatter_view_op],
                )
            if custom:
                new_node.meta["custom"] = custom.copy()
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            return

        _src_inp, src_src, src_scatter_view_op = src.args  # type: ignore[union-attr]
        with graph.inserting_before(src):  # type: ignore[arg-type]
            new_node = graph_call_function(
                graph,
                _generalized_scatter,
                inp,
                src_src,
                [scatter_view_op, *src_scatter_view_op],  # type: ignore[misc]
            )
            if custom:
                new_node.meta["custom"] = custom.copy()
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

            if src.users:  # type: ignore[union-attr]
                new_src = graph_call_function(
                    graph,
                    _SCATTER_OP_TO_VIEW[node.target],
                    new_node,
                    *node.args[2:],
                    **node.kwargs,
                )
                if custom:
                    new_src.meta["custom"] = custom.copy()

                handle_views(new_src)
                src.replace_all_uses_with(new_src)  # type: ignore[union-attr]

            graph.erase_node(src)  # type: ignore[arg-type]

    for node in graph.nodes:
        if _is_view_op(node.target):
            handle_views(node)
        elif node.target in _SCATTER_OP_TO_VIEW:
            handle_view_scatter(node)


inplaceable_ops: dict[Callable[..., Any], InplaceableOp] = {
    aten._scaled_addmm.default: InplaceableOp(aten._scaled_addmm_.default, 0),
    aten.index_put.default: InplaceableOp(
        aten.index_put_.default, 0, extra_check=should_reinplace_index_put
    ),
    aten._unsafe_index_put.default: InplaceableOp(
        inductor_prims._unsafe_index_put_, 0, extra_check=should_reinplace_index_put
    ),
    _generalized_scatter: InplaceableOp(
        _inplace_generalized_scatter,
        0,
        extra_check=should_reinplace_scatter,
    ),
    # Stateless Philox RNG: reinplace the functionalized clone onto the dead
    # output buffer, so out-of-place uniform()/normal()/bits() don't pay an
    # extra copy.
    aten._philox_uniform.default: InplaceableOp(aten._philox_uniform_.default, 0),
    aten._philox_normal.default: InplaceableOp(aten._philox_normal_.default, 0),
    aten._philox_randint.default: InplaceableOp(aten._philox_randint_.default, 0),
    aten._philox_uniform.Tensor: InplaceableOp(aten._philox_uniform_.Tensor, 0),
}

inplaceable_collective_ops: dict[Callable[..., Any], InplaceableOp] = {}
try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops = {
        c10d_functional.all_reduce.default: InplaceableOp(
            c10d_functional.all_reduce_.default,
            0,
            extra_check=should_reinplace_all_reduce,
        ),
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(
            c10d_functional.all_reduce_coalesced_.default, 0
        ),
    }
    inplaceable_ops.update(inplaceable_collective_ops)
except AttributeError:
    # _c10d_functional ops are only available when torch
    # is built with USE_DISTRIBUTED=1.
    pass

inplaceable_foreach_ops: dict[torch._ops.OpOverload, InplaceableOp] = {}
for outplace_op, inplace_op in inplaceable_foreach_ops_lowerings.items():
    inplaceable_foreach_ops[outplace_op] = InplaceableOp(inplace_op, 0)


# Operators that don't depend on the tensor data
META_ONLY_OPS = OrderedSet(
    [
        aten.sym_size.int,
        aten.sym_stride.int,
        aten.sym_numel.default,
        aten.sym_storage_offset.default,
    ]
)


def _get_view_base(node: torch.fx.Node) -> torch.fx.Node:
    while _is_view_op(node.target) and node.args:
        base = node.args[0]
        if not isinstance(base, torch.fx.Node):
            break
        node = base
    return node


def _is_layout_preserving_view_copy_back(
    dst: torch.fx.Node,
    src: torch.fx.Node,
    mutated_arg: torch.fx.Node,
    src_base: torch.fx.Node,
) -> bool:
    return (
        src.meta["val"].dtype == dst.meta["val"].dtype
        and src_base.meta["val"].is_conj() == mutated_arg.meta["val"].is_conj()
        and src_base.meta["val"].is_neg() == mutated_arg.meta["val"].is_neg()
        and src.meta["val"].is_conj() == dst.meta["val"].is_conj()
        and src.meta["val"].is_neg() == dst.meta["val"].is_neg()
        and _same_size_stride_and_storage_offset(
            cast(torch.Tensor, src_base.meta["val"]),
            cast(torch.Tensor, mutated_arg.meta["val"]),
        )
        and _same_size_stride_and_storage_offset(
            cast(torch.Tensor, src.meta["val"]),
            cast(torch.Tensor, dst.meta["val"]),
        )
    )


def _is_control_deps_ordering_only_use(
    user: torch.fx.Node, view: torch.fx.Node
) -> bool:
    """Check if view appears only in control_deps' ordering-only additional_deps tuple.

    control_deps schema (control_dependencies.py): control_deps(additional_deps, subgraph, *args)
    kwargs is always empty -- preserve_node_ordering threads all nodes positionally.
    """
    if not (
        user.op == "call_function"
        and isinstance(user.target, torch._ops.HigherOrderOperator)
        and user.target._name == "control_deps"
    ):
        return False
    additional_deps = cast(tuple[torch.fx.Node, ...], user.args[0])
    pass_through = cast(tuple[torch.fx.Node, ...], user.args[2:])
    return view in additional_deps and view not in pass_through


def _has_disjoint_strides(tensor: torch.Tensor) -> bool:
    # Each stride must step past the span of all faster dimensions.
    dims: list[tuple[int, int | torch.SymInt]] = []
    for size, stride in zip(tensor.shape, tensor.stride()):
        if statically_known_true(size <= 1):
            continue
        if not isinstance(stride, int):
            return False
        dims.append((stride, size))

    span: int | torch.SymInt = 1
    for stride, size in sorted(dims, key=operator.itemgetter(0)):
        if not statically_known_true(stride >= span):
            return False
        span += (size - 1) * stride
    return True


def reinplace_inplaceable_ops_core(
    graph: torch.fx.Graph,
    *,
    fake_tensor_updater: FakeTensorUpdater | None = None,
    rewire_auto_functionalized_outputs: bool = False,
) -> None:
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    Graph inputs are also observable by the caller, so they require a later
    copy_ into the input. A candidate-specific copyback preserves the requested
    mutation. For input views, a full-storage overwrite can instead hide the
    earlier mutation, provided the input is not read before the overwrite and
    no result made to alias it remains live across the overwrite. Permission
    to reuse input storage does not by itself permit removing that copy_.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.

    Without an updater, graphs without an owning module remain functional.
    Graphs without detectable FakeTensorMode metadata also remain functional.
    A supplied updater must own this graph. Each successful rewrite updates affected
    FakeTensor metadata and storage indexes before considering the next candidate.
    Incomplete structured-output rewrites and layout changes reaching lowerings
    without metadata providers remain functional.
    Equivalent mutable results redirect data users to their inputs, while collective
    result references are retained because their lowering may realize input copies.
    Auto-functionalized output rewiring requires subsequent decomposition and is
    enabled explicitly by callers that guarantee it.
    """

    if fake_tensor_updater is None:
        if graph.owning_module is None:
            log.debug("skipping reinplacement: graph has no owning module")
            return
        fake_tensor_updater = FakeTensorUpdater(graph.owning_module)
    elif fake_tensor_updater.gm.graph is not graph:
        raise ValueError("reinplacement updater must own the supplied graph")

    copy_args_to_copy_nodes = {}
    copy_nodes: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = defaultdict(OrderedSet)
    mutated_inputs = OrderedSet[Any]()
    storage_to_nodes = defaultdict(OrderedSet)
    node_to_storage = {}
    canonicalized_storages = OrderedSet[int]()
    node_order: dict[Any, int] = {}

    def record_copyback(node):
        if node.target is aten.copy_.default and node.args[0].op in (
            "placeholder",
            "get_attr",
        ):
            dst = node.args[0]
            src = node.args[1]
            # If the target is a getitem and it indexes a possible clone,
            # then skip over it
            if src.target is operator.getitem and (
                (
                    is_user_kernel_functional_hop(src.args[0].target)
                    and src.args[0].kwargs["kwargs"][src.args[1]] == node.args[0]
                )
                or (src.args[0].target in inplaceable_foreach_ops)
                or (src.args[0].target is torch.ops.higher_order.auto_functionalized)
            ):
                src = src.args[0]

            previous = copy_args_to_copy_nodes.get((dst, src))
            if (
                previous is None
                or previous._erased
                or node_order[node] < node_order[previous]
            ):
                copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst].add(node)

            mutated_inputs.add(node.args[0])

    def get_input_copybacks_after_candidate(node, mutated_arg):
        base = _get_view_base(mutated_arg)
        return (
            copy_node
            for copy_node in copy_nodes.get(base, ())
            if not copy_node._erased
            and copy_node.args[0] is base
            and node_order[copy_node] > node_order[node]
        )

    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage = get_node_storage(node)
        node_to_storage[node] = storage
        storage_to_nodes[storage].add(node)
        record_copyback(node)

    fake_mode = (
        V.fake_mode
        if isinstance(V.fake_mode, FakeTensorMode)
        else detect_fake_mode([n.meta.get("val") for n in graph.nodes])
    )
    if fake_mode is None:
        log.debug("skipping reinplacement: no FakeTensorMode metadata is available")
        return
    mutable_results = {
        op.inplace_op: op.mutated_args
        for target, op in inplaceable_ops.items()
        if target not in inplaceable_collective_ops
    }

    def lowering_ancestors() -> OrderedSet[torch.fx.Node]:
        ancestors = OrderedSet[torch.fx.Node]()
        pending = [
            node
            for node in graph.nodes
            if _is_unsupported_inductor_lowering(node.target)
        ]
        while pending:
            current = pending.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            pending.extend(current.all_input_nodes)
        return ancestors

    nodes_reaching_unsupported_lowering = lowering_ancestors()

    def same_layout(replaced: torch.fx.Node, replacement: Any) -> bool:
        replaced_values = pytree.tree_leaves(replaced.meta.get("val"))
        replacement_nodes = pytree.tree_leaves(replacement)
        return len(replaced_values) == len(replacement_nodes) and all(
            isinstance(value, torch.Tensor)
            and isinstance(replacement_node, torch.fx.Node)
            and isinstance(replacement_node.meta.get("val"), torch.Tensor)
            and _same_size_stride_and_storage_offset(
                value, replacement_node.meta["val"]
            )
            for value, replacement_node in zip(
                replaced_values, replacement_nodes, strict=True
            )
        )

    def cannot_refresh_lowering_metadata(
        replaced: torch.fx.Node, replacement: Any
    ) -> bool:
        return replaced in nodes_reaching_unsupported_lowering and not same_layout(
            replaced, replacement
        )

    def any_use_of_views_after_node(
        node, shared_view_nodes, *, copy_node, mutated_arg, include_current=False
    ):
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        def is_meta_only_user(node):
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            return node.target in META_ONLY_OPS

        for view in shared_view_nodes:
            for user in view.users:
                user_loc = node_order[user]
                # Skip all users before node
                if user_loc < node_loc or (
                    user_loc == node_loc and not include_current
                ):
                    continue
                # Ignore uses after the copy_ epilogue node, where the input
                # has already been mutated anyway
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # Reinplacing does not change shape metadata
                if is_meta_only_user(user):
                    continue
                # If our graph looks like:
                # foo(mutated_arg)
                # mutated_arg.copy_(other)
                # then it's safe for us to reinplace foo because mutated_arg
                # will get overwritten anyways.
                if (
                    user.target is torch.ops.aten.copy_.default
                    and mutated_arg is user.args[0]
                ):
                    continue
                # control_deps args[0] is an ordering-only deps tuple;
                # appearing there does not constitute a real data use.
                if _is_control_deps_ordering_only_use(user, view):
                    continue
                return True
        return False

    def writes_storage(node, storage):
        if node.target is torch.ops.higher_order.control_deps:
            subgraph_attr = cast(torch.fx.Node, node.args[1])
            subgraph = getattr(fake_tensor_updater.gm, cast(str, subgraph_attr.target))
            return storage in get_mutated_storages(subgraph)
        return any(
            get_node_storage(written) == storage
            for written in get_mutated_input_nodes(node)
        )

    def get_mutation_results(node, mutated_args):
        """Collect complete tensor results per input; [] means unknown or incompatible."""
        results = {arg: [] for arg in mutated_args}
        if node.target in inplaceable_foreach_ops:
            outputs = node.meta.get("val")
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
                mutated_args
            ):
                return results
            for user in node.users:
                if (
                    user.target is not operator.getitem
                    or not isinstance(user.args[1], int)
                    or not -len(mutated_args) <= user.args[1] < len(mutated_args)
                ):
                    return {arg: [] for arg in mutated_args}
                results[mutated_args[user.args[1]]].append(user)
            for arg, output in zip(mutated_args, outputs, strict=True):
                tracked = results[arg]
                # Foreach result i belongs only to input i; require complete tracking.
                if (
                    not same_tensor_meta(
                        arg, output, skip_strides=True, skip_storage_offset=True
                    )
                    or not tracked
                    or any(
                        get_node_storage(result) != get_storage(output)
                        for result in tracked
                    )
                ):
                    results[arg] = []
            return results

        tensor_results = (
            [node]
            if isinstance(node.meta.get("val"), torch.Tensor)
            else [
                user
                for user in node.users
                if isinstance(user.meta.get("val"), torch.Tensor)
            ]
        )
        if not isinstance(node.meta.get("val"), torch.Tensor) and any(
            user.target is not operator.getitem for user in node.users
        ):
            return results
        output_storages = FakeTensorUpdater._storage_ids(node.meta.get("val"))
        tracked_storages = OrderedSet(
            get_node_storage(result) for result in tensor_results
        )
        if output_storages is not None and all(
            storage in tracked_storages for storage in output_storages
        ):
            # Without an operator-specific relationship, protect every tensor result.
            return dict.fromkeys(mutated_args, tensor_results)
        return results

    @cache
    def storage_ids_for_node(node):
        if "val" not in node.meta:
            return None
        storages = FakeTensorUpdater._storage_ids(node.meta["val"])
        return None if storages is None else frozenset(storages)

    def get_storage_aliases(roots):
        """Find aliases and the containers that can carry them to readers."""
        pending = list(roots)
        root_storages = [storage_ids_for_node(root) for root in pending]
        unknown_roots = any(storages is None for storages in root_storages)
        protected_storages = frozenset(
            storage for storages in root_storages for storage in (storages or ())
        )

        def carries_protected_storage(node):
            if unknown_roots:
                return True
            storages = storage_ids_for_node(node)
            return storages is None or not protected_storages.isdisjoint(storages)

        aliases = OrderedSet()
        seen_storages = OrderedSet()
        while pending:
            current = pending.pop()
            if current in aliases:
                continue
            aliases.add(current)
            if isinstance(current.meta.get("val"), torch.Tensor):
                storage = get_node_storage(current)
                if storage is None or storage in seen_storages:
                    continue
                seen_storages.add(storage)
                related = storage_to_nodes[storage]
                aliases.update(related)
            else:
                related = (current,)

            pending.extend(
                parent
                for alias in related
                if alias.target is operator.getitem
                and isinstance(parent := alias.args[0], torch.fx.Node)
                and not isinstance(parent.meta.get("val"), torch.Tensor)
                and carries_protected_storage(parent)
            )
            pending.extend(
                user
                for alias in related
                for user in alias.users
                if carries_protected_storage(user)
            )
        return aliases

    def result_views_preserve_layout(result, mutated_arg):
        if same_tensor_meta(result, mutated_arg) and statically_known_true(
            result.untyped_storage().nbytes()
            == mutated_arg.meta["val"].untyped_storage().nbytes()
        ):
            return True
        same_strides = same_tensor_meta(result, mutated_arg, skip_storage_offset=True)
        for alias in get_storage_aliases(storage_to_nodes[get_storage(result)]):
            if any(
                user.target in (aten.sym_stride.int, aten.sym_storage_offset.default)
                or (
                    user.op == "call_function"
                    and user.target is not operator.getitem
                    and (
                        not isinstance(user.target, torch._ops.OpOverload)
                        or user.target.namespace not in ("aten", "prims")
                    )
                    and not _is_control_deps_ordering_only_use(user, alias)
                )
                for user in alias.users
            ):
                return False
            if not _is_view_op(alias.target):
                continue
            if alias.target in (
                aten.alias.default,
                aten.detach.default,
                aten.slice.Tensor,
                aten.select.int,
                aten.diagonal.default,
                aten.permute.default,
                aten.transpose.int,
                aten.t.default,
                aten.unsqueeze.default,
                aten.squeeze.default,
                aten.squeeze.dim,
                aten.squeeze.dims,
            ):
                continue
            if same_strides and alias.target in (
                aten.view.default,
                aten._unsafe_view.default,
                aten.reshape.default,
            ):
                continue
            return False
        return True

    def can_inplace(node, mutated_arg):
        def may_overlap(tensors) -> bool:
            try:
                return bool(compute_overlapping_tensors(tensors))
            except GuardOnDataDependentSymNode:
                return True

        is_tensor_list = isinstance(mutated_arg, (list, tuple))
        mutated_args = mutated_arg if is_tensor_list else [mutated_arg]
        foreach_results = None
        if is_tensor_list:
            unique_storages = OrderedSet(get_node_storage(arg) for arg in mutated_args)
            if len(unique_storages) != len(mutated_args):
                return False
            if node.target in inplaceable_foreach_ops:
                # A same-index operand may read exactly the elements being written.
                # Cross-index aliases and different views can read overwritten values.
                for operands in (*node.args[1:], *node.kwargs.values()):
                    if isinstance(operands, (list, tuple)) and len(operands) == len(
                        mutated_arg
                    ):
                        for arg, operand in zip(mutated_arg, operands, strict=True):
                            if (
                                isinstance(operand, torch.fx.Node)
                                and get_node_storage(operand) in unique_storages
                                and not (
                                    get_node_storage(operand) == get_node_storage(arg)
                                    and same_tensor_meta(arg, operand)
                                )
                            ):
                                return False
                    elif any(
                        isinstance(operand, torch.fx.Node)
                        and get_node_storage(operand) in unique_storages
                        for operand in pytree.tree_leaves(operands)
                    ):
                        return False
                foreach_results = get_mutation_results(node, mutated_args)
                if not all(foreach_results.values()):
                    return False

        for mutated_arg in mutated_args:
            result_nodes = foreach_results[mutated_arg] if foreach_results else []
            mutated_storage = get_node_storage(mutated_arg)
            if mutated_storage is None:
                return False

            tensor = mutated_arg.meta["val"]
            overlap = torch._debug_has_internal_overlap(tensor)
            if overlap == 1 or (overlap != 0 and not _has_disjoint_strides(tensor)):
                return False

            # Keep aliases whose tensor contents might overlap the mutated input.
            shared_view_nodes = [
                view
                for view in get_storage_aliases([mutated_arg])
                if storage_ids_for_node(view) is None
                or any(
                    may_overlap([tensor, value])
                    for value in pytree.tree_leaves(view.meta.get("val"))
                    if isinstance(value, torch.Tensor)
                )
            ]

            copy_node = None
            mutated_arg_base = mutated_arg
            if any(
                view.op in ("placeholder", "get_attr") for view in shared_view_nodes
            ):
                # Without a later copy into the input, the caller could observe an
                # extra mutation even if the graph has no remaining reads of it.
                mutated_arg_base = _get_view_base(mutated_arg)
                # Use the next input overwrite for permission and liveness checks.
                copy_node = min(
                    get_input_copybacks_after_candidate(node, mutated_arg_base),
                    key=node_order.__getitem__,
                    default=None,
                )
                if copy_node is None:
                    return False

            removable_copyback = copy_node_for_reinplaced_arg(node, mutated_arg)
            overwrite_source_storage = None
            # Candidate-specific copybacks keep their existing permission checks.
            if (
                copy_node is not None
                and (
                    mutated_arg.op not in ("placeholder", "get_attr")
                    or node.target in inplaceable_foreach_ops
                )
                and copy_node is not removable_copyback
            ):
                # An unrelated copy can hide the view mutation only if it overwrites
                # all input storage, including any bytes outside the input's shape.
                base_val = mutated_arg_base.meta["val"]
                if not (
                    is_contiguous_or_false(base_val)
                    and statically_known_true(base_val.storage_offset() == 0)
                    and statically_known_true(
                        base_val.numel() * base_val.element_size()
                        == base_val.untyped_storage().nbytes()
                    )
                ):
                    return False

                overwrite_source_storage = get_node_storage(copy_node.args[1])
                # A self-copy or aliased source would preserve the early mutation
                # instead of restoring the input's required final values.
                if (
                    overwrite_source_storage is None
                    or overwrite_source_storage == mutated_storage
                ):
                    return False

            if any_use_of_views_after_node(
                node,
                shared_view_nodes,
                copy_node=copy_node,
                mutated_arg=mutated_arg_base,
            ):
                return False

            result_roots = list(result_nodes)
            if get_node_storage(node) is not None or (
                not result_nodes and mutated_storage in canonicalized_storages
            ):
                result_roots.append(node)
            result_views = get_storage_aliases(result_roots)

            preserved_copyback_views = OrderedSet()
            if copy_node is not None and overwrite_source_storage is not None:
                # Expanded permission for mutable lists requires a result for each input.
                if not is_tensor_list:
                    result_nodes = get_mutation_results(node, [mutated_arg])[
                        mutated_arg
                    ]
                if not result_nodes:
                    return False
                for result in result_nodes:
                    views = get_storage_aliases([result])
                    result_views.update(views)
                    if overwrite_source_storage == get_node_storage(result):
                        # A copy of the result must preserve its view/copy semantics.
                        if not _is_layout_preserving_view_copy_back(
                            mutated_arg_base, copy_node.args[1], mutated_arg, result
                        ):
                            return False
                        preserved_copyback_views.update(views)

            layout_roots = result_nodes or [node]
            if any(
                not result_views_preserve_layout(result, mutated_arg)
                for root in layout_roots
                for result in pytree.tree_leaves(root.meta.get("val"))
                if isinstance(result, torch.Tensor) and torch._C._has_storage(result)
            ):
                return False

            if result_views:
                mutations = OrderedSet(
                    user
                    for alias in storage_to_nodes[mutated_storage]
                    for user in alias.users
                    if node_order[user] > node_order[node]
                    and user is not removable_copyback
                )
                for mutation in mutations:
                    if (
                        copy_node is not None
                        and mutation is copy_node
                        and node.target is torch.ops.higher_order.auto_functionalized_v2
                        and (source := copy_node.args[1]).target is operator.getitem
                        and source.args[0] is node
                        and get_mutated_output_indices(
                            node, node.kwargs["_all_bases"]
                        ).get(mutated_arg)
                        == source.args[1]
                        and _is_layout_preserving_view_copy_back(
                            copy_node.args[0], source, mutated_arg, source
                        )
                    ):
                        continue
                    views = (
                        result_views - preserved_copyback_views
                        if mutation is copy_node
                        else result_views
                    )
                    if writes_storage(
                        mutation, mutated_storage
                    ) and any_use_of_views_after_node(
                        mutation,
                        views,
                        copy_node=None,
                        mutated_arg=node,
                        include_current=True,
                    ):
                        return False

        return True

    def copy_node_for_reinplaced_arg(node, mutated_arg):
        copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
        if copy_node is not None and copy_node._erased:
            copy_node = None
        if copy_node is not None and node.target in inplaceable_foreach_ops:
            src = copy_node.args[1]
            inputs = node.args[0]
            if (
                src.target is not operator.getitem
                or src.args[0] is not node
                or not isinstance(index := src.args[1], int)
                or not -len(inputs) <= index < len(inputs)
                or inputs[index] is not mutated_arg
            ):
                copy_node = None
        if copy_node is None and isinstance(mutated_arg, torch.fx.Node):
            # Only a layout-preserving copy of this candidate's result can be removed
            # upon reinplacement of candidate.
            copy_node = min(
                (
                    copy_node
                    for copy_node in get_input_copybacks_after_candidate(
                        node, mutated_arg
                    )
                    if isinstance(copy_node.args[1], torch.fx.Node)
                    and _get_view_base(copy_node.args[1]) is node
                    and _is_layout_preserving_view_copy_back(
                        copy_node.args[0], copy_node.args[1], mutated_arg, node
                    )
                ),
                key=node_order.__getitem__,
                default=None,
            )
        return copy_node

    def copy_source_uses_candidate(node, source, independent):
        seen = OrderedSet()
        pending = [source]
        while pending:
            current = pending.pop()
            if current is node:
                return True
            if (
                current in seen
                or current in independent
                or node_order[current] < node_order[node]
            ):
                continue
            seen.add(current)
            pending.extend(
                argument
                for argument in current.all_input_nodes
                if not _is_control_deps_ordering_only_use(current, argument)
            )
        independent.update(seen)
        return False

    def foreach_reuse_preserves_fusion(node, mutated_args):
        copybacks = {
            arg: min(
                get_input_copybacks_after_candidate(node, _get_view_base(arg)),
                key=node_order.__getitem__,
                default=None,
            )
            for arg in mutated_args
        }
        has_input_copyback = any(copybacks.values())
        independent = OrderedSet()
        for arg in mutated_args:
            if copy_node_for_reinplaced_arg(node, arg) is not None:
                continue
            if not has_input_copyback and not is_node_realized(_get_view_base(arg)):
                return False
            if (copy_node := copybacks[arg]) is not None:
                source = _get_view_base(copy_node.args[1])
                if source.target is operator.getitem and source.args[0] is node:
                    continue
                if not is_node_realized(source) and copy_source_uses_candidate(
                    node, source, independent
                ):
                    return False
        return True

    def all_can_inplace(node, mutated_args):
        return all(can_inplace(node, arg) for arg in mutated_args)

    def log_inplace_results(
        node_name,
        old_tensors_to_clone,
        tensors_to_clone,
        missed_args,
        missed_nodes,
        trigger,
    ):
        # Total size of possibly_missed_reinplacing_opportunities for tensors with static shapes.
        missed_bytes = 0

        def bytes(node):
            t = node.meta.get("val", None)
            if (
                t is not None
                and isinstance(t.element_size(), int)
                and isinstance(t.numel(), int)
            ):
                return t.element_size() * t.numel()
            else:
                return 0

        for node in missed_nodes:
            if isinstance(node, (list, tuple)):
                for n in node:
                    missed_bytes += bytes(n)
            else:
                missed_bytes += bytes(node)

        log.info(
            "For node %s, attempted to reinplace %s. We were unable to reinplace %s; "
            "%s (if non-empty) are possible missed reinplacing opportunities that may be bad for "
            "memory usage and performance. Total size of missed opportunities with static shapes is"
            " : %s bytes.",
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_bytes,
        )

        ReinplaceCounters.add_missed_opportunities(trigger, len(missed_args))
        ReinplaceCounters.add_missed_bytes(trigger, missed_bytes)

    def resolve_getitem_replacements(
        current_node: torch.fx.Node, replacement: Any
    ) -> dict[torch.fx.Node, torch.fx.Node] | None:
        """Build a replacement plan for a complete nested `getitem` tree.

        Tensor leaves map to mutated inputs; intermediate nodes are erased after
        their children. Return `None` if any live output cannot be safely rewired.
        """
        replacements: dict[torch.fx.Node, torch.fx.Node] = {}

        def visit(node: torch.fx.Node, value: Any) -> bool:
            getitems = [user for user in node.users if user.target is operator.getitem]
            if not getitems:
                if not isinstance(value, torch.fx.Node):
                    return False
                if cannot_refresh_lowering_metadata(node, value):
                    return False
                replacements[node] = value
                return True
            if not isinstance(value, (list, tuple)) or any(
                user.target is not operator.getitem for user in node.users
            ):
                return False
            first_replacement: torch.fx.Node | None = None
            for user in getitems:
                index = user.args[1]
                if not isinstance(index, int) or not 0 <= index < len(value):
                    return False
                if not visit(user, value[index]):
                    return False
                if first_replacement is None:
                    first_replacement = replacements[user]
            if first_replacement is None:
                return False
            replacements[node] = first_replacement
            return True

        return replacements if visit(current_node, replacement) else None

    def get_mutated_output_indices(node, mutated_args):
        """Map mutable outputs that do not alias ordinary operator results."""
        if not rewire_auto_functionalized_outputs:
            return {}
        outputs = node.meta.get("val")
        if not isinstance(outputs, tuple) or len(outputs) < len(mutated_args):
            return {}
        offset = _updated_base_offset(outputs, len(mutated_args))
        original_storages = FakeTensorUpdater._storage_ids(outputs[:offset])
        if original_storages is None:
            return {}
        original_storages = OrderedSet(original_storages)
        indices = {}
        for index, arg in enumerate(mutated_args, offset):
            storages = FakeTensorUpdater._storage_ids(outputs[index])
            if storages is not None and not original_storages.intersection(storages):
                indices[arg] = index
        return indices

    def reinplace_and_refine_tensors_to_clone(
        old_tensors_to_clone, kwargs, node_name, trigger, replacements, output_indices
    ):
        tensors_to_clone: list[str] = []
        storage_of_reinplaced_args = OrderedSet[int | None]()

        # Those used to count possibly_missed_reinplacing_opportunities
        missed_nodes = []
        missed_args = []

        # TODO this logic can be made more precise using _overlap
        def tensor_with_same_storage_already_reinplaced(arg):
            if isinstance(arg, (list, tuple)):
                return any(
                    get_node_storage(a) in storage_of_reinplaced_args for a in arg
                )
            return get_node_storage(mutated_arg) in storage_of_reinplaced_args

        for arg in old_tensors_to_clone:
            if arg not in kwargs:
                raise AssertionError(f"expected {arg} to be in kwargs")

            mutated_arg = kwargs[arg]
            output_index = output_indices.get(arg)
            replacement_users = [
                user
                for user in node.users
                if output_index is not None
                and user.target is operator.getitem
                and user.args[1] == output_index
            ]
            output_replacements: dict[torch.fx.Node, torch.fx.Node] = {}
            outputs_safe = True
            for user in replacement_users:
                planned = resolve_getitem_replacements(user, mutated_arg)
                if planned is None:
                    # Unrewired auto-functionalized outputs remain valid after decomposition.
                    output_replacements.clear()
                    outputs_safe = trigger in (
                        ReInplaceTrigger.AUTO_FUNC_V1,
                        ReInplaceTrigger.AUTO_FUNC_V2,
                    )
                    break
                output_replacements.update(planned)

            # Let's say we have:
            # - op(x, y) that mutates both x and y
            # - new_x, new_y = functional_op(x, y) is the functional variant
            # If we are presented with functional_op(x, x), we must not reinplace
            # this into op(x, x), because then it would be writing to the same Tensor.
            # Instead, it's OK to reinplace one of them and to clone the other:
            # >>> y = x.clone()
            # >>> op(x, y)
            # This also applies if we have views: functional_op(x, x[0])
            # should not reinplace into op(x, x[0]).
            should_attempt_reinplace = (
                not tensor_with_same_storage_already_reinplaced(mutated_arg)
                and outputs_safe
            )
            if should_attempt_reinplace and can_inplace(node, mutated_arg):
                # In general, we probably do not need those optimizations.
                copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                if copy_node is not None:
                    replacements[copy_node] = copy_node.args[0]
                replacements.update(output_replacements)
                if isinstance(mutated_arg, (list, tuple)):
                    for a in mutated_arg:
                        storage_of_reinplaced_args.add(get_node_storage(a))
                else:
                    storage_of_reinplaced_args.add(get_node_storage(mutated_arg))
            else:
                if should_attempt_reinplace:
                    missed_args.append(arg)
                    missed_nodes.append(mutated_arg)

                tensors_to_clone.append(arg)

        log_inplace_results(
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_nodes,
            trigger,
        )
        return tensors_to_clone

    def redirect_uses(replaced, replacement, ordering_replacement):
        """Redirect data references while retaining the mutation's ordering reference."""
        rewired = OrderedSet[torch.fx.Node]()
        ordering_only = OrderedSet[torch.fx.Node]()
        for user in tuple(replaced.users):
            only_ordering = _is_control_deps_ordering_only_use(user, replaced)
            if only_ordering and ordering_replacement is replaced:
                continue
            if user.target is torch.ops.higher_order.control_deps:
                deps = pytree.tree_map(
                    lambda value: ordering_replacement if value is replaced else value,
                    user.args[0],
                )
                arguments = pytree.tree_map(
                    lambda value: replacement if value is replaced else value,
                    user.args[2:],
                )
                user.args = (deps, user.args[1], *arguments)
            else:
                user.replace_input_with(replaced, replacement)
            rewired.add(user)
            if only_ordering:
                ordering_only.add(user)
        return rewired, ordering_only

    def finalize_rewrite(node, changed_nodes, replacements):
        """Apply FX graph rewrites and metadata refresh after successful reinplacement."""
        if not changed_nodes and not replacements:
            return

        # Consumer metadata must reflect the completed graph rewrites.
        metadata_unchanged_users = OrderedSet[torch.fx.Node]()
        for replaced, replacement in replacements.items():
            if replaced._erased:
                continue
            while replacement in replacements:
                replacement = replacements[replacement]
            rewired, ordering_only = redirect_uses(replaced, replacement, node)
            if replaced.target is not aten.copy_.default and rewired - ordering_only:
                storage = get_node_storage(replacement)
                if storage is not None:
                    canonicalized_storages.add(storage)
            for user in rewired:
                if not (
                    user in ordering_only
                    or (
                        same_layout(replaced, replacement)
                        and getattr(
                            user.target,
                            "_inductor_lowering_output_metadata_ignores_input_storage",
                            False,
                        )
                        and getattr(
                            user.target,
                            "_inductor_lowering_output_metadata_is_input",
                            None,
                        )
                        is None
                    )
                ):
                    changed_nodes.add(user)
                else:
                    metadata_unchanged_users.add(user)
            changed_nodes.add(replaced)
            graph.erase_node(replaced)
        fake_tensor_updater.acknowledge_metadata(
            user
            for user in metadata_unchanged_users
            if not user._erased and user not in changed_nodes
        )

        # Refresh metadata after the completed consumer rewrites.
        roots = OrderedSet(
            node for node in changed_nodes if not node._erased and callable(node.target)
        )
        if roots:
            with V.set_fake_mode(fake_mode):
                changed_nodes.update(
                    fake_tensor_updater.incremental_update_from_nodes(roots)
                )

        # Redirect equivalent results while retaining the mutation node.
        redirected_users = OrderedSet[torch.fx.Node]()
        if (indices := mutable_results.get(node.target)) is not None and len(
            indices
        ) == 1:
            argument = node.args[indices[0]]
            candidates = []
            if isinstance(argument, torch.fx.Node):
                candidates.append((node, argument))
            elif isinstance(argument, (list, tuple)):
                for user in tuple(node.users):
                    if (
                        user.target is operator.getitem
                        and isinstance(user.args[1], int)
                        and -len(argument) <= user.args[1] < len(argument)
                    ):
                        candidates.append((user, argument[user.args[1]]))
            for result, replacement in candidates:
                storage = get_node_storage(result)
                if (
                    isinstance(replacement, torch.fx.Node)
                    and storage is not None
                    and storage == get_node_storage(replacement)
                    and same_tensor_meta(result, replacement)
                ):
                    rewired, ordering_only = redirect_uses(result, replacement, node)
                    if rewired - ordering_only:
                        canonicalized_storages.add(storage)
                    redirected_users.update(rewired)
                    changed_nodes.update(rewired)
                    if result is not node and not result.users:
                        graph.erase_node(result)
                        changed_nodes.add(result)
        fake_tensor_updater.acknowledge_metadata(redirected_users)

        # Copyback sources may now refer directly to mutated inputs.
        for changed in tuple(changed_nodes):
            if not changed._erased:
                record_copyback(changed)

        # Alias buckets must match refreshed values before the next candidate.
        for updated in tuple(changed_nodes):
            if not updated._erased:
                changed_nodes.update(updated.all_input_nodes)
        old_storages = OrderedSet[int]()
        for updated in tuple(changed_nodes):
            old_storage = node_to_storage.get(updated)
            if old_storage is not None and (
                updated._erased or old_storage != get_node_storage(updated)
            ):
                old_storages.add(old_storage)
        for old_storage in old_storages:
            changed_nodes.update(storage_to_nodes[old_storage])
        for updated in changed_nodes:
            if updated in node_to_storage:
                old_storage = node_to_storage[updated]
                if updated._erased or old_storage != get_node_storage(updated):
                    storage_to_nodes[old_storage].discard(updated)
                    if not storage_to_nodes[old_storage]:
                        del storage_to_nodes[old_storage]
                    del node_to_storage[updated]
            if not updated._erased and updated not in node_to_storage:
                storage = get_node_storage(updated)
                node_to_storage[updated] = storage
                storage_to_nodes[storage].add(updated)

    for node in graph.nodes:
        storage_ids_for_node.cache_clear()
        changed_nodes = OrderedSet[torch.fx.Node]()
        replacements: dict[torch.fx.Node, torch.fx.Node] = {}
        if (inplaceable_op := inplaceable_ops.get(node.target)) is not None:
            # Check if ALL mutated args can be inplaced
            # Only convert if we don't need to clone any tensor
            mutated_args = [node.args[idx] for idx in inplaceable_op.mutated_args]
            reinplace = (
                all_can_inplace(node, mutated_args)
                and inplaceable_op.extra_check(node)
                and all(
                    not cannot_refresh_lowering_metadata(node, mutated_arg)
                    for mutated_arg in mutated_args
                )
            )
            into_mutated_base = node.meta.pop(_MUTATED_BASE_SCATTER, False)
            if reinplace:
                for mutated_arg in mutated_args:
                    copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                    if copy_node is not None:
                        replacements[copy_node] = copy_node.args[0]
                node.target = inplaceable_op.inplace_op
                changed_nodes.add(node)
                if into_mutated_base:
                    counters["inductor"]["reinplace_mutated_base_scatter"] += 1
                    log.debug(
                        "reinplaced scatter %s into an auto_functionalized_v2 base",
                        node,
                    )
        elif node.target is torch.ops.higher_order.auto_functionalized_v2:
            _mutable_op = node.args[0]
            kwargs = node.kwargs

            all_bases = kwargs["_all_bases"]
            bases_to_clone = range(len(all_bases))
            base_tensors_dct = dict(enumerate(all_bases))
            new_bases_to_clone = reinplace_and_refine_tensors_to_clone(
                bases_to_clone,
                base_tensors_dct,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V2,
                replacements,
                get_mutated_output_indices(node, bases_to_clone),
            )
            if replacements:
                changed_nodes.add(node)
            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = new_bases_to_clone
        elif node.target is torch.ops.higher_order.auto_functionalized:
            _mutable_op = node.args[0]
            from torch._higher_order_ops.auto_functionalize import get_mutable_args

            mutable_args, _ = get_mutable_args(_mutable_op)
            # Don't try to reinplace Tensor | None args that are None.
            tensors_to_clone = [t for t in mutable_args if node.kwargs[t] is not None]
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                tensors_to_clone,
                node.kwargs,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V1,
                replacements,
                get_mutated_output_indices(node, mutable_args),
            )
            if replacements:
                changed_nodes.add(node)

            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = tensors_to_clone
        elif node.target is torch.ops.higher_order.with_effects:
            # Handle effectful ops wrapped with with_effects
            # args[0] is the token, args[1] is the inner op, args[2:] are the op's args
            inner_op = node.args[1]
            log.debug(
                "reinplace: checking with_effects node with inner_op=%s", inner_op
            )
            if (inplaceable_op := inplaceable_ops.get(inner_op)) is not None:
                log.debug("reinplace: found inplaceable_op for %s", inner_op)
                # Get the mutated arg indices, offset by 2 (token + op)
                mutated_arg_indices = inplaceable_op.mutated_args

                # Build flat list of tensors for can_inplace check
                # and a mapping of output index -> replacement tensor(s)
                mutated_tensors_flat = []
                output_idx_to_replacement: dict[int, Any] = {}

                for position, idx in enumerate(mutated_arg_indices):
                    actual_idx = idx + 2  # offset for token and op
                    if actual_idx >= len(node.args):
                        raise AssertionError(
                            f"mutated arg idx {actual_idx} out of range {len(node.args)}"
                        )
                    arg = node.args[actual_idx]

                    # Output index is position + 1 (index 0 is the token)
                    output_idx = position + 1
                    output_idx_to_replacement[output_idx] = arg

                    # Flatten for can_inplace check
                    if isinstance(arg, (list, tuple)):
                        mutated_tensors_flat.extend(arg)
                    else:
                        mutated_tensors_flat.append(arg)

                # Check if all mutated args can be inplaced
                can_inplace_all = all_can_inplace(node, mutated_tensors_flat)
                output_replacements: dict[torch.fx.Node, torch.fx.Node] = {}
                can_rewire_outputs = True
                result_users = (
                    () if inner_op in inplaceable_collective_ops else node.users
                )
                for user in result_users:
                    if user.target is not operator.getitem or not isinstance(
                        user.args[1], int
                    ):
                        can_rewire_outputs = False
                        break
                    index = user.args[1]
                    if index == 0:
                        continue
                    if index not in output_idx_to_replacement:
                        can_rewire_outputs = False
                        break
                    planned = resolve_getitem_replacements(
                        user, output_idx_to_replacement[index]
                    )
                    if planned is None:
                        can_rewire_outputs = False
                        break
                    output_replacements.update(planned)

                log.debug(
                    "reinplace with_effects: mutated_tensors=%s, can_inplace_all=%s",
                    [str(a) for a in mutated_tensors_flat],
                    can_inplace_all,
                )

                if (
                    can_inplace_all
                    and can_rewire_outputs
                    and inplaceable_op.extra_check(node)
                ):
                    log.debug(
                        "reinplace with_effects: converting %s -> %s",
                        inner_op,
                        inplaceable_op.inplace_op,
                    )
                    # Update the inner op to inplace version
                    node.update_arg(1, inplaceable_op.inplace_op)
                    changed_nodes.add(node)
                    replacements.update(output_replacements)
        elif user_kernel_hop := get_user_kernel_hop_info(node.target, "functional"):
            kernel_name = user_kernel_hop.get_kernel_name(node.kwargs)

            # User-kernel functional HOPs take an additional argument called
            # tensors_to_clone which contain a list of tensors to clone
            # This pass iterates over them and sees which ones are safe
            # to eliminate (i.e. no longer need the clones)
            old_tensors_to_clone = node.kwargs["tensors_to_clone"]
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                old_tensors_to_clone,
                node.kwargs["kwargs"],
                kernel_name,
                user_kernel_hop.reinplace_trigger,
                replacements,
                {arg: arg for arg in old_tensors_to_clone},
            )

            kwargs = dict(node.kwargs)
            if replacements or tensors_to_clone != old_tensors_to_clone:
                changed_nodes.add(node)
            kwargs["tensors_to_clone"] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)
            if "eager_input_vals" in node.meta:
                # We changed the kwargs, so we need to update eager_input_vals
                # to something sane.
                args, kwargs = node.meta["eager_input_vals"]
                new_kwargs = {**kwargs}
                new_kwargs["tensors_to_clone"] = immutable_list(tensors_to_clone)
                new_kwargs = immutable_dict(new_kwargs)
                node.meta["eager_input_vals"] = (args, new_kwargs)
        elif (inplaceable_op := inplaceable_foreach_ops.get(node.target)) is not None:
            mutated_args = node.args[inplaceable_op.mutated_arg]
            result_replacements: dict[torch.fx.Node, torch.fx.Node] = {}
            can_rewire_results = True
            for user in node.users:
                if (
                    user.target is operator.getitem
                    and isinstance(user.args[1], int)
                    and -len(mutated_args) <= user.args[1] < len(mutated_args)
                ):
                    planned = resolve_getitem_replacements(
                        user, mutated_args[user.args[1]]
                    )
                    if planned is None:
                        can_rewire_results = False
                        break
                    result_replacements.update(planned)
                else:
                    can_rewire_results = False
                    break

            if (
                can_rewire_results
                and can_inplace(node, mutated_args)
                and foreach_reuse_preserves_fusion(node, mutated_args)
            ):
                for arg in mutated_args:
                    copy_node = copy_node_for_reinplaced_arg(node, arg)
                    if copy_node is not None:
                        replacements[copy_node] = copy_node.args[0]
                replacements.update(result_replacements)

                node.target = inplaceable_op.inplace_op
                changed_nodes.add(node)
        finalize_rewrite(node, changed_nodes, replacements)


def reinplace_inplaceable_ops(
    fake_tensor_updater: torch._inductor.fx_utils.FakeTensorUpdater,
    graph: torch.fx.Graph,
    *,
    rewire_auto_functionalized_outputs: bool = False,
) -> None:
    """Enable auto-functionalized output rewiring only when decomposition follows."""
    with enable_python_dispatcher():
        canonicalize_view_scatter_ops(graph)
        # canonicalize_view_scatter_ops adds new operations to the graph.
        # We run fake_tensor_updater to update the alias information.
        # Correct alias information is required for `reinplace_inplaceable_ops_core`.
        fake_tensor_updater.incremental_update()
        # Calculate scatter chain length on scatter nodes if we intend to reinplace long
        # scatter chains. Reinplacing scatter may be more efficient on some hardware types.
        # Must calculate scatter chain length on original fx graph. Therefore should
        # precede reinplace_inplaceable_ops_core as that pass rewrites the scatter nodes.
        if config.reinplace_scatter_chain_threshold > 0:
            annotate_scatter_chain_lengths(graph)
        reinplace_inplaceable_ops_core(
            graph,
            fake_tensor_updater=fake_tensor_updater,
            rewire_auto_functionalized_outputs=rewire_auto_functionalized_outputs,
        )
        decompose_generalized_scatter(graph)
