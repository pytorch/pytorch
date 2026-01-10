import functools
import logging
from collections.abc import Callable, Sequence
from enum import Enum
from queue import Queue
from typing import Any, TypeAlias

import torch
from torch.fx import Graph, Node
from .common import CantChunk
from .core import (
    ChunkingMeta,
    copy_chunking_meta,
    get_chunking_meta,
    has_nop_chunking_meta,
    set_chunking_meta,
    set_chunking_meta_if_none,
    update_chunking_meta,
)
from .propagate_scale_by import propagate_scale_by
from .utils import (
    format_node_with_chunking_meta,
    get_args_of_node_type,
    get_fake_tensor_from_node_arg,
    get_first_chunking_meta,
    get_node_is_scalar,
    get_node_ndim,
    get_nodes_with_chunking_meta,
    has_any_chunking_meta,
    is_chunked_by_dim,
    is_tangent_node,
)


log = torch._logging.getArtifactLogger(__name__, "auto_chunker")
aten = torch.ops.aten
prims = torch.ops.prims


"""
NOTE [Why we need both fwd and bwd chunking metadata propagation?]
The starting point of chunking is we found an op that creates much larger outputs
than the inputs size. We attach chunking metadata upon the op and propagate it forward.

But for backward rules like NLLLossBackward, we do a scatter upon a zero matrix. That
zero matrix is created by torch.full. We will only know we should chunk that tensor
by propagating chunking metadata backward.
"""

"""
NOTE [Why we need a separate pass to propagate ChunkingMeta.scale_by?]

ChunkingMeta.scale_by only need to be propagated forward from the tangent placeholder nodes.
If we do this together with propagating other metadata, we can not fully control the propagating
order and end up with cases like:
    out = aten.sub(lhs, rhs)
where `lhs` has scale_by set, while `rhs` and `out` don't.
For sub op, we could propagate `scale_by` to `rhs` and `out` since that's the
only way to make sense. But overall this is unsafe since maybe this is a case
that chunking does not work and we should bail out.
Another example is, we can not really propagate `scale_by` backward for
aten.mul since we don't know which of the input should have this `scale_by` metadata.

But it's safer that we only propagate `scale_by` metadata in the topological order.

Have the `scale_by` handled in a separate pass also makes the fwd/bwd
chunking metadata propagation much simpler. We don't need special rules
for mul/div/where etc due to the special handling of scale_by:
https://gist.github.com/shunting314/324e57881f168009784991300acab852
"""


class PropagateStatus(Enum):
    SUCCEED_NO_CHANGE = 0
    SUCCEED_WITH_CHANGE = 1
    FAIL = 2


_HandlerRetType: TypeAlias = PropagateStatus | tuple[PropagateStatus, PropagateStatus]

_HandlerType: TypeAlias = Callable[[Node], _HandlerRetType]

# Rules to propagate chunking metadata from inputs to the current node
# or from the current node back to its inputs
propagate_rules: dict[torch._ops.OpOverload, _HandlerType] = {}


def _register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
    handler: _HandlerType,
) -> _HandlerType:
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]  # type: ignore[assignment, list-item]

    @functools.wraps(handler)
    def wrapper(node: Node, *args: Any, **kwargs: Any) -> PropagateStatus:
        fwd_bwd_status = handler(node, *args, **kwargs)
        if isinstance(fwd_bwd_status, PropagateStatus):
            return fwd_bwd_status
        assert isinstance(fwd_bwd_status, (list, tuple)) and len(fwd_bwd_status) == 2
        fwd_status, bwd_status = fwd_bwd_status
        log.debug(
            "Chunking metadata propagation for %s: Fwd status %s, bwd status %s",
            node,
            fwd_status,
            bwd_status,
        )
        if fwd_status == PropagateStatus.FAIL or bwd_status == PropagateStatus.FAIL:
            return PropagateStatus.FAIL
        if (
            fwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
            or bwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
        ):
            return PropagateStatus.SUCCEED_WITH_CHANGE
        return PropagateStatus.SUCCEED_NO_CHANGE

    assert isinstance(aten_op, (list, tuple)), f"{type(aten_op)=}"
    for op in aten_op:
        assert isinstance(op, torch._ops.OpOverload)
        propagate_rules[op] = wrapper
    return wrapper


def register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
) -> Callable[[_HandlerType], _HandlerType]:
    return functools.partial(_register_propagate_rule, aten_op)


def _is_success(*statuslist: PropagateStatus) -> bool:
    return not any(status == PropagateStatus.FAIL for status in statuslist)


def _enqueue(queue: Queue, item: Node) -> None:  # type: ignore[type-arg]
    """
    Have a function to make it easier to do debug logging in a central place
    """
    queue.put(item)
    log.debug("Enqueue: %s", item)


def can_reach_amplified_node(
    graph: Graph, amplifier_node: Node, is_fwd: bool
) -> dict[Node, bool]:
    """
    A amplified node means a node with the same numel as `amplifier_node`
    """
    filter_obj: dict[Node, bool] = {}
    nodelist = reversed(graph.nodes) if is_fwd else graph.nodes
    target_numel = get_fake_tensor_from_node_arg(amplifier_node).numel()  # type: ignore[union-attr]

    for node in nodelist:
        reach = False
        if node.op == "output":
            # output node does not have a meta['val']
            reach = False

        elif get_fake_tensor_from_node_arg(node) is None:
            reach = False

        # for the back propagation, we should continue propagate if we can
        # reach a tangent node
        elif get_fake_tensor_from_node_arg(node).numel() == target_numel or (  # type: ignore[union-attr]
            not is_fwd and is_tangent_node(node)
        ):
            reach = True
        else:
            neighbors = node.users if is_fwd else get_args_of_node_type(node)
            reach = any(filter_obj[neighbor] for neighbor in neighbors)
        filter_obj[node] = reach
    return filter_obj


def propagate(amplifier_node: Node) -> None:
    log.debug("amplifier_node is %s", amplifier_node.format_node())
    # Chunk the batch dimension (dim 0) of the amplifier_node
    graph = amplifier_node.graph

    fwd_filter = can_reach_amplified_node(graph, amplifier_node, True)
    bwd_filter = can_reach_amplified_node(graph, amplifier_node, False)

    log.debug("fwd_filter %s", fwd_filter)
    log.debug("bwd_filter %s", bwd_filter)

    assert len(get_nodes_with_chunking_meta(graph)) == 0, (
        "Expect no nodes with chunking meta yet"
    )

    set_chunking_meta(amplifier_node, chunk_dim=0)

    queue: Queue[Node] = Queue()
    _enqueue(queue, amplifier_node)

    while not queue.empty():
        propagate_single_node(queue, fwd_filter, bwd_filter, queue.get())

    nodes_with_chunking_meta = get_nodes_with_chunking_meta(graph)
    propagate_scale_by(nodes_with_chunking_meta)

    if log.isEnabledFor(logging.DEBUG):
        print("All nodes with chunking metadata set:")
        for node in nodes_with_chunking_meta:
            format_node_with_chunking_meta(node)


def propagate_single_node(
    queue: Queue, fwd_filter: dict[Node, bool], bwd_filter: dict[Node, bool], node: Node
) -> None:  # type: ignore[type-arg]
    log.debug("Propagate_single_node: %s", node.format_node())

    if node.op != "call_function":
        raise CantChunk(
            "Chunker can only propagate chunking metadata thru call_function nodes"
        )

    target = node.target
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Before propagation, the node has the following chunking meta:")
        format_node_with_chunking_meta(node, True)

    if not isinstance(target, torch._ops.OpOverload) or target not in propagate_rules:
        raise CantChunk(
            f"Missing propagation rule for target {target}: {node.format_node()}"
        )

    status = propagate_rules[target](node)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("After propagation, the node has the following chunking meta:")
        format_node_with_chunking_meta(node, True)

    if status == PropagateStatus.FAIL:
        raise CantChunk(f"Propagate rule for {target} fail: {node.format_node()}")
    elif status == PropagateStatus.SUCCEED_WITH_CHANGE:
        # propagate to used nodes
        for arg in get_args_of_node_type(node):
            # don't propagate back thru a placeholder node
            if arg.op == "placeholder":
                if "tangent" in arg.target:  # type: ignore[operator]
                    # we have a separate pass to propagate scale_by information fwd.
                    set_chunking_meta(arg, scale_by=arg)
            elif bwd_filter[arg]:
                _enqueue(queue, arg)

        # propagate to user nodes
        if fwd_filter[node]:
            for user in node.users:
                _enqueue(queue, user)
    else:
        assert status == PropagateStatus.SUCCEED_NO_CHANGE, (
            f"status type {type(status)}, value {status}"
        )


def _bool_to_status(changed: bool) -> PropagateStatus:
    """
    Return the variant of the success flag depending on whether there is any change.
    """
    return (
        PropagateStatus.SUCCEED_WITH_CHANGE
        if changed
        else PropagateStatus.SUCCEED_NO_CHANGE
    )


@register_propagate_rule(aten.addmm.default)
def propagate_addmm(addmm_node: Node) -> _HandlerRetType:
    bias_node, input_node, weight_node = addmm_node.args

    def propagate_addmm_fwd() -> PropagateStatus:
        assert isinstance(bias_node, Node)
        assert isinstance(input_node, Node)
        assert isinstance(weight_node, Node)
        if not has_any_chunking_meta(bias_node, input_node, weight_node):
            return PropagateStatus.SUCCEED_NO_CHANGE

        # only input is chunked by dim 0
        if (
            has_nop_chunking_meta(bias_node)
            and has_nop_chunking_meta(weight_node)
            and is_chunked_by_dim(input_node, 0)
        ):
            # set a nop chunking metadata on bias_node & weight_node
            # to indicate that they should be a part of the chunking
            # subgraph. (i.e. we pass in them as placeholder node)
            return _bool_to_status(
                copy_chunking_meta(addmm_node, input_node)
                | set_chunking_meta(bias_node)
                | set_chunking_meta(weight_node)
            )
        return PropagateStatus.FAIL

    def propagate_addmm_bwd() -> PropagateStatus:
        assert isinstance(bias_node, Node)
        assert isinstance(input_node, Node)
        assert isinstance(weight_node, Node)

        if not (meta := get_chunking_meta(addmm_node)):
            return PropagateStatus.SUCCEED_NO_CHANGE

        if meta.chunked_by_dim(0):
            # if the output is chunked by the batch dimension, then
            # bias and input should as well
            changed = set_chunking_meta(input_node, meta) | set_chunking_meta(
                weight_node
            )

            # We should chunk the bias only if it's not broadcasted
            bias_node_ft = get_fake_tensor_from_node_arg(bias_node)
            input_node_ft = get_fake_tensor_from_node_arg(input_node)
            assert bias_node_ft is not None
            assert input_node_ft is not None
            if bias_node_ft.ndim < input_node_ft.ndim:
                changed |= set_chunking_meta(bias_node)
            else:
                changed |= set_chunking_meta(bias_node, meta)
            return _bool_to_status(changed)

        return PropagateStatus.FAIL

    return propagate_addmm_fwd(), propagate_addmm_bwd()


@register_propagate_rule(aten.mm.default)
def propagate_mm(mm_node: Node) -> _HandlerRetType:
    lhs_node, rhs_node = mm_node.args[:2]

    def fwd() -> PropagateStatus:
        assert isinstance(lhs_node, Node)
        assert isinstance(rhs_node, Node)
        lhs_meta = get_chunking_meta(lhs_node)
        rhs_meta = get_chunking_meta(rhs_node)

        if has_nop_chunking_meta(lhs_node) and has_nop_chunking_meta(rhs_node):
            return _bool_to_status(False)

        # only lhs is chunked
        if not has_nop_chunking_meta(lhs_node) and has_nop_chunking_meta(rhs_node):
            assert lhs_meta is not None
            return _bool_to_status(
                copy_chunking_meta(mm_node, lhs_meta) | set_chunking_meta(rhs_node)
            )

        # either lhs or rhs is chunked at the reduction dimension
        if (lhs_meta is not None and lhs_meta.chunk_dim == 1) or (
            rhs_meta is not None and rhs_meta.chunk_dim == 0
        ):
            # The output is not chunked, but need to be sum'ed up!
            return _bool_to_status(
                set_chunking_meta(mm_node, chunk_dim=None, need_sum=True)
                | update_chunking_meta(lhs_node, chunk_dim=1)
                | update_chunking_meta(rhs_node, chunk_dim=0)
            )

        return PropagateStatus.FAIL

    def bwd() -> PropagateStatus:
        assert isinstance(lhs_node, Node)
        assert isinstance(rhs_node, Node)
        out_meta = get_chunking_meta(mm_node)
        if out_meta is None:
            return _bool_to_status(False)

        # first dim of a 2D output is chunked
        ft = get_fake_tensor_from_node_arg(mm_node)
        assert ft is not None
        if ft.ndim == 2 and out_meta.chunk_dim == 0:
            rhs_meta = get_chunking_meta(rhs_node)
            assert ChunkingMeta.is_nop(rhs_meta)
            return _bool_to_status(
                copy_chunking_meta(lhs_node, mm_node) | set_chunking_meta(rhs_node)
            )

        if out_meta.need_sum:
            changed = set_chunking_meta(lhs_node, chunk_dim=1) | set_chunking_meta(
                rhs_node, chunk_dim=0
            )
            return _bool_to_status(changed)

        return PropagateStatus.FAIL

    return fwd(), bwd()


@register_propagate_rule(
    [
        prims.convert_element_type.default,
        aten.exp.default,
        aten.log.default,
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.div.Tensor,
        aten.mul.Tensor,
        prims.fma.default,
        aten.where.self,
        aten.neg.default,
    ]
)
def propagate_general_copy_metadata(
    out_node: Node, ignore_broadcast: bool = False
) -> _HandlerRetType:
    """
    A general propagation rules that basically copy around the chunking
    metadata.
    """
    node_args = get_args_of_node_type(out_node)
    node_is_scalar = get_node_is_scalar(node_args)
    node_ndim = get_node_ndim(node_args)

    scalar_args = [node for node in node_args if node_is_scalar[node]]
    non_scalar_args = [node for node in node_args if not node_is_scalar[node]]

    out_ndim = out_node.meta["val"].ndim

    # This general rule only allow scalar tensors without chunking meta
    if scalar_args and not all(
        ChunkingMeta.is_nop(get_chunking_meta(arg)) for arg in scalar_args
    ):
        return PropagateStatus.FAIL

    def _chunk_broadcasted_tensor(chunk_dim: int) -> bool:
        for node in non_scalar_args:
            if node_ndim[node] != out_ndim and chunk_dim >= out_ndim - node_ndim[node]:
                return True
        return False

    def propagate_fwd() -> PropagateStatus:
        if len(node_args) == 0:
            return PropagateStatus.FAIL

        first_meta = get_first_chunking_meta(*non_scalar_args)
        if first_meta is None:
            return _bool_to_status(False)

        need_handle_broadcast = (
            not ignore_broadcast and first_meta.chunk_dim is not None
        )
        if (
            need_handle_broadcast
            and first_meta.chunk_dim is not None
            and _chunk_broadcasted_tensor(first_meta.chunk_dim)
        ):
            # We don't chunking a broadcasted tensor for now.
            # Can add the rule if such a use case come up
            return PropagateStatus.FAIL

        changed = set_chunking_meta_if_none(
            non_scalar_args, first_meta, lambda node: node_ndim[node] != out_ndim
        )

        for other_node in non_scalar_args:
            other_meta = get_chunking_meta(other_node)

            if need_handle_broadcast and node_ndim[other_node] != out_ndim:
                if not ChunkingMeta.is_nop(other_meta):
                    return PropagateStatus.FAIL
            else:
                if other_meta != first_meta:
                    return PropagateStatus.FAIL

        changed |= copy_chunking_meta(out_node, first_meta)
        return _bool_to_status(changed)

    def propagate_bwd() -> PropagateStatus:
        if not (meta := get_chunking_meta(out_node)):
            return PropagateStatus.SUCCEED_NO_CHANGE

        need_handle_broadcast = not ignore_broadcast and meta.chunk_dim is not None
        if (
            need_handle_broadcast
            and meta.chunk_dim is not None
            and _chunk_broadcasted_tensor(meta.chunk_dim)
        ):
            return PropagateStatus.FAIL

        # apply any to a list to avoid short-circuit
        changed = any(  # noqa: C419
            [  # noqa: C419
                copy_chunking_meta(node, meta)
                if not need_handle_broadcast or node_ndim[node] == out_ndim
                else set_chunking_meta(node)
                for node in non_scalar_args
            ]
        )

        # [NOTE: NOP Chunking metadata]
        # For scalar node arguments, we add a nop ChunkingMeta so the
        # propagation continues. This is mainly needed to reach the point
        # where we attach chunking metadata to tangents that need to be
        # included in the chunking subgraph.
        # This is different to having a None ChunkingMeta
        changed |= any(  # noqa: C419
            [  # noqa: C419
                set_chunking_meta(node)
                for node in scalar_args
                if get_chunking_meta(node) is None
            ]
        )

        return _bool_to_status(changed)

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(
    [
        aten.squeeze.dim,
        aten.gather.default,
        aten.scatter.value,
    ]
)
def propagate_general_copy_metadata_ignore_broadcast(out_node: Node) -> _HandlerRetType:
    return propagate_general_copy_metadata(out_node, ignore_broadcast=True)  # type: ignore[call-arg]


@register_propagate_rule(
    [
        aten.amax.default,
        aten.sum.dim_IntList,
    ]
)
def propagate_reduce(reduce_node: Node) -> _HandlerRetType:
    arg_node, reduce_dims = reduce_node.args[0:2]

    def propagate_fwd() -> PropagateStatus:
        assert isinstance(arg_node, Node)
        assert isinstance(reduce_dims, (tuple, list))
        arg_meta = get_chunking_meta(arg_node)
        if arg_meta is None:
            return PropagateStatus.SUCCEED_NO_CHANGE
        if arg_meta.chunk_dim not in reduce_dims:
            # Reduce across the non chunked dimension
            return _bool_to_status(copy_chunking_meta(reduce_node, arg_meta))

        # sum across the chunked dimension. E.g. happens for computing
        # the gradient of bias for an addmm
        if reduce_node.target == aten.sum.dim_IntList and list(reduce_dims) == [
            arg_meta.chunk_dim
        ]:
            return _bool_to_status(
                set_chunking_meta(reduce_node, arg_meta, chunk_dim=None, need_sum=True)
            )

        return PropagateStatus.FAIL

    def propagate_bwd() -> PropagateStatus:
        assert isinstance(arg_node, Node)
        assert isinstance(reduce_dims, (tuple, list))
        out_meta = get_chunking_meta(reduce_node)
        if out_meta is None:
            return PropagateStatus.SUCCEED_NO_CHANGE
        if out_meta.chunk_dim is not None:
            assert out_meta.chunk_dim not in reduce_dims
            return _bool_to_status(copy_chunking_meta(arg_node, out_meta))

        if out_meta.chunk_dim is None and out_meta.need_sum and len(reduce_dims) == 1:
            assert reduce_node.target == aten.sum.dim_IntList
            return _bool_to_status(
                set_chunking_meta(
                    arg_node, out_meta, chunk_dim=reduce_dims[0], need_sum=False
                )
            )

        return PropagateStatus.FAIL

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(aten.permute.default)
def propagate_permute(permute_node: Node) -> _HandlerRetType:
    input_node, order = permute_node.args[:2]

    def propagate_fwd() -> PropagateStatus:
        assert isinstance(input_node, Node)
        assert isinstance(order, (tuple, list))
        input_meta = get_chunking_meta(input_node)
        output_meta = get_chunking_meta(permute_node)
        if input_meta is None:
            return _bool_to_status(False)

        if input_meta.chunk_dim is None:
            return PropagateStatus.FAIL

        orig_chunk_dim = input_meta.chunk_dim
        reverse_lookup: dict[int, int] = {v: k for k, v in enumerate(order)}
        new_chunk_dim = reverse_lookup[orig_chunk_dim]

        # sanity check
        if output_meta is not None:
            assert output_meta.chunk_dim == new_chunk_dim
        return _bool_to_status(
            set_chunking_meta(permute_node, meta=input_meta, chunk_dim=new_chunk_dim)
        )

    def propagate_bwd() -> PropagateStatus:
        assert isinstance(input_node, Node)
        assert isinstance(order, (tuple, list))

        input_meta = get_chunking_meta(input_node)
        output_meta = get_chunking_meta(permute_node)
        if output_meta is None:
            return _bool_to_status(False)

        if output_meta.chunk_dim is None:
            return PropagateStatus.FAIL

        orig_chunk_dim = output_meta.chunk_dim

        new_chunk_dim = order[orig_chunk_dim]

        # sanity check
        if input_meta is not None:
            assert input_meta.chunk_dim == new_chunk_dim
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_chunk_dim)
        )

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(
    [
        aten.full.default,  # nop since there is no inputs for fwd/bwd metadata propagation
    ]
)
def propagate_nop(node: Node) -> _HandlerRetType:
    def fwd() -> PropagateStatus:
        return _bool_to_status(False)

    def bwd() -> PropagateStatus:
        return _bool_to_status(False)

    return fwd(), bwd()


@register_propagate_rule(
    [
        aten.expand.default,
    ]
)
def propagate_expand(expand_node: Node) -> _HandlerRetType:
    input_node = expand_node.args[0]
    assert isinstance(input_node, Node)

    if input_node.meta["val"].numel() != 1:
        return PropagateStatus.FAIL

    # Combined fwd/bwd rule
    output_meta = get_chunking_meta(expand_node)
    if output_meta is None:
        return _bool_to_status(False)

    return _bool_to_status(set_chunking_meta(input_node))


@register_propagate_rule(
    [
        aten.sum.default,
    ]
)
def propagate_sum_to_scalar(sum_node: Node) -> _HandlerRetType:
    input_node = sum_node.args[0]
    assert isinstance(input_node, Node)

    def fwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)  # pyrefly: ignore[bad-argument-type]
        if has_nop_chunking_meta(input_node):  # pyrefly: ignore[bad-argument-type]
            return _bool_to_status(False)

        assert input_meta is not None
        if input_meta.chunk_dim is not None:
            changed = update_chunking_meta(sum_node, need_sum=True, chunk_by=None)
            return _bool_to_status(changed)
        return PropagateStatus.FAIL

    def bwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)  # pyrefly: ignore[bad-argument-type]
        if has_nop_chunking_meta(sum_node):
            return _bool_to_status(False)

        # We won't know how the input is chunked if sum_node.meta.need_sum is True.
        # On the other hand, sum_node.meta.need_sum is True can only happen
        # by propagating from input_node. Doing sanity check here is fine
        # since the input_node.meta should have already been properly
        # setup.
        output_meta = get_chunking_meta(sum_node)
        if (
            output_meta is not None
            and output_meta.need_sum
            and input_meta is not None
            and input_meta.chunk_dim is not None
        ):
            return _bool_to_status(False)

        return PropagateStatus.FAIL

    return fwd(), bwd()
