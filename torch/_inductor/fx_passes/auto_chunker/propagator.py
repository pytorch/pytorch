import functools
import operator
import logging
import math
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
        if not (isinstance(fwd_bwd_status, (list, tuple)) and len(fwd_bwd_status) == 2):
            raise AssertionError(
                f"expected (list, tuple) of len 2, got {type(fwd_bwd_status)}"
            )
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

    if not isinstance(aten_op, (list, tuple)):
        raise AssertionError(f"{type(aten_op)=}")
    for op in aten_op:
        if not isinstance(op, torch._ops.OpOverload):
            raise AssertionError(f"expected OpOverload, got {type(op)}")
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
    An amplified node means a node with the same numel as `amplifier_node`
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

    if len(get_nodes_with_chunking_meta(graph)) != 0:
        raise AssertionError("Expect no nodes with chunking meta yet")

    set_chunking_meta(amplifier_node, chunk_dim=0)

    queue: Queue[Node] = Queue()
    _enqueue(queue, amplifier_node)
    # Promoted view amplifiers may refuse bwd into the pre-split matmul, which
    # would otherwise yield SUCCEED_NO_CHANGE and never enqueue downstream users.
    if fwd_filter[amplifier_node]:
        for user in amplifier_node.users:
            _enqueue(queue, user)

    while not queue.empty():
        propagate_single_node(queue, fwd_filter, bwd_filter, queue.get())

    nodes_with_chunking_meta = get_nodes_with_chunking_meta(graph)
    propagate_scale_by(nodes_with_chunking_meta)

    if log.isEnabledFor(logging.DEBUG):
        print("All nodes with chunking metadata set:")
        for node in nodes_with_chunking_meta:
            format_node_with_chunking_meta(node)



def propagate_getitem(node: Node) -> PropagateStatus:
    """Copy chunking meta through tuple/list indexing (e.g. nll_loss outputs)."""
    seq, _idx = node.args[0], node.args[1]
    if not isinstance(seq, Node):
        return PropagateStatus.FAIL

    def fwd() -> PropagateStatus:
        meta = get_chunking_meta(seq)
        if meta is None:
            return _bool_to_status(False)
        return _bool_to_status(copy_chunking_meta(node, meta))

    def bwd() -> PropagateStatus:
        meta = get_chunking_meta(node)
        if meta is None:
            return _bool_to_status(False)
        return _bool_to_status(copy_chunking_meta(seq, meta))

    fwd_status, bwd_status = fwd(), bwd()
    if fwd_status == PropagateStatus.FAIL or bwd_status == PropagateStatus.FAIL:
        return PropagateStatus.FAIL
    if (
        fwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
        or bwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
    ):
        return PropagateStatus.SUCCEED_WITH_CHANGE
    return PropagateStatus.SUCCEED_NO_CHANGE


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

    if target is operator.getitem:
        status = propagate_getitem(node)
    elif not isinstance(target, torch._ops.OpOverload) or target not in propagate_rules:
        raise CantChunk(
            f"Missing propagation rule for target {target}: {node.format_node()}"
        )
    else:
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
        if status != PropagateStatus.SUCCEED_NO_CHANGE:
            raise AssertionError(f"status type {type(status)}, value {status}")


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
        if not isinstance(bias_node, Node):
            raise AssertionError(
                f"expected bias_node to be a Node, got {type(bias_node)}"
            )
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(weight_node, Node):
            raise AssertionError(
                f"expected weight_node to be a Node, got {type(weight_node)}"
            )
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
        if not isinstance(bias_node, Node):
            raise AssertionError(
                f"expected bias_node to be a Node, got {type(bias_node)}"
            )
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(weight_node, Node):
            raise AssertionError(
                f"expected weight_node to be a Node, got {type(weight_node)}"
            )

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
            if bias_node_ft is None:
                raise AssertionError("expected bias_node_ft to not be None")
            if input_node_ft is None:
                raise AssertionError("expected input_node_ft to not be None")
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
        if not isinstance(lhs_node, Node):
            raise AssertionError(
                f"expected lhs_node to be a Node, got {type(lhs_node)}"
            )
        if not isinstance(rhs_node, Node):
            raise AssertionError(
                f"expected rhs_node to be a Node, got {type(rhs_node)}"
            )
        lhs_meta = get_chunking_meta(lhs_node)
        rhs_meta = get_chunking_meta(rhs_node)

        if has_nop_chunking_meta(lhs_node) and has_nop_chunking_meta(rhs_node):
            return _bool_to_status(False)

        # only lhs is chunked on its batch dim (dim 0). lhs chunk_dim==1 is the
        # reduction-dim case handled below (need_sum); None-meta rhs looks like
        # nop and must not steal that path.
        if (
            lhs_meta is not None
            and lhs_meta.chunk_dim == 0
            and has_nop_chunking_meta(rhs_node)
        ):
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
        if not isinstance(lhs_node, Node):
            raise AssertionError(
                f"expected lhs_node to be a Node, got {type(lhs_node)}"
            )
        if not isinstance(rhs_node, Node):
            raise AssertionError(
                f"expected rhs_node to be a Node, got {type(rhs_node)}"
            )
        out_meta = get_chunking_meta(mm_node)
        if out_meta is None:
            return _bool_to_status(False)

        # first dim of a 2D output is chunked
        ft = get_fake_tensor_from_node_arg(mm_node)
        if ft is None:
            raise AssertionError("expected ft to not be None")
        if ft.ndim == 2 and out_meta.chunk_dim == 0:
            rhs_meta = get_chunking_meta(rhs_node)
            if not ChunkingMeta.is_nop(rhs_meta):
                raise AssertionError("expected rhs_meta to be a nop ChunkingMeta")
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
        aten.tanh.default,
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.div.Tensor,
        aten.mul.Tensor,
        aten.where.self,
        aten.neg.default,
        aten.eq.Tensor,
        aten.alias.default,
        aten._softmax.default,
        aten._log_softmax.default,
        aten._log_softmax_backward_data.default,
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
            # We don't chunk a broadcasted tensor for now.
            # Can add the rule if such a use case comes up
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
        changed = any(
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
        changed |= any(
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
        aten.scatter_add.default,
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
        if not isinstance(arg_node, Node):
            raise AssertionError(
                f"expected arg_node to be a Node, got {type(arg_node)}"
            )
        if not isinstance(reduce_dims, (tuple, list)):
            raise AssertionError(
                f"expected reduce_dims to be a tuple or list, got {type(reduce_dims)}"
            )
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
        if not isinstance(arg_node, Node):
            raise AssertionError(
                f"expected arg_node to be a Node, got {type(arg_node)}"
            )
        if not isinstance(reduce_dims, (tuple, list)):
            raise AssertionError(
                f"expected reduce_dims to be a tuple or list, got {type(reduce_dims)}"
            )
        out_meta = get_chunking_meta(reduce_node)
        if out_meta is None:
            return PropagateStatus.SUCCEED_NO_CHANGE
        if out_meta.chunk_dim is not None:
            if out_meta.chunk_dim in reduce_dims:
                raise AssertionError(
                    f"expected chunk_dim {out_meta.chunk_dim} not in {reduce_dims}"
                )
            return _bool_to_status(copy_chunking_meta(arg_node, out_meta))

        if out_meta.chunk_dim is None and out_meta.need_sum and len(reduce_dims) == 1:
            if reduce_node.target != aten.sum.dim_IntList:
                raise AssertionError(
                    f"expected target aten.sum.dim_IntList, got {reduce_node.target}"
                )
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
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(order, (tuple, list)):
            raise AssertionError(
                f"expected order to be a tuple or list, got {type(order)}"
            )
        input_meta = get_chunking_meta(input_node)
        output_meta = get_chunking_meta(permute_node)
        if input_meta is None:
            return _bool_to_status(False)

        if input_meta.chunk_dim is None:
            return PropagateStatus.FAIL

        orig_chunk_dim = input_meta.chunk_dim
        # pyrefly: ignore [bad-argument-type, bad-assignment]
        reverse_lookup: dict[int, int] = {v: k for k, v in enumerate(order)}
        new_chunk_dim = reverse_lookup[orig_chunk_dim]

        # sanity check
        if output_meta is not None:
            if output_meta.chunk_dim != new_chunk_dim:
                raise AssertionError(
                    f"expected chunk_dim {new_chunk_dim}, got {output_meta.chunk_dim}"
                )
        return _bool_to_status(
            set_chunking_meta(permute_node, meta=input_meta, chunk_dim=new_chunk_dim)
        )

    def propagate_bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(order, (tuple, list)):
            raise AssertionError(
                f"expected order to be a tuple or list, got {type(order)}"
            )

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
            if input_meta.chunk_dim != new_chunk_dim:
                raise AssertionError(
                    f"expected chunk_dim {new_chunk_dim}, got {input_meta.chunk_dim}"
                )
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


@register_propagate_rule(aten.unsqueeze.default)
def propagate_unsqueeze(unsqueeze_node: Node) -> _HandlerRetType:
    input_node, unsqueeze_dim = unsqueeze_node.args[:2]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )
    if not isinstance(unsqueeze_dim, int):
        raise AssertionError(
            f"expected unsqueeze_dim to be an int, got {type(unsqueeze_dim)}"
        )
    input_ndim = get_fake_tensor_from_node_arg(input_node).ndim  # type: ignore[union-attr]
    # Normalize negative dim: unsqueeze valid range is [-(ndim+1), ndim]
    normalized_dim = (
        unsqueeze_dim + input_ndim + 1 if unsqueeze_dim < 0 else unsqueeze_dim
    )

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        if input_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(unsqueeze_node, input_meta))

        # pyrefly: ignore[unsupported-operation]
        new_dim = input_meta.chunk_dim + (
            1 if input_meta.chunk_dim >= normalized_dim else 0
        )
        return _bool_to_status(
            set_chunking_meta(unsqueeze_node, meta=input_meta, chunk_dim=new_dim)
        )

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(unsqueeze_node)
        if output_meta is None:
            return _bool_to_status(False)
        if output_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(input_node, output_meta))
        # pyrefly: ignore[unsupported-operation]
        new_dim = output_meta.chunk_dim - (
            1 if output_meta.chunk_dim > normalized_dim else 0
        )
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_dim)
        )

    return fwd(), bwd()


def _find_chunk_dim_after_reshape(
    old_shape: Sequence[int], new_shape: Sequence[int], chunk_dim: int
) -> int | None:
    """
    Find the equivalent chunk_dim position after a reshape by matching
    the prefix product (number of elements before the dimension) and
    the dimension size.

    Besides preserving the chunk dim exactly, also allow:
      - split: [M, N] -> [M1, M2, N] with M1*M2=M, chunk_dim=0 -> 0
        (map to the leading factor; applier rewrites view shapes by numel)
      - merge: [M1, M2, N] -> [M, N] with M1*M2=M, chunk_dim=0 -> 0
        (chunk_dim must be the leading dim of the merged group)

    Returns None if the chunk dimension is rearranged unsafely.
    """
    chunk_size = old_shape[chunk_dim]
    old_prefix = math.prod(old_shape[:chunk_dim])
    old_suffix = math.prod(old_shape[chunk_dim + 1 :])

    # Exact preserve: same prefix and same extent.
    new_prefix = 1
    for new_dim, new_size in enumerate(new_shape):
        if (
            new_prefix == old_prefix
            and new_size == chunk_size
            and math.prod(new_shape[new_dim + 1 :]) == old_suffix
        ):
            return new_dim
        new_prefix *= new_size

    # Split one chunk dim into consecutive new dims; map to the leading one.
    new_prefix = 1
    for start in range(len(new_shape)):
        if new_prefix == old_prefix:
            prod = 1
            for end in range(start, len(new_shape)):
                prod *= new_shape[end]
                if prod == chunk_size and math.prod(new_shape[end + 1 :]) == old_suffix:
                    return start
                if prod > chunk_size:
                    break
        new_prefix *= new_shape[start]

    # Merge consecutive old dims starting at chunk_dim into one new dim.
    new_prefix = 1
    for new_dim, new_size in enumerate(new_shape):
        if new_prefix == old_prefix:
            prod = 1
            for end in range(chunk_dim, len(old_shape)):
                prod *= old_shape[end]
                if prod == new_size and math.prod(old_shape[end + 1 :]) == math.prod(
                    new_shape[new_dim + 1 :]
                ):
                    return new_dim
                if prod > new_size:
                    break
        new_prefix *= new_size

    return None


@register_propagate_rule(aten.view.default)
def propagate_view(view_node: Node) -> _HandlerRetType:
    input_node = view_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )
    input_shape = list(get_fake_tensor_from_node_arg(input_node).shape)  # type: ignore[union-attr]
    output_shape = list(get_fake_tensor_from_node_arg(view_node).shape)  # type: ignore[union-attr]

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        if input_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(view_node, input_meta))
        new_dim = _find_chunk_dim_after_reshape(
            input_shape, output_shape, input_meta.chunk_dim
        )
        if new_dim is None:
            return PropagateStatus.FAIL
        return _bool_to_status(
            set_chunking_meta(view_node, meta=input_meta, chunk_dim=new_dim)
        )

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(view_node)
        if output_meta is None:
            return _bool_to_status(False)
        if output_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(input_node, output_meta))
        new_dim = _find_chunk_dim_after_reshape(
            output_shape, input_shape, output_meta.chunk_dim
        )
        if new_dim is None:
            return PropagateStatus.FAIL
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_dim)
        )

    return fwd(), bwd()


@register_propagate_rule(
    [
        aten.expand.default,
    ]
)
def propagate_expand(expand_node: Node) -> _HandlerRetType:
    input_node = expand_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )

    input_ft = get_fake_tensor_from_node_arg(input_node)
    if input_ft is None:
        raise AssertionError("expected input_ft to not be None")
    output_ft = get_fake_tensor_from_node_arg(expand_node)
    if output_ft is None:
        raise AssertionError("expected output_ft to not be None")
    input_shape = list(input_ft.shape)
    output_shape = list(output_ft.shape)

    if input_ft.numel() == 1:
        # Scalar input: combined fwd/bwd rule
        output_meta = get_chunking_meta(expand_node)
        if output_meta is None:
            return _bool_to_status(False)
        return _bool_to_status(set_chunking_meta(input_node))

    # How many leading dims are added by expand
    dim_offset = len(output_shape) - len(input_shape)

    def is_expand_dim(out_dim: int) -> bool:
        """Check if out_dim is a broadcast dimension (newly added or size 1 in input)."""
        return out_dim < dim_offset or input_shape[out_dim - dim_offset] == 1

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        # Fail if chunk_dim is an expand dimension (input size 1 broadcast to larger size)
        if input_meta.chunk_dim is not None and is_expand_dim(
            input_meta.chunk_dim + dim_offset
        ):
            return PropagateStatus.FAIL
        return _bool_to_status(copy_chunking_meta(expand_node, input_meta))

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(expand_node)
        if output_meta is None:
            return _bool_to_status(False)
        # Fail if chunk_dim is an expand dimension (input size 1 broadcast to larger size)
        if output_meta.chunk_dim is not None and is_expand_dim(output_meta.chunk_dim):
            return PropagateStatus.FAIL
        return _bool_to_status(copy_chunking_meta(input_node, output_meta))

    return fwd(), bwd()


@register_propagate_rule(aten.nll_loss_forward.default)
def propagate_nll_loss_forward(nll_node: Node) -> _HandlerRetType:
    """
    CrossEntropy/NLL with Mean/Sum reduction collapses the chunked batch dim
    to a scalar, same as aten.sum.default.
    """
    input_node = nll_node.args[0]
    reduction = nll_node.args[3] if len(nll_node.args) > 3 else 1
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )

    def fwd() -> PropagateStatus:
        # Reduction: 0=None, 1=Mean, 2=Sum
        if reduction == 0:
            return PropagateStatus.FAIL
        input_meta = get_chunking_meta(input_node)
        if has_nop_chunking_meta(input_node):
            return _bool_to_status(False)
        if input_meta is None:
            raise AssertionError("expected input_meta to not be None")
        if input_meta.chunk_dim is not None:
            changed = set_chunking_meta(nll_node, need_sum=True, chunk_dim=None)
            # target is indexed by the same collapsed batch as input; chunk it.
            target = nll_node.args[1] if len(nll_node.args) > 1 else None
            if isinstance(target, Node):
                changed = (
                    set_chunking_meta(target, chunk_dim=input_meta.chunk_dim)
                    or changed
                )
            # weight is optional / not batch-chunked.
            weight = nll_node.args[2] if len(nll_node.args) > 2 else None
            if isinstance(weight, Node) and get_chunking_meta(weight) is None:
                changed = set_chunking_meta(weight) or changed
            # nll_loss_forward returns a tuple, so fwd_filter treats it as
            # unreachable and will not enqueue getitem users. Mark the scalar
            # getitems here so the tuple node stays subgraph body and only the
            # tensor getitems become need_sum accumulators.
            for user in nll_node.users:
                if user.target is operator.getitem:
                    changed = (
                        set_chunking_meta(user, need_sum=True, chunk_dim=None)
                        or changed
                    )
            return _bool_to_status(changed)
        return PropagateStatus.FAIL

    def bwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)
        if has_nop_chunking_meta(nll_node):
            return _bool_to_status(False)
        output_meta = get_chunking_meta(nll_node)
        if (
            output_meta is not None
            and output_meta.need_sum
            and input_meta is not None
            and input_meta.chunk_dim is not None
        ):
            return _bool_to_status(False)
        return PropagateStatus.FAIL

    return fwd(), bwd()


@register_propagate_rule(aten.nll_loss_backward.default)
def propagate_nll_loss_backward(nll_bwd_node: Node) -> _HandlerRetType:
    """
    grad_input has the same batch layout as the forward logits input.
    """
    grad_output = nll_bwd_node.args[0]
    input_node = nll_bwd_node.args[1]

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            return PropagateStatus.FAIL
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        if input_meta.chunk_dim is None:
            return PropagateStatus.FAIL
        changed = copy_chunking_meta(nll_bwd_node, input_meta)
        # target shares the collapsed batch with input; weight/total_weight do not.
        target = nll_bwd_node.args[2] if len(nll_bwd_node.args) > 2 else None
        if isinstance(target, Node):
            changed = (
                set_chunking_meta(target, chunk_dim=input_meta.chunk_dim) or changed
            )
        for arg in nll_bwd_node.args[3:]:
            if isinstance(arg, Node) and get_chunking_meta(arg) is None:
                changed = set_chunking_meta(arg) or changed
        return _bool_to_status(changed)

    def bwd() -> PropagateStatus:
        out_meta = get_chunking_meta(nll_bwd_node)
        if out_meta is None:
            return _bool_to_status(False)
        changed = False
        if isinstance(input_node, Node):
            changed = copy_chunking_meta(input_node, out_meta) or changed
        if isinstance(grad_output, Node) and out_meta.chunk_dim is not None:
            changed = (
                set_chunking_meta(grad_output, need_sum=True, chunk_dim=None)
                or changed
            )
        return _bool_to_status(changed)

    return fwd(), bwd()


@register_propagate_rule(
    [
        aten.sum.default,
    ]
)
def propagate_sum_to_scalar(sum_node: Node) -> _HandlerRetType:
    input_node = sum_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )

    def fwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)  # pyrefly: ignore[bad-argument-type]
        if has_nop_chunking_meta(input_node):  # pyrefly: ignore[bad-argument-type]
            return _bool_to_status(False)

        if input_meta is None:
            raise AssertionError("expected input_meta to not be None")
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
