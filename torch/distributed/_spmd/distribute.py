import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx

from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger

from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor._op_schema import OpSchema
from torch.distributed._tensor._redistribute import redistribute_local_tensor
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten


logger: Optional[logging.Logger] = None

aten = torch.ops.aten


class TrainingPhase(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass
class Schema:
    mesh: DeviceMesh
    placements: List[Placement]


@dataclass
class DSymInt:
    """DSymInt represents a value retrieved by a SymInt op from a DTensor.

    DSymInt helps View and Factory ops to determine the placement and shape of the
    output tensor, as those operators either do not have an input DTensor or
    the input DTensor is insufficient to determine the output tensor's placement.
    """

    global_value: int  # value that the SymInt evaluates to
    local_value: int  # vaue that this SymInt evaluates to on the local shard
    mesh: DeviceMesh  # device mesh of the DTensor where this SymInt is retrieved from

    def is_shard(self) -> bool:
        return self.local_value != self.global_value

    @classmethod
    def from_node(cls, node: fx.Node, dtensor: DTensor) -> "DSymInt":
        dim: int = 0
        if node.target == aten.sym_size:
            dim = cast(int, node.args[1])
            return cls(
                global_value=dtensor.size(dim),
                local_value=dtensor.to_local().size(dim),
                mesh=dtensor.device_mesh,
            )
        elif node.target == aten.sym_numel:
            return cls(
                global_value=dtensor.numel(),
                local_value=dtensor.to_local().numel(),
                mesh=dtensor.device_mesh,
            )
        elif node.target == aten.sym_stride:
            dim = cast(int, node.args[1])
            return cls(
                global_value=dtensor.stride(dim),
                local_value=dtensor.to_local().stride(dim),
                mesh=dtensor.device_mesh,
            )
        else:
            raise NotImplementedError(f"DSymInt does not support {node.target}")


def _is_partial_dtensor(obj: Any) -> bool:
    """Check if object is 1) DTensor and  2) with any placement of _Partial."""
    if not isinstance(obj, DTensor):
        return False

    is_partial = False
    for placement in obj.placements:
        if isinstance(placement, _Partial):
            is_partial = True
            break

    return is_partial


def _dispatch_with_local_tensors(
    op: torch._ops.OpOverload,
    local_args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    specs: Optional[
        Dict[
            torch.Tensor,
            Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]],
        ]
    ] = None,
) -> Any:
    if kwargs is None:
        kwargs = {}
    if specs is None:
        specs = {}

    def redistribute(arg: Any) -> Any:
        tensor_shape, mesh, current_placement, target_placement = specs[arg]
        tensor_meta = TensorMeta(
            tensor_shape,
            stride=arg.stride(),
            dtype=arg.dtype,
        )
        current_spec = DTensorSpec(
            mesh, tuple(current_placement), tensor_meta=tensor_meta
        )
        target_spec = DTensorSpec(
            mesh, tuple(target_placement), tensor_meta=tensor_meta
        )

        return (
            redistribute_local_tensor(arg, current_spec, target_spec)  # type: ignore[index]
            if isinstance(arg, torch.Tensor) and arg in specs  # type: ignore[operator]
            else arg
        )

    # TODO: this is broken because it won't redistributed potential tensors on the kwargs
    return op(*tree_map(redistribute, local_args), **kwargs)


# Figure out how to specify a type spec for the return specs value
# without the entire structure.
# pyre-fixme
def _update_specs_for_redistribute(args, target_schema, redistribute):
    # Code adapted from pack_args_kwargs_with_local_tensor
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema = pytree.tree_leaves(target_schema.args_schema)

    specs: Dict[
        torch.Tensor,
        Tuple[
            torch.Size,
            DeviceMesh,
            Sequence[Placement],
            Sequence[Placement],
        ],
    ] = {}
    for i, arg in enumerate(flatten_args):
        if isinstance(arg, DTensor):
            if redistribute:
                specs[arg._local_tensor] = (
                    arg.size(),
                    flatten_args_schema[i].mesh,
                    arg.placements,
                    flatten_args_schema[i].placements,
                )
            flatten_args_schema[i] = arg._local_tensor

    unflattened_args = tree_unflatten(flatten_args_schema, args_tree_spec)
    return specs, unflattened_args


# When no tensor redistribution is required, we only need to update non-tensor args
# of the node according to op_schema and avoid building a GraphModule just for the
# node.
def _update_node_from_op_schema(node: torch.fx.Node, op_schema: OpSchema) -> None:
    flat_args, args_tree_spec = tree_flatten(node.args)
    flat_args_schema = pytree.tree_leaves(op_schema.args_schema)

    def is_sym_int_or_int(arg: Union[int, torch.fx.Node]) -> bool:
        if isinstance(arg, torch.fx.Node):
            return arg.target in [
                aten.sym_size,
                aten.sym_numel,
                aten.sym_stride,
            ]
        return isinstance(arg, int)

    assert len(flat_args) == len(flat_args_schema)
    for i, (arg, arg_schema) in enumerate(zip(flat_args, flat_args_schema)):
        if is_sym_int_or_int(arg) and isinstance(arg_schema, int):
            flat_args[i] = arg_schema

    args = tree_unflatten(flat_args, args_tree_spec)
    for idx, arg in enumerate(args):
        node.update_arg(idx, arg)
    return None


def _remap_arg(node_to_obj: Dict[fx.Node, Any], arg: Any) -> Any:
    if isinstance(arg, torch.fx.Node):
        obj = node_to_obj[arg]
        if _get_tracer():
            # This is a shared arg, already has a tracer from previous
            # tracing. Delete the tracer.
            del cast(Dict[Any, Any], obj.__dict__)[proxy_slot]
        return obj
    else:
        return arg


def unpack_sizes_and_dims(
    sizes: List[Union[DSymInt, int]], mesh: DeviceMesh
) -> Tuple[List[int], List[Placement]]:
    local_sizes: List[int] = [
        s.local_value if isinstance(s, DSymInt) else s for s in sizes
    ]
    placements: List[Placement] = [
        Shard(i)
        for i, a in enumerate(sizes)
        if (isinstance(a, DSymInt) and a.is_shard())
    ] or [Replicate()]

    assert len(placements) == mesh.ndim, (
        f"The number of sharded dimensions ({len(placements)}) must "
        f"match number of dimensions in device mesh ({mesh.ndim})."
    )

    return local_sizes, placements


def binop_sym_int_consumer_rule(node: fx.Node, args: Tuple[Any, ...]) -> DTensor:
    assert len(args) == 2, f"Expect two args but got op {node.target} with args {args}"
    assert isinstance(
        args[0], DTensor
    ), f"Expect 1st argument to be DTensor but got {args[0]}"
    assert isinstance(args[1], list), f"Expect 2nd argument as list but got {args[1]}"

    # extract sharded dimensions in the size list, the output DTensor should
    # follow these placements.
    local_sizes, placements = unpack_sizes_and_dims(args[1], args[0].device_mesh)

    # set node args to real int sizes.
    node.args = (node.args[0], local_sizes)
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(
        local_tensor=op(args[0]._local_tensor, local_sizes),
        device_mesh=args[0].device_mesh,
        placements=placements,
        run_check=False,
    )


def slice_backwad_sym_int_consumer_rule(
    node: fx.Node, args: Tuple[Any, ...]
) -> DTensor:
    grad_output, input_sizes, dim, start, end, step = args

    local_sizes: List[int] = [
        s.local_value if isinstance(s, DSymInt) else s for s in input_sizes
    ]

    input_tensor = torch.zeros(
        local_sizes, device=grad_output.device, dtype=grad_output.dtype
    )
    return DTensor.from_local(
        local_tensor=torch.slice_scatter(
            input_tensor, grad_output.to_local(), dim, start, end, step
        ),
        device_mesh=grad_output.device_mesh,
        placements=grad_output.placements,
        run_check=False,
    )


def factory_with_sizes_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    flat_args = pytree.arg_tree_leaves(*args)
    assert not any(isinstance(a, DTensor) for a in flat_args), (
        f"Not expect DTensor argument for factory op, but got {node.target} "
        f"with arguments {args}."
    )
    assert isinstance(args[0], list), f"Expect 2nd argument as list but got {args[1]}"

    local_sizes, placements = unpack_sizes_and_dims(args[0], default_mesh)
    node.args = (local_sizes, *args[1:])
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(
        local_tensor=op(*node.args, **kwargs),
        device_mesh=default_mesh,
        placements=placements,
        run_check=False,
    )


def factory_arange_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    node.args = tree_map(lambda a: a.local_value if isinstance(a, DSymInt) else a, args)
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(
        local_tensor=op(*node.args, **kwargs),
        device_mesh=default_mesh,
        placements=[Replicate()],
        run_check=False,
    )


def default_factory_op_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    node.args, node.kwargs = args, kwargs
    op = cast(torch._ops.OpOverload, node.target)
    return DTensor.from_local(
        local_tensor=op(*node.args, **node.kwargs),
        device_mesh=default_mesh,
        placements=[Replicate()],
        run_check=False,
    )


# Dispatch override for view and factory ops that consume SymInt arguments,
# where the output spec should follow dimension placement where the SymInt comes
# from.
VIEW_SYM_INT_CONSUMERS: Dict[torch._ops.OpOverload, Callable] = {
    aten._unsafe_view.default: binop_sym_int_consumer_rule,
    aten.expand.default: binop_sym_int_consumer_rule,
    aten.slice_backward.default: slice_backwad_sym_int_consumer_rule,
    aten.view.default: binop_sym_int_consumer_rule,
}

FACTORY_SYM_INT_CONSUMERS: Dict[torch._ops.OpOverload, Callable] = {
    aten.full.default: factory_with_sizes_rule,
    aten.arange.default: factory_arange_rule,
    aten.arange.start: factory_arange_rule,
}


# Dispatch override for factory ops, as DTensor cannot propagate sharding spec
# without DTensor inputs.
FACTORY_OPS: Dict[torch._ops.OpOverload, Callable] = {
    aten.scalar_tensor.default: default_factory_op_rule,
    aten.arange.start: default_factory_op_rule,
    aten.zeros.default: default_factory_op_rule,
}


def _get_dtensor_dispatch_graph(
    node: fx.Node,
    node_to_obj: Dict[fx.Node, Any],
    *,
    force_make_fx: bool = False,
    default_mesh: Optional[DeviceMesh] = None,
) -> Optional[fx.GraphModule]:
    with torch.no_grad():
        # Args should be a list of objects post remapping.
        args = tree_map(partial(_remap_arg, node_to_obj), node.args)
        kwargs = tree_map(partial(_remap_arg, node_to_obj), node.kwargs)

        op_overload = cast(torch._ops.OpOverload, node.target)

        if any(
            a.is_shard()
            for a in pytree.arg_tree_leaves(*args)
            if isinstance(a, DSymInt)
        ):
            if op_overload in VIEW_SYM_INT_CONSUMERS:
                assert len(kwargs) == 0, f"Expect empty kwargs, but got {kwargs}"
                node_to_obj[node] = VIEW_SYM_INT_CONSUMERS[op_overload](node, args)
                return None
            elif op_overload in FACTORY_SYM_INT_CONSUMERS:
                assert default_mesh is not None, "Requires default mesh for factory ops"
                node_to_obj[node] = FACTORY_SYM_INT_CONSUMERS[op_overload](
                    node, args, kwargs, default_mesh
                )
                return None
            else:
                assert isinstance(logger, logging.Logger)
                logger.warning(
                    "Assuming using local_value from SymInt for %s"
                    "is mathematically correct. Full args are %s.",
                    op_overload,
                    args,
                )

        if node.target == aten.view.default:
            # HACK: this is a hack to get around with the fact that some
            # view operations on a "global" tensor is invalid usage
            # but somehow the view operation on the batch input might hit it
            # so we convert the view op to reshape before calling DTensor
            op_overload = aten.reshape.default

        # DSymInt args are not sharded on any dimension, local value and global
        # value should be the same
        args = tree_map(lambda a: a.local_value if isinstance(a, DSymInt) else a, args)
        kwargs = tree_map(
            lambda a: a.local_value if isinstance(a, DSymInt) else a, kwargs
        )

        if op_overload in FACTORY_OPS:
            # Don't pass factory ops to DTensor dispatch, as DTensor cannot
            # propagate sharding spec without DTensor inputs.
            node_to_obj[node] = FACTORY_OPS[op_overload](
                node, args, kwargs, default_mesh
            )
            return None

        dispatch = partial(
            _dispatch_with_local_tensors,
            op_overload,
            kwargs=kwargs,
            specs=args,
        )

        gm = make_fx(dispatch, _allow_non_fake_inputs=False)(args)
        # FIXME(@wanchaol, @mrshenli): the above seems to accidentally captured
        # DeviceMesh tensor ops when handling inplace operators? The ``_to_copy`` is
        # not connected to graph output. So, using DCE to get rid of it, but this
        # doesn't look correct.
        #
        # The following operators appear in the captured graph, where the dtype is
        # torch.int64.
        #
        # get_attr       _tensor_constant0  _tensor_constant0         ()
        # call_function  transpose          aten.transpose.int        (_tensor_constant0, -1, 0)
        # call_function  view               aten.view.default         (transpose, [-1, 2])
        # call_function  view_1             aten.view.default         (view, [2])
        # call_function  _to_copy           aten._to_copy.default     (view_1,)
        gm.graph.eliminate_dead_code()

        return gm


def _build_dummy_add_graph(
    dt: DTensor, node_to_obj: Dict[fx.Node, Any]
) -> Tuple[fx.GraphModule, Any]:
    """Create a graph for a dummy add function from a partial DTensor.

    This dummy add is used for triggering all_reduce on a Partial DTensor
    during the DTensor expansion of the traced graph.
    Also returns the actual DTensor after resharding.
    """

    def dummy_add(grad: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        return grad + zero

    grad: torch.Tensor = dt._local_tensor
    zero: torch.Tensor = torch.zeros_like(dt._local_tensor)

    traced_add = make_fx(dummy_add)(grad, zero)

    placeholders = [n for n in traced_add.graph.nodes if n.op == OP.PLACEHOLDER]
    call_functions = [n for n in traced_add.graph.nodes if n.op == OP.CALL_FUNCTION]
    assert len(placeholders) == 2
    assert len(call_functions) == 1
    node_to_obj[placeholders[0]] = dt
    node_to_obj[placeholders[1]] = DTensor.from_local(
        zero, dt.device_mesh, [Replicate()], run_check=False
    )

    traced_dispatch = _get_dtensor_dispatch_graph(
        call_functions[0], node_to_obj, force_make_fx=True
    )
    assert traced_dispatch is not None

    # TODO(anj): This depends on the call function node -> actual DTensor output
    # mapping that we want to avoid for SPMD expansion
    return traced_dispatch, node_to_obj[call_functions[0]]


def _convert_output(
    gm: fx.GraphModule,
    node: fx.Node,
    node_to_obj: Dict[fx.Node, Any],
) -> fx.Node:
    new_args = []
    has_partial = False
    for argument in node.args[0]:  # type: ignore[union-attr]
        if not isinstance(argument, fx.Node):
            new_args.append(argument)
            continue

        obj = node_to_obj[argument]

        if not _is_partial_dtensor(obj):
            new_args.append(argument)
            continue

        has_partial = True

        # we know it's a dtensor from is partial DT check...
        dt = cast(DTensor, obj)

        traced_dispatch, result_obj = _build_dummy_add_graph(dt, node_to_obj)

        wait = [
            n
            for n in traced_dispatch.graph.nodes
            if n.name == "wait_comm" or n.name == "wait_tensor"
        ]
        add = [n for n in traced_dispatch.graph.nodes if n.name == "add"]
        assert len(wait) == 1 and len(add) == 1

        # remove add node and replace it with wait node
        add[0].replace_all_uses_with(wait[0])
        traced_dispatch.graph.eliminate_dead_code()
        # also update the actual DTensor corresponding to the node
        # TODO(anj): We require mapping of the final DTensor output to the wait
        # comm node.
        node_to_obj[wait[0]] = result_obj

        value_remap: Dict[fx.Node, fx.Node] = {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                # do nothing, ignore placeholders, as it has
                # already been prepared in value_remap
                value_remap[dtn] = argument
            elif dtn.op == OP.OUTPUT:
                assert (
                    len(dtn.args) == 1 and len(dtn.args[0]) == 1
                ), f"Expecting single output, but got {dtn.args} {len(dtn.args)}"
                new_args.append(value_remap[dtn.args[0][0]])
                # the concrete DTensor value of output was added when creating the
                # inner graph (in _build_dummy_add_graph). Just add it to the final
                # output node so that we can report the final output specs correctly.
                # TODO(anj): We are depending on the concrete DTensor output of the dummy add.
                node_to_obj[value_remap[dtn.args[0][0]]] = node_to_obj[dtn.args[0][0]]

            else:
                if dtn.op == OP.GET_ATTR:
                    setattr(
                        gm,
                        dtn.target,
                        getattr(traced_dispatch, dtn.target),
                    )
                with gm.graph.inserting_before(node):
                    value_remap[dtn] = gm.graph.node_copy(dtn, lambda n: value_remap[n])
    if has_partial:
        gm.graph.erase_node(node)
        return gm.graph.output(new_args)
    else:
        return node


def _rebuild_graph(
    gm: fx.GraphModule,
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule],
) -> None:
    # replace nodes in local traced graph with DTensor's dispatch graph
    for node in gm.graph.nodes:
        if node not in node_replacements:
            continue

        traced_dispatch = node_replacements[node]
        # Map DT's dispatch graph input placeholder nodes to the ones in
        # local traced graph. It uses index-based accessing, which is
        # brittle, just for testing purpose.
        flatten_args = pytree.arg_tree_leaves(*node.args)
        i, value_remap = 0, {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                value_remap[dtn] = flatten_args[i]
                i += 1

        # insert DT's dispatch graph to traced local graph.
        with gm.graph.inserting_before(node):
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == OP.PLACEHOLDER:
                    # do nothing, ignore placeholders, as it has already
                    # been prepared in value_remap
                    pass
                elif dtn.op == OP.OUTPUT:
                    assert (
                        len(dtn.args) == 1
                    ), f"Expecting single output, but got {dtn.args} {len(dtn.args[0])}"
                    outputs = dtn.args[0]
                    # we currently support two very specific types of output
                    # 1. single output
                    # 2. multiple outputs resulting from getitem of all elements of tuple
                    if len(outputs) == 1:
                        # for single output, we replace the node with the single node
                        output = outputs[0]
                    else:
                        # for multiple outputs, we check that these outputs correspond
                        # to all elements of a tuple. In that case, we replace
                        # uses of the output directly with the original tuple
                        source = None
                        for i, out in enumerate(outputs):
                            # we allow None outputs for certain items in the tuple
                            if out is None:
                                continue
                            assert out.op == "call_function"
                            assert out.target.__module__ == "_operator"
                            assert out.target.__name__ == "getitem"
                            assert source is None or source == out.args[0]
                            source = out.args[0]
                            assert out.args[1] == i
                        assert source is not None
                        output = source

                    new_node = value_remap[output]
                    node.replace_all_uses_with(new_node)
                else:
                    value_remap[dtn] = gm.graph.node_copy(dtn, lambda n: value_remap[n])
                    if all(
                        isinstance(n.target, torch._ops.OpOverload)
                        and n.target._schema.name.startswith(
                            ("aten::_foreach", "aten::_fused_adam")
                        )
                        for n in [dtn, node]
                    ):
                        # FIXME(@mrshenli): This is a temporary solution enable
                        # foreach ops. The problem is that foreach ops returns
                        # List[Tensor], but make_fx will flatten that before
                        # passing those tensors to output node, which will
                        # introduce additional getitem nodes. These redundant
                        # getitem nodes breaks graph correctness as we cannot do
                        # getitem(getitem(foreach_out, 0), 0). This temporary
                        # solution skips getitem nodes in DTensor expanded
                        # subgraphs.
                        node.replace_all_uses_with(value_remap[dtn])
                        break
            # explicitly erase node instead of relying on DCE, as DCE does not
            # remove inplace copy_ correctly.
            gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _get_last_consumer_to_nodes(
    graph: fx.Graph,
) -> Dict[fx.Node, List[fx.Node]]:
    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_consumer: Dict[fx.Node, fx.Node] = {}
    last_consumer_to_nodes: Dict[fx.Node, List[fx.Node]] = {}

    def _register_final_consumer(arg_node: fx.Node, consumer: fx.Node) -> None:
        if arg_node not in node_to_last_consumer:
            node_to_last_consumer[arg_node] = consumer
            last_consumer_to_nodes.setdefault(consumer, []).append(arg_node)

    for node in reversed(graph.nodes):
        fx.node.map_arg(
            node.args, lambda arg_node: _register_final_consumer(arg_node, node)
        )
        fx.node.map_arg(
            node.kwargs,
            lambda kwarg_node: _register_final_consumer(kwarg_node, node),
        )

    return last_consumer_to_nodes


def _convert_to_distributed(
    gm: fx.GraphModule,
    inps: List[torch.Tensor],
    schemas: List[Schema],
    default_mesh: Optional[DeviceMesh] = None,
    _allow_partial: bool = False,
) -> Tuple[fx.GraphModule, Dict[str, Schema]]:
    """Transform a graph module to a distributed graph module.

    Returns:
        - transformed graph module
        - map from output name to DTensorSpec

    """
    global logger
    logger = get_logger("spmd_exp")
    operators = {getattr(operator, name) for name in operator.__all__}
    node_to_obj: Dict[fx.Node, Any] = {}
    # map local op node in traced_f to its corresponding subgraph of
    # DTensor ops.
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

    last_consumer_to_nodes = _get_last_consumer_to_nodes(gm.graph)

    output_schemas: Dict[str, Schema] = {}
    for i, node in enumerate(gm.graph.nodes):
        assert logger is not None
        logger.info("node%s: op=%s target=%s", i, node.op, node.target)
        if node.op == OP.PLACEHOLDER:
            assert i < len(
                inps
            ), f"got more placeholder nodes ({i + 1}) than inputs ({len(inps)})"

            # our example inputs are local shards. Create DTensors from them.
            node_to_obj[node] = DTensor.from_local(
                inps[i].clone(),  # use clone to avoid modifications from inplace ops
                schemas[i].mesh,
                schemas[i].placements,
                # prevent running this collective in backwards pass
                run_check=False,
            )
        elif isinstance(node.target, torch._ops.OpOverloadPacket):
            dtensor = cast(DTensor, node_to_obj[node.args[0]])
            node_to_obj[node] = DSymInt.from_node(node, dtensor)
        elif isinstance(node.target, torch._ops.OpOverload):
            replacement = _get_dtensor_dispatch_graph(
                node, node_to_obj, default_mesh=default_mesh
            )
            if replacement is not None:
                node_replacements[node] = replacement
        elif node.op == OP.OUTPUT:
            if not _allow_partial:
                # Returns an expanded dummy add node that ensures
                # that the partial output tensor has been converted
                # to a replicated tensor.
                node = _convert_output(gm, node, node_to_obj)

            # Save output sharding for the inputs to backward pass.
            # TODO(anj): Pipe the output schema for the BW pass
            # instead of requiring the full output DTensor to be
            # materialized.
            for inp_arg in node.args[0]:
                if isinstance(inp_arg, fx.Node):
                    obj = node_to_obj[inp_arg]
                    if isinstance(obj, DTensor):
                        output_schemas[inp_arg.name] = Schema(
                            obj.device_mesh, obj.placements  # type: ignore[arg-type]
                        )
        elif node.op == OP.CALL_FUNCTION:
            args = tree_map(partial(_remap_arg, node_to_obj), node.args)
            kwargs = tree_map(partial(_remap_arg, node_to_obj), node.kwargs)

            dsymints = list(
                filter(lambda a: isinstance(a, DSymInt), args + tuple(kwargs.values()))
            )

            if node.target in operators and len(dsymints) > 0:
                assert all(
                    dsymints[0].mesh == d.mesh for d in dsymints
                ), "all DSymInts must have the same mesh. "

                local_args = tree_map_only(DSymInt, lambda a: a.local_value, args)
                local_kwargs = tree_map_only(DSymInt, lambda a: a.local_value, kwargs)

                global_args = tree_map_only(DSymInt, lambda a: a.global_value, args)
                global_kwargs = tree_map_only(DSymInt, lambda a: a.global_value, kwargs)

                node.args = local_args
                node.kwargs = local_kwargs

                node_to_obj[node] = DSymInt(
                    local_value=node.target(*local_args, **local_kwargs),
                    global_value=node.target(*global_args, **global_kwargs),
                    mesh=dsymints[0].mesh,
                )
            else:
                assert len(dsymints) == 0, (
                    "SPMD expansion does not support SymInt in non-operator "
                    f"nodes, got {node.target}."
                )
                node_to_obj[node] = node.target(*args, **kwargs)
        else:
            raise ValueError(f"Unrecognized node.op type {node.op}")

        if node in last_consumer_to_nodes:
            # Save memory by deleting objs that wont be used anymore.
            for arg_node in last_consumer_to_nodes[node]:
                del node_to_obj[arg_node]

    _rebuild_graph(gm, node_replacements)

    return gm, output_schemas
