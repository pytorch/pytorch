# mypy: ignore-errors
import math
from typing import Any, Callable, cast, Dict, Union

import torch
import torch.utils._pytree as pytree
from torch.utils._ordered_set import OrderedSet

from .. import ir, scheduler
from ..dependencies import StarDep, WeakDep
from ..utils import buf_name_to_fused_snode
from ..virtualized import V
from .estimator import OpType


def get_fx_node(
    snode_or_ir_node: Union["scheduler.BaseSchedulerNode", "ir.IRNode"],
    expected_op: OpType,
) -> torch.fx.Node:
    origins = None
    if isinstance(snode_or_ir_node, scheduler.BaseSchedulerNode):
        origins = snode_or_ir_node.node.get_origins()
    elif isinstance(snode_or_ir_node, ir.IRNode):
        origins = snode_or_ir_node.origins
    else:
        raise ValueError(
            f"Expected BaseSchedulerNode or IRNode, got {type(snode_or_ir_node)}. Offending value: {snode_or_ir_node}"
        )
    origins_with_expected_op = [o for o in origins if o.target == expected_op]
    assert len(origins_with_expected_op) == 1
    return origins_with_expected_op[0]


def _schedule_snode(
    snode: "scheduler.BaseSchedulerNode",
    new_order: list["scheduler.BaseSchedulerNode"],
    scheduled: list["scheduler.BaseSchedulerNode"],
):
    if snode in scheduled:
        return

    new_order.append(snode)
    scheduled.add(snode)


def _find_recursive_deps_of_snode(
    snode: "scheduler.BaseSchedulerNode",
    collected_node_set: OrderedSet["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
    allow_weak_dep: bool = True,
):
    if criteria_cb and criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for dep in snode.unmet_dependencies:
        if isinstance(dep, WeakDep) and not allow_weak_dep:
            continue
        defining_op_for_dep = buf_name_to_fused_snode(
            dep.name, name_to_buf, name_to_fused_node
        )
        if defining_op_for_dep in collected_node_set:
            continue
        _find_recursive_deps_of_snode(
            defining_op_for_dep,
            collected_node_set,
            name_to_buf,
            name_to_fused_node,
            criteria_cb=criteria_cb,
        )


def _find_recursive_users_of_snode(
    snode: "scheduler.BaseSchedulerNode",
    collected_node_set: OrderedSet["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
):
    if criteria_cb and criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for o in snode.get_outputs():
        for user in o.users:
            assert user.node is not None
            if user.node.get_name() == "OUTPUT":
                continue
            if user.node.get_name() not in name_to_fused_node:
                continue
            user_op = name_to_fused_node[user.node.get_name()]
            if user_op in collected_node_set:
                continue
            _find_recursive_users_of_snode(
                user_op,
                collected_node_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=criteria_cb,
            )


def _replace_scheduler_buffer(
    orig_sched_buf: "scheduler.SchedulerBuffer",
    new_sched_buf: "scheduler.SchedulerBuffer",
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
):
    new_buf = new_sched_buf.node
    new_buf_name = new_buf.get_name()
    orig_buf = orig_sched_buf.node
    orig_buf_name = orig_buf.get_name()
    V.graph.buffers[V.graph.buffers.index(orig_buf)] = new_buf
    V.graph.name_to_buffer[orig_buf_name] = new_buf
    name_to_buf[orig_buf_name] = new_sched_buf
    new_buf.name = orig_buf_name
    new_sched_buf.defining_op.set_read_writes(
        new_sched_buf.defining_op.read_writes.rename({new_buf_name: orig_buf_name})
    )
    new_sched_buf.users = orig_sched_buf.users
    # # Check that they are no longer referenced anywhere else
    # assert sys.getrefcount(orig_sched_buf) == 2, f"sys.getrefcount(orig_sched_buf): {sys.getrefcount(orig_sched_buf)}"
    # del orig_sched_buf
    # assert sys.getrefcount(orig_buf) == 2, f"sys.getrefcount(orig_buf): {sys.getrefcount(orig_buf)}"
    # del orig_buf


def _remove_operation(
    operation: ir.Operation,
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
):
    assert isinstance(operation, ir.Operation), (
        f"Expected ir.Operation, but got {type(ir.Operation)}. Offending value: {ir.Operation}"
    )
    idx = V.graph.operations.index(operation)
    del V.graph.operations[idx]
    del V.graph.name_to_op[operation.get_operation_name()]
    del name_to_fused_node[operation.get_operation_name()]


def _schedule_fallback_operation(
    target: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    scheduler: "scheduler.Scheduler",
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    schedule_snode_fn: Callable[[Any], None],
    new_operation_name_to_snode: Dict[str, "scheduler.BaseSchedulerNode"],
    dep_operations: Union[ir.Operation, list[ir.Operation], None] = None,
    as_group_snode: bool = False,
    new_group_snode: bool = False,
) -> Union[ir.Operation, list[ir.Operation]]:
    # NOTE: `dep_operations` enforces strong ordering between ops, helpful if the dependency chain is not clear from direct input-output relationship
    # (i.e. if OP1 mutates a view of buffer X and then OP2 reads from X, and OP1 is expected to run before OP2 -> OP2 must have `dep_operations` pointing to OP1 to ensure reordering pass would not mess up the order).

    def wrap_tensors(x):
        if isinstance(x, ir.MutationOutput):
            mutated_buf_names = x.get_mutation_names()
            assert (
                isinstance(mutated_buf_names, list) and len(mutated_buf_names) == 1
            ), "Expect only one mutated buffer in MutationOutput"
            return wrap_tensors(name_to_buf[mutated_buf_names[0]].node)
        elif isinstance(x, ir.IRNode):
            if isinstance(x, ir.StorageBox):
                return x
            else:
                return ir.TensorBox.create(x)
        else:
            return x

    operations_prev_watermark = len(V.graph.operations)
    # this will append newly created operations to V.graph.operations
    ir.FallbackKernel.create(
        target,
        *pytree.tree_map(wrap_tensors, args),
        **pytree.tree_map(wrap_tensors, kwargs),
    )
    new_operations = V.graph.operations[operations_prev_watermark:]
    new_snodes = []
    if isinstance(dep_operations, ir.Operation):
        dep_operations = [dep_operations]
    for new_operation in new_operations:
        new_snode = scheduler.create_scheduler_node(new_operation)
        if dep_operations is not None:
            # make the new snode depend on all output buffers of all the dep operations,
            # to ensure that the new snode will always be executed after all the dep operations.
            for dep_operation in dep_operations:
                dep_snode = name_to_fused_node[dep_operation.get_operation_name()]
                for buf_name in dep_snode.get_buffer_names():
                    new_snode.set_read_writes(
                        new_snode.read_writes.with_read(
                            StarDep(name=buf_name, mode=None)
                        )
                    )

        schedule_snode_fn(new_snode)
        new_snodes.append(new_snode)
        new_operation_name_to_snode[new_operation.get_operation_name()] = new_snode
        for o in new_snode.get_outputs():
            name_to_buf[o.get_name()] = o
        name_to_fused_node[new_snode.get_name()] = new_snode
    multi_output_operations = []
    # identify the trailing MultiOutput operations, if any
    for operation in reversed(new_operations):
        if isinstance(operation, ir.MultiOutput):
            multi_output_operations.insert(0, operation)
        else:
            break
    if len(multi_output_operations) == 0:
        # if no MultiOutput operations, it means this fallback kernel has no output - in this case, just return the FallbackKernel operation.
        assert len(new_operations) == 1
        return new_operations[0]
    elif len(multi_output_operations) == 1:
        return multi_output_operations[0]
    else:
        return multi_output_operations


def bucket_all_gathers(
    schedule_fallback_operation: Callable,
    group_size: int,
    group_name: str,
    ag_input_ir_nodes: list["ir.IRNode"],
    orig_ag_snodes: list["scheduler.BaseSchedulerNode"],
    orig_wait_snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    schedule_snode_fn: Union[Callable[..., Any], Any] = None,
    return_ag_only: bool = False,
):
    """
    bucket multiple all_gather nodes into one all_gather node
    return_ag_only set to True: only return the bucketed all_gather node
    return_ag_only set to False: return the bucketed all_gather node and the bucketed wait node (in GroupedSchedulerNode)
    """
    grouped_ag_num = 0
    grouped_wait_num = 0

    orig_ag_fx_nodes = [
        get_fx_node(
            sn, expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default
        )
        for sn in orig_ag_snodes
    ]
    ag_input_fx_nodes = [n.args[0] for n in orig_ag_fx_nodes]
    assert all(
        n.meta["val"].dtype == orig_ag_fx_nodes[0].meta["val"].dtype
        for n in orig_ag_fx_nodes
    ), "All all_gather inputs in the same bucket must have the same dtype"

    # must schedule all the all_gather input nodes first, before the bucketed all_gather node
    param_all_gather_inputs_orig: list[Union[ir.IRNode, scheduler.SchedulerBuffer]] = []
    for ag_input_ir_node in ag_input_ir_nodes:
        if ag_input_sched_buf := name_to_buf.get(ag_input_ir_node.get_name()):
            if not return_ag_only:
                schedule_snode_fn(ag_input_sched_buf.defining_op)
                grouped_ag_num += 1
            param_all_gather_inputs_orig.append(ag_input_sched_buf.node)
        else:
            assert ag_input_ir_node.is_input_buffer()
            param_all_gather_inputs_orig.append(ag_input_ir_node)

    # schedule the bucketed all_gather node
    param_all_gather_inputs_flattened = [
        schedule_fallback_operation(torch.ops.aten.reshape.default, (n, [-1]), {})
        for n in param_all_gather_inputs_orig
    ]
    grouped_ag_num += len(param_all_gather_inputs_orig)

    inp_split_sizes = [n.meta["val"].numel() for n in ag_input_fx_nodes]
    param_all_gather_outputs = [
        schedule_fallback_operation(
            torch.ops.aten.empty.memory_format,
            ([n.meta["val"].numel() * group_size],),
            {
                "dtype": n.meta["val"].dtype,
                "device": n.meta["val"].device,
                "pin_memory": False,
            },
            as_group_snode=True,
            new_group_snode=False,
        )
        for n in ag_input_fx_nodes
    ]
    grouped_ag_num += len(ag_input_fx_nodes)
    # TODO(yf225): This assumes dim-0 sharding.
    # If we need to support sharding on another dim, we should look at how FSDP2 does it (e.g. search for `shard_dim` in FSDP2 codebase)
    param_all_gather_outputs_shape_orig = [
        (n.meta["val"].shape[0] * group_size,) + n.meta["val"].shape[1:]
        for n in ag_input_fx_nodes
    ]
    all_gather_input_numel = sum(inp_split_sizes)
    param_all_gather_outputs_flattened = schedule_fallback_operation(
        torch.ops.aten.empty.memory_format,
        ([all_gather_input_numel * group_size],),
        {
            "dtype": ag_input_fx_nodes[0].meta["val"].dtype,
            "device": ag_input_fx_nodes[0].meta["val"].device,
            "pin_memory": False,
        },
    )

    example_ag_input_tensor = ag_input_fx_nodes[0].meta["val"]
    all_gather_input, all_gather_output = schedule_fallback_operation(
        torch.ops.fsdp.all_gather_copy_in.default,
        (
            param_all_gather_inputs_flattened,
            param_all_gather_outputs_flattened,
            inp_split_sizes,
            all_gather_input_numel,
            example_ag_input_tensor.device.index,
        ),
        {},
    )
    all_gather_into_tensor_out = schedule_fallback_operation(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        (all_gather_input, group_size, group_name),
        {"out": all_gather_output},
    )
    if return_ag_only:
        assert len(all_gather_into_tensor_out) == 1
        return all_gather_into_tensor_out.inputs[0]

    wait_tensor = schedule_fallback_operation(
        torch.ops._c10d_functional.wait_tensor.default,
        (all_gather_into_tensor_out,),
        {},
    )
    all_gather_output_reshaped = schedule_fallback_operation(
        torch.ops.aten.reshape.default,
        (wait_tensor, [group_size, -1]),
        {},
    )
    outs_flattened = [
        schedule_fallback_operation(
            torch.ops.aten.reshape.default,
            (n, [group_size, -1]),
            {},
        )
        for n in param_all_gather_outputs
    ]
    split_with_sizes_copy = schedule_fallback_operation(
        torch.ops.fsdp.split_with_sizes_copy.default,
        (all_gather_output_reshaped, inp_split_sizes),
        {"dim": 1, "out": outs_flattened},
    )
    outs = [
        schedule_fallback_operation(
            torch.ops.aten.reshape.default,
            (n, orig_shape),
            {},
            dep_operations=split_with_sizes_copy,
        )
        for n, orig_shape in zip(outs_flattened, param_all_gather_outputs_shape_orig)
    ]
    # Make sure downstream users of original wait nodes are now dependent on the new `outs` nodes
    assert len(outs) == len(orig_wait_snodes)
    assert len(outs) == len(orig_ag_snodes)
    return outs


def bucket_reduce_scatters(
    schedule_fallback_operation: Callable,
    group_size: int,
    group_name: str,
    reduce_op: Any,
    rs_input_ir_nodes: list["ir.IRNode"],
    orig_rs_snodes: list["scheduler.BaseSchedulerNode"],
    orig_wait_snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    return_rs_only: bool = False,
):
    orig_rs_fx_nodes = [
        get_fx_node(
            sn, expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        )
        for sn in orig_rs_snodes
    ]
    # must schedule all the reduce_scatter input nodes first, before the bucketed reduce_scatter node
    unsharded_grads = []
    unsharded_grads_fx_nodes = [n.args[0] for n in orig_rs_fx_nodes]
    for rs_input_ir_node in rs_input_ir_nodes:
        if rs_input_sched_buf := name_to_buf.get(rs_input_ir_node.get_name()):
            unsharded_grads.append(rs_input_sched_buf.node)
        else:
            assert rs_input_ir_node.is_input_buffer()
            unsharded_grads.append(rs_input_ir_node)
    reduce_dtype = unsharded_grads_fx_nodes[0].meta["val"].dtype
    # Only float32 and bfloat16 are supported for now.
    # To support fp16, please see FSDP2 `_get_gradient_divide_factors`.
    assert reduce_dtype in (torch.float32, torch.bfloat16), (
        f"reduce_dtype {reduce_dtype} is not supported"
    )
    assert all(n.meta["val"].dtype == reduce_dtype for n in unsharded_grads_fx_nodes)
    device = unsharded_grads_fx_nodes[0].meta["val"].device
    rank = device.index
    # TODO(yf225): need more work if we want to support non-dim-0 sharding (e.g. search for `shard_dim` in FSDP2 codebase)
    shard_dim = 0

    def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
        padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
        return cast(torch.Size, torch.Size([padded_dim0]) + tensor_size[1:])

    def _get_dim_chunked_size(
        chunk: torch.Tensor, unchunked_size: torch.Size, dim: int
    ) -> torch.Size:
        if chunk.numel() > 0:
            return chunk.size()
        # For 0 numel, we need to preserve nonzero-sized dims for DTensor APIs
        return unchunked_size[:dim] + torch.Size([0]) + unchunked_size[dim + 1 :]

    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(n.meta["val"].size(), group_size)
        for n in unsharded_grads_fx_nodes
    )

    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_input = schedule_fallback_operation(
        torch.ops.aten.empty.memory_format,
        ([reduce_scatter_input_numel],),
        {
            "dtype": reduce_dtype,
            "device": device,
            "pin_memory": False,
        },
    )
    reduce_scatter_input_reshaped = schedule_fallback_operation(
        torch.ops.aten.reshape.default,
        (reduce_scatter_input, [group_size, -1]),
        {},
    )
    # NOTE(yf225): have to turn off Inductor config shape_padding and comprehensive_padding,
    # otherwise we get "torch.Size([4096, 80096]) and strides (80128, 1) cannot be viewed as shape (2, 164036608)" error.
    chunk_cat = schedule_fallback_operation(
        torch.ops.fsdp.chunk_cat.default,
        (unsharded_grads,),
        {
            "dim": 0,
            "num_chunks": group_size,
            "out": reduce_scatter_input_reshaped,
        },
    )

    # chunk_cat, reduce_scatter_input = schedule_fallback_operation(
    #     torch.ops.fsdp.chunk_cat_with_output.default,
    #     (unsharded_grads,),
    #     {
    #         "dim": 0,
    #         "num_chunks": group_size,
    #        # "out": reduce_scatter_input_reshaped,
    #     },
    #     return_outputs = True,
    # )

    reduce_scatter_tensor = schedule_fallback_operation(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        (reduce_scatter_input, reduce_op, group_size, group_name),
        {},
        dep_operations=chunk_cat,
    )

    if return_rs_only:
        assert len(reduce_scatter_tensor.inputs[0]) == 1
        return reduce_scatter_tensor.inputs[0]

    wait_tensor = schedule_fallback_operation(
        torch.ops._c10d_functional.wait_tensor.default,
        (reduce_scatter_tensor,),
        {},
    )

    def _chunk_with_empty(
        tensor: torch.Tensor, num_chunks: int, dim: int
    ) -> list[torch.Tensor]:
        chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
        while len(chunks) < num_chunks:
            chunks.append(chunks[0].new_empty(0))
        return chunks

    reduce_output = wait_tensor
    # View out and accumulate sharded gradients
    new_sharded_grads = []

    flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
    for padded_unsharded_size, unsharded_grad_fx_node in zip(
        padded_unsharded_sizes, unsharded_grads_fx_nodes
    ):
        # NOTE: we only care about the shape of tensors in `chunks`, so using meta tensor here
        chunks = _chunk_with_empty(
            torch.empty_like(unsharded_grad_fx_node.meta["val"], device="meta"),
            group_size,
            dim=shard_dim,
        )
        sharded_param = chunks[rank]
        sharded_size = sharded_param.size()
        # sharded_size = _get_dim_chunked_size(
        #     sharded_param, sharded_size, dim=shard_dim
        # )
        contiguous_sharded_stride = torch._prims_common.make_contiguous_strides_for(
            sharded_size
        )
        # Assume even sharding for Shard(i), i > 0; otherwise would require
        # copy-out for contiguous strides
        # print("sharded_size", sharded_size, "contiguous_sharded_stride", contiguous_sharded_stride)
        new_sharded_grad = schedule_fallback_operation(
            torch.ops.aten.as_strided.default,
            (reduce_output,),
            {
                "size": sharded_size,
                "stride": contiguous_sharded_stride,
                "storage_offset": flat_grad_offset,
            },
        )
        new_sharded_grads.append(new_sharded_grad)
        padded_sharded_numel = padded_unsharded_size.numel() // group_size
        flat_grad_offset += padded_sharded_numel
    assert len(orig_wait_snodes) == len(new_sharded_grads)
    assert len(orig_wait_snodes) == len(orig_rs_snodes)
    return new_sharded_grads
