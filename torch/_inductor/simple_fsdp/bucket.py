# mypy: ignore-errors
import functools
from typing import Dict

import torch
from torch.utils._ordered_set import OrderedSet

from .. import ir, scheduler
from ..comms import get_op_idx
from ..dependencies import StarDep, WeakDep
from ..utils import is_collective, is_wait
from ..virtualized import V
from .bucket_utils import (
    _find_recursive_deps_of_snode,
    _find_recursive_users_of_snode,
    _remove_operation,
    _replace_scheduler_buffer,
    _schedule_fallback_operation,
    _schedule_snode,
    bucket_all_gathers,
    bucket_reduce_scatters,
    get_fx_node,
)
from .reorder import _check_ir_node_fsdp


def bucket_fsdp_all_gather_concat_on_scheduler_ir(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    all_gather_bucket_plan: Dict[str, list["scheduler.BaseSchedulerNode"]],
    non_bucketable_pg,
) -> list["scheduler.BaseSchedulerNode"]:
    # Given a list of scheduler nodes `snodes`, pick out all_gather nodes and bucket them according to `all_gather_bucket_plan`.
    # It will return a new list of scheduler nodes, which is the same as `snodes` except that all_gather nodes are bucketed.
    # If `all_gather_bucket_plan` is not provided (all_gather_bucket_plan is [{}]), it generate a dummy plan by bucket every 5 AGs.
    # If there is no all_gather in snodes, it will return the input snodes.
    new_order: list[scheduler.BaseSchedulerNode] = []
    scheduled = OrderedSet()
    ag_exists = False
    ag_snode_to_cast_snode: Dict[
        scheduler.BaseSchedulerNode, scheduler.BaseSchedulerNode
    ] = {}
    ag_snode_to_wait_snode: Dict[
        scheduler.BaseSchedulerNode, scheduler.BaseSchedulerNode
    ] = {}
    new_operation_name_to_snode = {}

    schedule_snode = functools.partial(
        _schedule_snode, new_order=new_order, scheduled=scheduled
    )
    replace_scheduler_buffer = functools.partial(
        _replace_scheduler_buffer, name_to_buf=name_to_buf
    )
    remove_operation = functools.partial(
        _remove_operation, name_to_fused_node=name_to_fused_node
    )
    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
        schedule_snode_fn=schedule_snode,
        new_operation_name_to_snode=new_operation_name_to_snode,
    )

    # Step 1: Find all all_gather nodes
    for snode in snodes:
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            ag_exists = True
            ag_snode = snode
            ag_related_snode_set: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()

            # Find the "cast + all_gather" code block
            _find_recursive_deps_of_snode(
                ag_snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
                allow_weak_dep=False,
            )            # sort nodes by original operation order
            ag_related_snodes = sorted(
                ag_related_snode_set, key=lambda x: get_op_idx(x)
            )
            if len(ag_related_snodes) >= 2:
                cast_node = ag_related_snodes[-2]
                for node in ag_related_snodes:
                    if not is_wait(node.node) and not is_collective(node.node):
                        cast_node = node
                ag_snode = snode
                ag_snode_to_cast_snode[ag_snode] = cast_node
            else:
                ag_snode = ag_related_snodes[0]

            # Find the "all_gather + wait_tensor" code block
            assert len(ag_snode.outputs) == 1
            assert len(ag_snode.outputs[0].users) == 1
            wait_snode = ag_snode.outputs[0].users[0].node
            ag_snode_to_wait_snode[ag_snode] = wait_snode

    if ag_exists:
        assert len(ag_snode_to_wait_snode) > 0
    else:
        return snodes

    # Step 2: Put all_gather nodes into buckets
    ag_snode_to_bucket_id = {}
    ag_snode_to_bucket_id_coarsen = {}
    cur_bucket_id: int = 0

    if all_gather_bucket_plan == [{}]:
        # generate a dummy plan by bucket every 5 AGs
        for ag_snode in ag_snode_to_wait_snode.keys():
            if len(all_gather_bucket_plan[-1]) == 5:
                all_gather_bucket_plan.append([])
            all_gather_bucket_plan[-1].append(ag_snode)

    for all_gather_bucket in all_gather_bucket_plan:
        for all_gather_info, all_gather_list in all_gather_bucket.items():
            ag_snode_to_bucket_id.update(
                dict.fromkeys(all_gather_list, all_gather_info + (cur_bucket_id,))
            )
            ag_snode_to_bucket_id_coarsen.update(
                dict.fromkeys(all_gather_list, cur_bucket_id)
            )
        cur_bucket_id += 1
    assert len(ag_snode_to_bucket_id) == len(ag_snode_to_wait_snode)

    # Step 3: Create new (bucketed) all_gather nodes
    # TODO(yf225): horizontally fuse all cast ops into one op
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}

    for bucket_id, ag_bucket in enumerate(all_gather_bucket_plan):
        all_ag_snodes = []
        all_wait_snodes = []
        all_ag_input_ir_nodes = []
        group_sizes = []
        group_names = []
        for ag_info, ag_snodes in ag_bucket.items():
            if len(ag_snodes) == 0:
                continue
            example_ag_fx_node = get_fx_node(
                ag_snodes[0],
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            _, group_size, group_name = example_ag_fx_node.args
            ag_input_ir_nodes: list[ir.IRNode] = []
            wait_snodes = []
            for ag_snode in ag_snodes:
                ag_fx_node = get_fx_node(
                    ag_snode,
                    expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                )
                assert (
                    ag_fx_node.args[1] == group_size
                    and ag_fx_node.args[2] == group_name
                ), (
                    f"Expected group_size {group_size} and group_name {group_name}, but got {ag_fx_node.args[1:]}"
                )
                # TODO(yf225): this needs to consider the "no cast op" case, in which case we should directly take graph input as input
                # storage = V.graph.graph_inputs[name].data
                # assert isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
                if cast_snode := ag_snode_to_cast_snode.get(ag_snode, None):
                    assert len(cast_snode.get_outputs()) == 1
                    ag_input_ir_node = list(cast_snode.get_outputs())[0].node
                else:
                    met_deps = ag_snode.read_writes.reads - ag_snode.unmet_dependencies
                    assert len(met_deps) == 1, (
                        f"ag_snode: {ag_snode}, ag_snode.debug_str(): {ag_snode.debug_str()}, met_deps: {met_deps}"
                    )
                    ag_input_name = list(met_deps)[0].name
                    ag_input_ir_node = V.graph.graph_inputs[ag_input_name].data
                    assert (
                        isinstance(ag_input_ir_node, ir.StorageBox)
                        and ag_input_ir_node.is_input_buffer()
                    )
                ag_input_ir_nodes.append(ag_input_ir_node)
                wait_snodes.append(ag_snode_to_wait_snode[ag_snode])
            group_sizes.append(group_size)
            group_names.append(group_name)
            all_ag_snodes.append(ag_snodes)
            all_wait_snodes.append(wait_snodes)
            all_ag_input_ir_nodes.append(ag_input_ir_nodes)
        bucket_id_to_bucketed_op_info[bucket_id] = (
            all_ag_input_ir_nodes,
            group_sizes,
            group_names,
            all_ag_snodes,
            all_wait_snodes,
        )

    ag_snodes = OrderedSet(ag_snode_to_wait_snode.keys())
    ag_and_wait_snodes = OrderedSet()
    ag_and_wait_snodes |= ag_snodes  # all_gather
    ag_and_wait_snodes |= OrderedSet(ag_snode_to_wait_snode.values())  # wait_tensor

    for snode in snodes:
        if (
            snode not in ag_and_wait_snodes
        ):  # and snode not in list(ag_snode_to_cast_snode.values()):
            # not all_gather or its wait_tensor - schedule it normally
            schedule_snode(snode)
        elif snode in ag_snodes:
            assert snode in ag_snode_to_bucket_id, (
                f"{snode} not in {ag_snode_to_bucket_id}"
            )
            # bucket_id is the smaller one with (group_info, bucket)
            bucket_id = ag_snode_to_bucket_id[snode]
            # coarsen_bucket_id is the bigger one with bucket info
            coarsen_bucket_id = ag_snode_to_bucket_id_coarsen[snode]

            if coarsen_bucket_id not in bucket_id_is_scheduled:
                (
                    all_ag_input_ir_nodes,
                    group_sizes,
                    group_names,
                    all_orig_ag_snodes,
                    all_orig_wait_snodes,
                ) = bucket_id_to_bucketed_op_info[coarsen_bucket_id]

                AG_Group_node_list = []
                Wait_Group_node_list = []
                for idx, (
                    ag_input_ir_nodes,
                    orig_ag_snodes,
                    orig_wait_snodes,
                    group_size,
                    group_name
                ) in enumerate(
                    zip(all_ag_input_ir_nodes, all_orig_ag_snodes, all_orig_wait_snodes, group_sizes, group_names)
                ):
                    if len(orig_ag_snodes) == 1:
                        # If there is only one all_gather in the bucket, schedule it normally.
                        if orig_ag_snodes[0] in ag_snode_to_cast_snode:
                            AG_Group_node_list.append(
                                ag_snode_to_cast_snode[orig_ag_snodes[0]]
                            )
                        AG_Group_node_list.append(orig_ag_snodes[0])
                        Wait_Group_node_list.append(orig_wait_snodes[0])
                    else:
                        original_length = len(new_order)
                        outs = bucket_all_gathers(
                            schedule_fallback_operation,
                            group_size,
                            group_name,
                            ag_input_ir_nodes,
                            orig_ag_snodes,
                            name_to_buf,
                            orig_wait_snodes,
                            schedule_snode,
                        )
                        # Swap out the original wait output buffer with the new buffer,
                        # so that downstream user nodes can read from the new buffer just by using the original dep buffer name.
                        for out_operation, orig_ag_snode, orig_wait_snode in zip(
                            outs, orig_ag_snodes, orig_wait_snodes
                        ):
                            out_snode = new_operation_name_to_snode[
                                out_operation.get_operation_name()
                            ]
                            assert len(orig_ag_snode.outputs) == 1
                            orig_ag_snode_output = orig_ag_snode.outputs[-1]
                            orig_wait_snode_output = orig_wait_snode.outputs[-1]
                            out_snode_output = out_snode.outputs[-1]
                            replace_scheduler_buffer(
                                orig_sched_buf=orig_ag_snode_output,
                                new_sched_buf=out_snode_output,
                            )
                            # wait_tensor node output is modeled as a mutation on the all_gather node output.
                            # We need to preserve this property even after swapping.
                            assert (
                                isinstance(
                                    orig_wait_snode_output.node, ir.MutationOutput
                                )
                                and len(orig_wait_snode_output.get_mutations()) == 1
                                and orig_wait_snode_output.get_mutations()[0]
                                == orig_ag_snode_output.get_name()
                            )
                            out_snode.outputs.append(orig_wait_snode_output)
                            out_snode.read_writes.writes.add(
                                StarDep(
                                    name=orig_wait_snode_output.get_name(), mode=None
                                )
                            )
                            # Remove original all_gather and wait_tensor operations
                            remove_operation(orig_ag_snode.node)
                            remove_operation(orig_wait_snode.node)
                        new_length = len(new_order)
                        current_Wait_Group_node = []
                        current_AG_Group_node = []
                        wait_node = True
                        for node in range(new_length - original_length):
                            node = new_order.pop()
                            node.min_order = 0
                            node.max_order = 0
                            if wait_node:
                                current_Wait_Group_node.append(node)
                            else:
                                current_AG_Group_node.append(node)

                            if (
                                isinstance(node.node, ir.FallbackKernel)
                                and node.node.python_kernel_name
                                == "torch.ops._c10d_functional.wait_tensor.default"
                            ):
                                wait_node = False
                        current_AG_Group_node.reverse()
                        current_Wait_Group_node.reverse()
                        AG_Group_node_list.extend(current_AG_Group_node)
                        Wait_Group_node_list.extend(current_Wait_Group_node)
                AG_Group_node = scheduler.GroupedSchedulerNode.create(
                    AG_Group_node_list
                )
                Wait_Group_node = scheduler.GroupedSchedulerNode.create(
                    Wait_Group_node_list
                )
                AG_Group_node.temp_grouping = True
                Wait_Group_node.temp_grouping = True
                new_order.append(AG_Group_node)
                new_order.append(Wait_Group_node)
                bucket_id_is_scheduled[coarsen_bucket_id] = True

    return new_order


def bucket_fsdp_reduce_scatter_concat_on_scheduler_ir(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    reduce_scatter_bucket_plan: Dict[str, list["scheduler.BaseSchedulerNode"]],
    non_bucketable_pg,
) -> list["scheduler.BaseSchedulerNode"]:
    # Given a list of scheduler nodes `snodes`, pick out reduce_scatter nodes and bucket them according to `reduce_scatter_bucket_plan`.
    # It will return a new list of scheduler nodes, which is the same as `snodes` except that reduce_scatter nodes are bucketed.
    # If `reduce_scatter_bucket_plan` is not provided (reduce_scatter_bucket_plan is [[]]), it generate a dummy plan by bucket every 5 RSs.
    # If there is no reduce_scatter in snodes, it will return the input snodes.

    new_order: list[scheduler.BaseSchedulerNode] = []
    scheduled = OrderedSet()
    rs_exists = False
    rs_snode_to_wait_snode = {}
    new_operation_name_to_snode = {}

    schedule_snode = functools.partial(
        _schedule_snode, new_order=new_order, scheduled=scheduled
    )
    replace_scheduler_buffer = functools.partial(
        _replace_scheduler_buffer, name_to_buf=name_to_buf
    )
    remove_operation = functools.partial(
        _remove_operation, name_to_fused_node=name_to_fused_node
    )
    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
        schedule_snode_fn=schedule_snode,
        new_operation_name_to_snode=new_operation_name_to_snode,
    )

    # Step 1: Find all reduce_scatter nodes
    for snode in snodes:
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            rs_exists = True
            rs_snode = snode

            # Find the "reduce_scatter + wait_tensor" code block
            assert len(rs_snode.outputs) == 1
            assert len(rs_snode.outputs[0].users) == 1, (
                f"rs_snode.outputs[0].users: {rs_snode.outputs[0].users}"
            )
            wait_snode = rs_snode.outputs[0].users[0].node
            rs_snode_to_wait_snode[rs_snode] = wait_snode

    if rs_exists:
        assert len(rs_snode_to_wait_snode) > 0
    else:
        return snodes

    # Step 2: Put reduce_scatter nodes into buckets
    rs_snode_to_bucket_id = {}
    rs_snode_to_bucket_id_coarsen = {}
    cur_bucket_id: int = 0

    # if reduce_scatter_bucket_plan == [{}]:
    #     # generate a dummy plan by bucket every 5 RSs
    #     for rs_snode in rs_snode_to_wait_snode.keys():
    #         if len(reduce_scatter_bucket_plan[-1]) == 5:
    #             reduce_scatter_bucket_plan.append([])
    #         reduce_scatter_bucket_plan[-1].append(rs_snode)

    for reduce_scatter_bucket in reduce_scatter_bucket_plan:
        for reduce_scatter_info, reduce_scatter_list in reduce_scatter_bucket.items():
            rs_snode_to_bucket_id.update(
                dict.fromkeys(
                    reduce_scatter_list, reduce_scatter_info + (cur_bucket_id,)
                )
            )
            rs_snode_to_bucket_id_coarsen.update(
                dict.fromkeys(reduce_scatter_list, cur_bucket_id)
            )
        cur_bucket_id += 1

    assert len(rs_snode_to_bucket_id) == len(rs_snode_to_wait_snode)

    # Step 3: Create new (bucketed) reduce_scatter nodes
    order = {x: i for i, x in enumerate(snodes)}
    rs_snodes = OrderedSet(rs_snode_to_wait_snode.keys())
    rs_and_its_recursive_users = OrderedSet()
    rs_and_its_recursive_users |= rs_snodes  # all_gather
    rs_and_its_recursive_users |= OrderedSet(
        rs_snode_to_wait_snode.values()
    )  # wait_tensor

    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    for bucket_id, rs_bucket in enumerate(reduce_scatter_bucket_plan):
        all_rs_input_ir_nodes = []
        all_wait_snodes = []
        all_wait_snode_recursive_users = []
        all_rs_snodes = []
        for rs_info, rs_snodes in rs_bucket.items():
            if len(rs_snodes) == 0:
                continue
            example_rs_fx_node = get_fx_node(
                list(rs_snode_to_wait_snode.keys())[0],
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            _, reduce_op, group_size, group_name = example_rs_fx_node.args
            rs_input_ir_nodes: list[ir.IRNode] = []
            wait_snodes = []
            wait_snode_recursive_users = OrderedSet()
            for rs_snode in rs_snodes:
                rs_fx_node = get_fx_node(
                    rs_snode,
                    expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                )
                assert (
                    rs_fx_node.args[1] == reduce_op
                    and rs_fx_node.args[2] == group_size
                    and rs_fx_node.args[3] == group_name
                ), (
                    f"Expected reduce_op {reduce_op} and group_size {group_size} and group_name {group_name}, but got {rs_fx_node.args[1:]}"
                )
                unmet_real_deps = [
                    dep
                    for dep in rs_snode.unmet_dependencies
                    if not isinstance(dep, WeakDep)
                ]
                assert len(unmet_real_deps) == 1
                # rs_input_ir_nodes.append(name_to_buf[unmet_real_deps[0].name].node)
                rs_input_ir_nodes.append(rs_snode.node.inputs[0])
                wait_snode = rs_snode_to_wait_snode[rs_snode]
                wait_snodes.append(wait_snode)
                _find_recursive_users_of_snode(
                    wait_snode,
                    wait_snode_recursive_users,
                    name_to_buf,
                    name_to_fused_node,
                )
                # _find_recursive_users_of_snode() is inclusive - need to manually remove wait_snode from set
                wait_snode_recursive_users.remove(wait_snode)
                rs_and_its_recursive_users |= wait_snode_recursive_users

            all_rs_input_ir_nodes.append(rs_input_ir_nodes)
            all_wait_snodes.append(wait_snodes)
            all_wait_snode_recursive_users.append(wait_snode_recursive_users)
            all_rs_snodes.append(rs_snodes)
        bucket_id_to_bucketed_op_info[bucket_id] = (
            all_rs_input_ir_nodes,
            reduce_op,
            group_size,
            group_name,
            all_rs_snodes,
            all_wait_snodes,
            all_wait_snode_recursive_users,
        )

    last_bucket_id = 0
    RS_Group_node_list = []
    Wait_Group_node_list = []
    for snode in snodes:
        if snode not in rs_and_its_recursive_users:
            # not reduce_scatter or its wait_tensor - schedule it normally
            schedule_snode(snode)
        elif snode in rs_snode_to_wait_snode:
            assert snode in rs_snode_to_bucket_id, (
                f"{snode} not in {rs_snode_to_bucket_id}"
            )
            bucket_id = rs_snode_to_bucket_id[snode]
            coarsen_bucket_id = bucket_id[-1]

            if (
                coarsen_bucket_id not in bucket_id_is_scheduled
                and snode
                == bucket_id_to_bucketed_op_info[coarsen_bucket_id][-3][-1][-1]
            ):
                # If we are at the last node in the bucket, we can start to schedule the bucketed reduce_scatter node
                (
                    all_rs_input_ir_nodes,
                    reduce_op,
                    group_size,
                    group_name,
                    all_orig_rs_snodes,
                    all_orig_wait_snodes,
                    all_orig_wait_snode_recursive_users,
                ) = bucket_id_to_bucketed_op_info[coarsen_bucket_id]

                RS_Group_node_list = []
                Wait_Group_node_list = []
                for idx, (
                    rs_input_ir_nodes,
                    orig_rs_snodes,
                    orig_wait_snodes,
                    orig_wait_snode_recursive_users,
                ) in enumerate(
                    zip(
                        all_rs_input_ir_nodes,
                        all_orig_rs_snodes,
                        all_orig_wait_snodes,
                        all_orig_wait_snode_recursive_users,
                    )
                ):
                    if len(rs_input_ir_nodes) == 1:
                        # If there is only one input, we can directly use the input as the output
                        RS_Group_node_list.append(orig_rs_snodes[0])
                        Wait_Group_node_list.append(orig_wait_snodes[0])
                        Wait_Group_node_list.extend(orig_wait_snode_recursive_users)
                    else:
                        original_length = len(new_order)
                        new_sharded_grads = bucket_reduce_scatters(
                            schedule_fallback_operation,
                            group_size,
                            group_name,
                            reduce_op,
                            rs_input_ir_nodes,
                            orig_rs_snodes,
                            name_to_buf,
                            orig_wait_snodes,
                        )
                        for out_operation, orig_rs_snode, orig_wait_snode in zip(
                            new_sharded_grads, orig_rs_snodes, orig_wait_snodes
                        ):
                            out_snode = new_operation_name_to_snode[
                                out_operation.get_operation_name()
                            ]
                            assert len(orig_rs_snode.outputs) == 1
                            orig_rs_snode_output = orig_rs_snode.outputs[-1]
                            orig_wait_snode_output = orig_wait_snode.outputs[-1]
                            out_snode_output = out_snode.outputs[-1]
                            replace_scheduler_buffer(
                                orig_sched_buf=orig_rs_snode_output,
                                new_sched_buf=out_snode_output,
                            )
                            # wait_tensor node output is modeled as a mutation on the reduce_scatter node output.
                            # We need to preserve this property even after swapping.
                            assert (
                                isinstance(
                                    orig_wait_snode_output.node, ir.MutationOutput
                                )
                                and len(orig_wait_snode_output.get_mutations()) == 1
                                and orig_wait_snode_output.get_mutations()[0]
                                == orig_rs_snode_output.get_name()
                            )
                            out_snode.outputs.append(orig_wait_snode_output)
                            out_snode.read_writes.writes.add(
                                StarDep(
                                    name=orig_wait_snode_output.get_name(), mode=None
                                )
                            )
                            # Remove original reduce_scatter and wait_tensor operations
                            remove_operation(orig_rs_snode.node)
                            remove_operation(orig_wait_snode.node)

                    if len(rs_input_ir_nodes) != 1:
                        new_length = len(new_order)
                        current_RS_Group_node = []
                        current_Wait_Group_node = []
                        wait_node = True
                        for node in range(new_length - original_length):
                            node = new_order.pop()
                            node.min_order = 0
                            node.max_order = 0
                            if wait_node:
                                current_Wait_Group_node.append(node)
                            else:
                                current_RS_Group_node.append(node)
                            if (
                                isinstance(node.node, ir.FallbackKernel)
                                and node.node.python_kernel_name
                                == "torch.ops._c10d_functional.wait_tensor.default"
                            ):
                                wait_node = False
                        current_RS_Group_node.reverse()
                        current_Wait_Group_node.reverse()
                        RS_Group_node_list.extend(current_RS_Group_node)
                        Wait_Group_node_list.extend(current_Wait_Group_node)

                RS_Group_node = scheduler.GroupedSchedulerNode.create(
                    RS_Group_node_list
                )
                for (
                    orig_wait_snode_recursive_users
                ) in all_orig_wait_snode_recursive_users:
                    for user in sorted(
                        orig_wait_snode_recursive_users, key=lambda x: order[x]
                    ):
                        Wait_Group_node_list.append(user)
                Wait_Group_node = scheduler.GroupedSchedulerNode.create(
                    Wait_Group_node_list
                )
                RS_Group_node.temp_grouping = True
                Wait_Group_node.temp_grouping = True
                new_order.append(RS_Group_node)
                new_order.append(Wait_Group_node)
                bucket_id_is_scheduled[coarsen_bucket_id] = True
            else:
                continue

    if len(RS_Group_node_list) > 0:
        RS_Group_node = scheduler.GroupedSchedulerNode.create(RS_Group_node_list)
        Wait_Group_node = scheduler.GroupedSchedulerNode.create(Wait_Group_node_list)
        RS_Group_node.temp_grouping = True
        Wait_Group_node.temp_grouping = True
        new_order.append(RS_Group_node)
        new_order.append(Wait_Group_node)
    return new_order
