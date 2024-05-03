import torch
import operator
import copy
import logging
from collections import defaultdict, OrderedDict
from torch.distributed._composable.fsdp import _fsdp_collectives

torch_log = logging.getLogger("torch")


_view_ops = [
    # TODO(yf225): list all possible view ops here
    torch.ops.aten.select.Dimname,
    torch.ops.aten.select.int,
    torch.ops.aten.permute.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.reshape.default,
    torch.ops.aten.broadcast_tensors.default,
    torch.ops.aten.t.default,
    torch.ops.aten.as_strided.default,
]

# TODO(yf225): if user model has these existing ops without FSDP, how to accommodate them?
must_not_appear_ops_after_fsdp_fx_passes = [
    "torch.ops.aten.full.",
    "torch.ops.aten.clone.",
    "torch.ops.aten.slice_scatter.",
    "torch.ops.aten.as_strided_scatter.",
    "torch.ops.aten.foreach_copy.",
    "torch.ops.aten.foreach_copy_.",
    "torch.ops.aten.copy.",
]


def _input_is_used_in_view_ops(ops, inp_n):
    for n in ops:
        if inp_n in _flatten_arg_list(n.args) and inp_n.target in _view_ops:
            return True
    return False


def _find_next_use_of_node(node_list, node):
    for i, n in enumerate(node_list):
        if node in _flatten_arg_list(n.args):
            return n
    return None


def replace_noop_consecutive_permutes_with_original_input_if_first_permute_out_has_no_other_use(mod):
    """
    # NOTE: we only handle len(permute_dims) = 2 case for now.

    permute_3: "f32[12340, 12340]" = torch.ops.aten.permute.default(getitem_106, [1, 0])
    permute_4: "f32[12340, 12340]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None

    ->

    getitem_106
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.permute.default and len(n.args[1]) == 2:
            permute_dims = n.args[1]
            first_permute_node = n
            second_permute_node = None
            first_permute_output_has_other_use = False
            # First check that the first permute output has no other use
            for j, node in enumerate(node_list[i+1:]):
                if first_permute_node in _flatten_arg_list(node.args):
                    if node.target is not torch.ops.aten.permute.default:
                        first_permute_output_has_other_use = True
                    else:
                        if node.args[1] == permute_dims:
                            # if permute_dims also match, we know these two consecutive permutes lead to a no-op.
                            second_permute_node = node
                        else:
                            first_permute_output_has_other_use = True
            if second_permute_node is not None and not first_permute_output_has_other_use:
                second_permute_node.replace_all_uses_with(first_permute_node.args[0])
                mod.graph.erase_node(second_permute_node)
                mod.graph.erase_node(first_permute_node)
                mod.graph.lint()
                mod.recompile()


# TODO(yf225): dedup with `is_alias_of_primal_input` in partitioners.py
def _is_alias_of_graph_input(graph, graph_inputs, node):
    if hasattr(node, "target") and node.target in _view_ops:
        view_chain = [node]
        flattened_arg_list = _flatten_arg_list(node.args)
        for arg in flattened_arg_list:
            if str(arg) in graph_inputs:
                return True, view_chain
            else:
                upstream_is_alias, upstream_view_chain = _is_alias_of_graph_input(graph, graph_inputs, arg)
                if upstream_is_alias:
                    view_chain.extend(upstream_view_chain)
                    return True, view_chain
    return False, []


def _flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(_flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def _get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _collect_node_types_and_indices(mod):
    node_type_to_node_set = defaultdict(set)
    node_to_node_index = {}
    for i, n in enumerate(mod.graph.nodes):
        node_type_to_node_set[n.target].add(n)
        node_to_node_index[n] = i
    return node_type_to_node_set, node_to_node_index


# TODO(yf225): dedup with partitioner.py view chain move code
def _is_descendent_of_primal_input(node, primal_inputs):
    if node in primal_inputs:
        return True, [node], node
    elif hasattr(node, "target"):
        op_chain = [node]
        flattened_arg_list = _flatten_arg_list(node.args)
        for arg in flattened_arg_list:
            upstream_is_alias, upstream_op_chain, primal_input = _is_descendent_of_primal_input(arg, primal_inputs)
            if upstream_is_alias:
                op_chain.extend(upstream_op_chain)
                return True, op_chain, primal_input
    else:
        return False, [], None


def _get_op_chains_from_primals(end_nodes, primal_inputs, all_node_list):
    # Check end_node is descendent of primals_X, and move up the entire chain starting from primals_X.
    op_chains_from_primals = []
    for end_node in end_nodes:
        is_descendent_of_primal_input, op_chain, primal_input = _is_descendent_of_primal_input(end_node, primal_inputs)
        assert op_chain[-1] == primal_input
        # primals_X must not be used in any view ops in this graph (otherwise it might have alias mutation
        # which is hard to track and would cause it to potentially not be legal to move the chain up).
        if is_descendent_of_primal_input and not _input_is_used_in_view_ops(all_node_list, primal_input):
            op_chains_from_primals.append(op_chain)
    return op_chains_from_primals


def _find_next_block_of_inplace_copy_nodes(node_list, start_index, expected_block_length, getitem_nodes):
    inplace_copy_nodes_start_index = None
    inplace_copy_nodes = []
    for k in range(start_index, len(node_list)):
        nk = node_list[k]
        if nk.target is torch.ops.aten.copy_.default and nk.args[0] in getitem_nodes:
            if inplace_copy_nodes_start_index is None:
                inplace_copy_nodes_start_index = k
            inplace_copy_nodes.append(nk)
            if len(inplace_copy_nodes) == expected_block_length:
                break
        else:
            inplace_copy_nodes_start_index = None
            inplace_copy_nodes = []
            continue
    return inplace_copy_nodes_start_index, inplace_copy_nodes


def _create_new_node_and_replace(mod, node, *, propagate_meta=False):
    new_node = mod.graph.call_function(node.target, node.args, node.kwargs)
    node.replace_all_uses_with(new_node, propagate_meta=propagate_meta)
    mod.graph.erase_node(node)
    return new_node


def raise_all_gather_to_overlap_with_prev_layer_compute(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_common.py:140 in _to_dtype_if_needed, code: return tensor.to(dtype)
    convert_element_type: "bf16[8192000]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
    convert_element_type_1: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
    convert_element_type_2: "bf16[8192000]" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:80 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(16384256, 8, 0, torch.bfloat16, device(type='cuda', index=0), [8192000, 256, 8192000], [convert_element_type, convert_element_type_1, convert_element_type_2]);  convert_element_type = convert_element_type_1 = convert_element_type_2 = None
    getitem: "bf16[16384256]" = all_gather_copy_in[0]
    getitem_1: "bf16[131074048]" = all_gather_copy_in[1];  all_gather_copy_in = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 8, '0');  getitem = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor: "bf16[131074048]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None

    ... (uses wait_tensor in compute)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_common.py:140 in _to_dtype_if_needed, code: return tensor.to(dtype)
    convert_element_type_3: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
    convert_element_type_4: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
    convert_element_type_5: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
    convert_element_type_6: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
    convert_element_type_7: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
    convert_element_type_8: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
    convert_element_type_9: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
    convert_element_type_10: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
    convert_element_type_11: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:80 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
    all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default(6423040, 8, 0, torch.bfloat16, device(type='cuda', index=0), [524288, 524288, 524288, 524288, 1441792, 1441792, 1441792, 256, 256], [convert_element_type_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, convert_element_type_10, convert_element_type_11]);  convert_element_type_3 = convert_element_type_4 = convert_element_type_5 = convert_element_type_6 = convert_element_type_7 = convert_element_type_8 = convert_element_type_9 = convert_element_type_10 = convert_element_type_11 = None
    getitem_14: "bf16[6423040]" = all_gather_copy_in_1[0]
    getitem_15: "bf16[51384320]" = all_gather_copy_in_1[1];  all_gather_copy_in_1 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
    all_gather_into_tensor_1: "bf16[51384320]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_14, 8, '0');  getitem_14 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_1: "bf16[51384320]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None

    ... (uses wait_tensor_1 in compute)

    ->

    convert_element_type: "bf16[8192000]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
    convert_element_type_1: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
    convert_element_type_2: "bf16[8192000]" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(16384256, 8, 0, torch.bfloat16, device(type='cuda', index=0), [8192000, 256, 8192000], [convert_element_type, convert_element_type_1, convert_element_type_2]);  convert_element_type = convert_element_type_1 = convert_element_type_2 = None
    getitem: "bf16[16384256]" = all_gather_copy_in[0]
    getitem_1: "bf16[131074048]" = all_gather_copy_in[1];  all_gather_copy_in = None
    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 8, '0');  getitem = None
    wait_tensor: "bf16[131074048]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    convert_element_type_3: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
    convert_element_type_4: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
    convert_element_type_5: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
    convert_element_type_6: "bf16[524288]" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
    convert_element_type_7: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
    convert_element_type_8: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
    convert_element_type_9: "bf16[1441792]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
    convert_element_type_10: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
    convert_element_type_11: "bf16[256]" = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
    all_gather_copy_in_1 = torch.ops.fsdp.all_gather_copy_in.default(6423040, 8, 0, torch.bfloat16, device(type='cuda', index=0), [524288, 524288, 524288, 524288, 1441792, 1441792, 1441792, 256, 256], [convert_element_type_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, convert_element_type_10, convert_element_type_11]);  convert_element_type_3 = convert_element_type_4 = convert_element_type_5 = convert_element_type_6 = convert_element_type_7 = convert_element_type_8 = convert_element_type_9 = convert_element_type_10 = convert_element_type_11 = None
    getitem_14: "bf16[6423040]" = all_gather_copy_in_1[0]
    getitem_15: "bf16[51384320]" = all_gather_copy_in_1[1];  all_gather_copy_in_1 = None
    all_gather_into_tensor_1: "bf16[51384320]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_14, 8, '0');  getitem_14 = None

    ... (uses wait_tensor in compute)

    wait_tensor_1: "bf16[51384320]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None

    ... (uses wait_tensor_1 in compute)
    """
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    node_list = list(mod.graph.nodes)
    # First step: identify all nodes that need to be moved.
    all_gather_blocks = []
    j = 0
    while j < len(node_list):
        nj = node_list[j]
        if (
            nj.target is torch.ops.fsdp.all_gather_copy_in.default
            and (node_list[j+1].target is operator.getitem and node_list[j+1].args[0] == nj)
            and (node_list[j+2].target is operator.getitem and node_list[j+2].args[0] == nj)
        ):
            all_gather_copy_in_node = nj
            getitem_0_node = node_list[j+1]
            getitem_1_node = node_list[j+2]

            all_gather_node_start_index = None
            for k in range(j+3, len(node_list)):
                if node_list[k].target is torch.ops._c10d_functional.all_gather_into_tensor.default and node_list[k].args[0] == getitem_0_node:
                    all_gather_node_start_index = k
                    break
            if all_gather_node_start_index is None:
                j += 1
                continue
            all_gather_node = node_list[all_gather_node_start_index]
            assert all_gather_node.target is torch.ops._c10d_functional.all_gather_into_tensor.default
            all_gather_wait_node = node_list[all_gather_node_start_index+1]
            assert all_gather_wait_node.target is torch.ops._c10d_functional.wait_tensor.default

            end_nodes = all_gather_copy_in_node.args[6]
            op_chains_from_primals = _get_op_chains_from_primals(end_nodes, primal_inputs_tensor_only, node_list)
            if len(op_chains_from_primals) != len(end_nodes):
                j += 1
                continue
            all_gather_blocks.append({
                "op_chains_from_primals": op_chains_from_primals,
                "all_gather_copy_in_node": all_gather_copy_in_node,
                "getitem_0_node": getitem_0_node,
                "getitem_1_node": getitem_1_node,
                "all_gather_node": all_gather_node,
                "all_gather_wait_node": all_gather_wait_node,
            })
            j = all_gather_node_start_index + 2
        else:
            j += 1
            continue

    # for block in all_gather_blocks:
    #     torch_log.warning(f"block: {block}")
    # Second step: move the nodes (starting with the last block in graph)
    prev_all_gather_wait_node = None
    for i in range(len(all_gather_blocks)-1, 0, -1):  # range ends at 0 (exclusive), because we don't move the first block in graph
        prev_all_gather_wait_node = all_gather_blocks[i-1]["all_gather_wait_node"]
        # torch_log.warning(f"prev_all_gather_wait_node: {prev_all_gather_wait_node}")
        block = all_gather_blocks[i]
        # torch_log.warning(f"block to move: {block}")
        op_chains_from_primals = block["op_chains_from_primals"]
        all_gather_copy_in_node = block["all_gather_copy_in_node"]
        getitem_0_node = block["getitem_0_node"]
        getitem_1_node = block["getitem_1_node"]
        all_gather_node = block["all_gather_node"]
        all_gather_wait_node = block["all_gather_wait_node"]

        with mod.graph.inserting_after(prev_all_gather_wait_node):
            # NOTE: the last inserted op within `mod.graph.inserting_after` ctx appears *first* in graph.
            new_all_gather_node = _create_new_node_and_replace(mod, all_gather_node, propagate_meta=True)
            new_getitem_1_node = _create_new_node_and_replace(mod, getitem_1_node, propagate_meta=True)
            new_getitem_0_node = _create_new_node_and_replace(mod, getitem_0_node, propagate_meta=True)
            new_all_gather_copy_in_node = _create_new_node_and_replace(mod, all_gather_copy_in_node, propagate_meta=True)
            for op_chain in op_chains_from_primals:
                for op in op_chain[:-1]:  # the last op is just the primal input node, we don't need to move it as it's already at the top of graph.
                    new_op = _create_new_node_and_replace(mod, op, propagate_meta=True)

    mod.graph.lint()
    mod.recompile()


def sink_prev_reduce_scatter_wait_to_before_next_reduce_scatter(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
    reduce_scatter_tensor_1: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_390, 'avg', 8, '0');  view_390 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_20: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None

    ... (as_strided nodes using wait_tensor_20)
    ... (some other nodes)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
    reduce_scatter_tensor_2: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_X, 'avg', 8, '0');  view_390 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_21: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None

    ... (as_strided nodes using wait_tensor_21)
    ... (some other nodes)

    ->

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
    reduce_scatter_tensor_1: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_390, 'avg', 8, '0');  view_390 = None

    ... (some other nodes)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_20: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None

    ... (as_strided nodes using wait_tensor_20)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
    reduce_scatter_tensor_2: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_X, 'avg', 8, '0');  view_390 = None

    ... (some other nodes)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_21: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None

    ... (as_strided nodes using wait_tensor_21)

    ...
    """
    node_list = list(mod.graph.nodes)
    return_op = None
    for node in node_list:
        if node.target == "output":
            return_op = node
            break
    reduce_scatter_wait_blocks = []
    for i, n in enumerate(node_list):
        if n.target == torch.ops._c10d_functional.wait_tensor.default and n.args[0].target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
            reduce_scatter_wait_node = n
            nodes_to_track_users = [reduce_scatter_wait_node]
            nodes_to_track_users_set = set(nodes_to_track_users)
            for j in range(i + 1, len(node_list)):
                nj = node_list[j]
                for arg in _flatten_arg_list(nj.args):
                    if arg in nodes_to_track_users_set:
                        nodes_to_track_users.append(nj)
                        nodes_to_track_users_set.add(nj)
            reduce_scatter_wait_blocks.append({
                "reduce_scatter_wait_node": reduce_scatter_wait_node,
                "downstream_recursive_users_of_rs_wait": nodes_to_track_users[1:],
            })
    for i, block in enumerate(reduce_scatter_wait_blocks):
        reduce_scatter_wait_node = block["reduce_scatter_wait_node"]
        downstream_recursive_users_of_rs_wait = block["downstream_recursive_users_of_rs_wait"]
        if i < len(reduce_scatter_wait_blocks) - 1:
            next_reduce_scatter_node = reduce_scatter_wait_blocks[i+1]["reduce_scatter_wait_node"].args[0]
            insert_before = next_reduce_scatter_node
        else:
            insert_before = return_op
        with mod.graph.inserting_before(insert_before):
            new_reduce_scatter_wait_node = _create_new_node_and_replace(mod, reduce_scatter_wait_node, propagate_meta=True)
            for user in downstream_recursive_users_of_rs_wait:
                if user != return_op:
                    new_user = _create_new_node_and_replace(mod, user, propagate_meta=True)
    mod.graph.lint()
    mod.recompile()


def _propagate_node_meta(src_node, dst_node):
    for k, v in src_node.meta.items():
        dst_node.meta[k] = v


def reinplace_all_gather(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:86 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(16384256, 8, 0, torch.bfloat16, device(type='cuda', index=0), [8192000, 256, 8192000], [convert_element_type, convert_element_type_1, convert_element_type_2]);  convert_element_type = convert_element_type_1 = convert_element_type_2 = None
    getitem: "bf16[16384256]" = all_gather_copy_in[0]
    getitem_1: "bf16[131074048]" = all_gather_copy_in[1];  all_gather_copy_in = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 8, '0');  getitem = None

    ->

    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(16384256, 8, 0, torch.bfloat16, device(type='cuda', index=0), [8192000, 256, 8192000], [convert_element_type, convert_element_type_1, convert_element_type_2]);  convert_element_type = convert_element_type_1 = convert_element_type_2 = None
    getitem: "bf16[16384256]" = all_gather_copy_in[0]
    getitem_1: "bf16[131074048]" = all_gather_copy_in[1];  all_gather_copy_in = None

    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor_.default(getitem_1, getitem, 8, '0');  getitem = None
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target == torch.ops.fsdp.all_gather_copy_in.default:
            all_gather_copy_in_node = n
            getitem_0_node = node_list[i+1]
            assert getitem_0_node.target == operator.getitem and getitem_0_node.args[0] == all_gather_copy_in_node
            getitem_1_node = node_list[i+2]
            assert getitem_1_node.target == operator.getitem and getitem_1_node.args[0] == all_gather_copy_in_node
            all_gather_node_start_index = None
            for k in range(i+3, len(node_list)):
                if node_list[k].target is torch.ops._c10d_functional.all_gather_into_tensor.default and node_list[k].args[0] == getitem_0_node:
                    all_gather_node_start_index = k
                    break
            if all_gather_node_start_index is None:
                continue
            all_gather_node = node_list[all_gather_node_start_index]

            with mod.graph.inserting_before(all_gather_node):
                inplace_all_gather_node = mod.graph.call_function(torch.ops._c10d_functional.all_gather_into_tensor_.default, (getitem_1_node, getitem_0_node, *all_gather_node.args[1:]))
            all_gather_node.replace_all_uses_with(inplace_all_gather_node, propagate_meta=True)
            mod.graph.erase_node(all_gather_node)
    mod.graph.lint()
    mod.recompile()
