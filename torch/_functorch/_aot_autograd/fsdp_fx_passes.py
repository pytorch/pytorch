import torch
import operator
import copy
from collections import defaultdict


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


def _input_is_used_in_view_ops(ops, inp_n):
    for n in ops:
        if inp_n in _flatten_arg_list(n.args) and inp_n.target in _view_ops:
            return True
    return False


def reinplace_foreach_copy_if_input_has_no_other_aliases_in_graph(mod):
    """
    _foreach_copy_1 = torch.ops.aten._foreach_copy.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    getitem_44: "f32[2, 76137800]" = _foreach_copy_1[0]
    getitem_45: "f32[2, 6170]" = _foreach_copy_1[1]
    getitem_46: "f32[2, 76137800]" = _foreach_copy_1[2]
    getitem_47: "f32[2, 6170]" = _foreach_copy_1[3];  _foreach_copy_1 = None

    ->

    # _foreach_copy__1 = torch.ops.aten._foreach_copy_.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    ... = torch.ops.aten.copy_.default(primal_1, getitem_28)
    ... = torch.ops.aten.copy_.default(primal_2, getitem_33)
    ... = torch.ops.aten.copy_.default(primal_3, getitem_38)
    ... = torch.ops.aten.copy_.default(primal_4, getitem_43)
    """
    # TODO: this pass is maybe super slow, need optimization
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten._foreach_copy.default:
            _foreach_copy_outplace_node = n
            if all(not _input_is_used_in_view_ops(
                node_list,
                inp_n,
            ) for inp_ind, inp_n in enumerate(_foreach_copy_outplace_node.args[0])):
                _foreach_copy_outplace_node.target = torch.ops.aten._foreach_copy_.default
                # with mod.graph.inserting_before(_foreach_copy_outplace_node):
                #     for i, arg in enumerate(_foreach_copy_outplace_node.args[0]):
                #         copy_to = arg
                #         copy_from = _foreach_copy_outplace_node.args[1][i]
                #         # _foreach_copy_inplace_node = mod.graph.call_function(torch.ops.aten._foreach_copy_.default, _foreach_copy_outplace_node.args, {})
                #         # NOTE: Inductor seems to fail when encountering `_foreach_copy_` op. Need more investigation.
                #         mod.graph.call_function(torch.ops.aten.copy_.default, (copy_to, copy_from), {})
                getitem_nodes = set()
                for node in node_list[i+1:]:
                    if node.target is operator.getitem and node.args[0] == _foreach_copy_outplace_node:
                        getitem_nodes.add(node)
                for node in getitem_nodes:
                    node.replace_all_uses_with(_foreach_copy_outplace_node.args[0][node.args[1]])
                    mod.graph.erase_node(node)
                # mod.graph.erase_node(_foreach_copy_outplace_node)
                mod.graph.lint()
                mod.recompile()


def _find_next_use_of_node(node_list, node):
    for i, n in enumerate(node_list):
        if node in _flatten_arg_list(n.args):
            return n
    return None


def remove_no_use_slice(mod):
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target == torch.ops.aten.slice.Tensor:
            next_use_of_node = _find_next_use_of_node(node_list[i+1:], n)
            if next_use_of_node is None:
                mod.graph.erase_node(n)
    mod.graph.lint()
    mod.recompile()


def remove_no_use_empty(mod):
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target == torch.ops.aten.empty.memory_format:
            next_use_of_node = _find_next_use_of_node(node_list[i+1:], n)
            if next_use_of_node is None:
                mod.graph.erase_node(n)
    mod.graph.lint()
    mod.recompile()


def remove_no_use_reshape(mod):
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target == torch.ops.aten.reshape.default:
            next_use_of_node = _find_next_use_of_node(node_list[i+1:], n)
            if next_use_of_node is None:
                mod.graph.erase_node(n)
    mod.graph.lint()
    mod.recompile()


def remove_unnecessary_split_with_sizes(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor: "f32[3047980]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:128 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
    view_1: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1])
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_1, [761378, 617, 761378, 617], 1);  view_1 = None
    getitem_28: "f32[2, 761378]" = split_with_sizes_6[0];  split_with_sizes_6 = None

    view_3: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1])
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_3, [761378, 617, 761378, 617], 1);  view_3 = None
    getitem_33: "f32[2, 617]" = split_with_sizes_7[1];  split_with_sizes_7 = None

    view_5: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1])
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_5, [761378, 617, 761378, 617], 1);  view_5 = None
    getitem_38: "f32[2, 761378]" = split_with_sizes_8[2];  split_with_sizes_8 = None

    view_7: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_7, [761378, 617, 761378, 617], 1);  view_7 = None
    getitem_43: "f32[2, 617]" = split_with_sizes_9[3];  split_with_sizes_9 = None

    ->

    wait_tensor: "f32[3047980]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    view_1: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1])
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_1, [761378, 617, 761378, 617], 1);  view_1 = None
    getitem_28: "f32[2, 761378]" = split_with_sizes_6[0];  split_with_sizes_6 = None
    getitem_33: "f32[2, 617]" = split_with_sizes_6[1];  split_with_sizes_7 = None
    getitem_38: "f32[2, 761378]" = split_with_sizes_6[2];  split_with_sizes_8 = None
    getitem_43: "f32[2, 617]" = split_with_sizes_6[3];  split_with_sizes_9 = None
    """
    node_list = list(mod.graph.nodes)
    return_op = None
    for n in node_list:
        if n.op == "output":
            return_op = n
            break
    for i, n in enumerate(node_list):
        if (
            n.target is torch.ops._c10d_functional.wait_tensor.default \
            and (node_list[i+1].target is torch.ops.aten.reshape.default and node_list[i+1].args[0] == n) \
            and (node_list[i+2].target is torch.ops.aten.split_with_sizes.default and node_list[i+2].args[0] == node_list[i+1]) \
            and (node_list[i+3].target is operator.getitem and node_list[i+3].args[0] == node_list[i+2]) \
            and (node_list[i+4].target is torch.ops.aten.reshape.default and node_list[i+4].args[0] == n) \
            and (node_list[i+5].target is torch.ops.aten.split_with_sizes.default and node_list[i+5].args[0] == node_list[i+4]) \
        ):
            first_view_node_index = i+1
            first_split_with_sizes_node = node_list[i+2]
            num_blocks = len(first_split_with_sizes_node.args[1])
            if not all(
                oc[0].target is torch.ops.aten.reshape.default \
                and oc[1].target is torch.ops.aten.split_with_sizes.default \
                and oc[2].target is operator.getitem \
                for oc in _get_chunks(node_list[first_view_node_index:first_view_node_index+num_blocks*3], 3)
            ):
                continue
            for j in range(1, num_blocks):
                view_node = node_list[first_view_node_index+j*3]
                split_with_sizes_node = node_list[first_view_node_index+j*3+1]
                getitem_node = node_list[first_view_node_index+j*3+2]
                assert getitem_node.target is operator.getitem
                next_use_of_getitem_node = _find_next_use_of_node(node_list[first_view_node_index+j*3+3:], getitem_node)
                assert next_use_of_getitem_node is not None
                with mod.graph.inserting_before(next_use_of_getitem_node):
                    new_getitem_node = mod.graph.call_function(operator.getitem, (first_split_with_sizes_node, j), {})
                getitem_node.replace_all_uses_with(new_getitem_node)
                mod.graph.erase_node(getitem_node)
                mod.graph.erase_node(split_with_sizes_node)
                mod.graph.erase_node(view_node)
    mod.graph.lint()
    mod.recompile()


def replace_foreach_all_gather_copy_out_pattern(mod):
    """
    NOTE: this pattern is from `foreach_all_gather_copy_out` in ppFSDP:
    ```
    ... = dist.all_gather_into_tensor(...)  # in another function
    all_gather_output = all_gather_output.view(world_size, -1)
    splits = torch.split(all_gather_output, all_gather_input_numels, dim=1)
    for i, fsdp_param in enumerate(fsdp_params):
        splits_unpadded.append(
            torch.as_strided(
                splits[i].contiguous().view(splits[i].numel()),
                ...
            )
        )
    with torch.no_grad():
        torch._foreach_copy_(out, splits_unpadded)
    ```

    ====== Pattern #1: len(splits) > 1 ======

    wait_tensor: "f32[3047980]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    view_1: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_1, [761378, 617, 761378, 617], 1);  view_1 = None

    getitem_28: "f32[2, 761378]" = split_with_sizes_6[0]
    clone: "f32[2, 761378]" = torch.ops.aten.clone.default(getitem_28, memory_format = torch.contiguous_format);  getitem_28 = None
    view_2: "f32[1522756]" = torch.ops.aten.reshape.default(clone, [1522756]);  clone = None
    as_strided: "f32[1234, 1234]" = torch.ops.aten.as_strided.default(view_2, [1234, 1234], [1234, 1], 0);  view_2 = None

    getitem_33: "f32[2, 617]" = split_with_sizes_6[1]
    clone_1: "f32[2, 617]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format);  getitem_33 = None
    view_4: "f32[1234]" = torch.ops.aten.reshape.default(clone_1, [1234]);  clone_1 = None
    as_strided_1: "f32[1234]" = torch.ops.aten.as_strided.default(view_4, [1234], [1], 0);  view_4 = None

    getitem_38: "f32[2, 761378]" = split_with_sizes_6[2]
    clone_2: "f32[2, 761378]" = torch.ops.aten.clone.default(getitem_38, memory_format = torch.contiguous_format);  getitem_38 = None
    view_6: "f32[1522756]" = torch.ops.aten.reshape.default(clone_2, [1522756]);  clone_2 = None
    as_strided_2: "f32[1234, 1234]" = torch.ops.aten.as_strided.default(view_6, [1234, 1234], [1234, 1], 0);  view_6 = None

    getitem_43: "f32[2, 617]" = split_with_sizes_6[3];  split_with_sizes_6 = None
    clone_3: "f32[2, 617]" = torch.ops.aten.clone.default(getitem_43, memory_format = torch.contiguous_format);  getitem_43 = None
    view_8: "f32[1234]" = torch.ops.aten.reshape.default(clone_3, [1234]);  clone_3 = None
    as_strided_3: "f32[1234]" = torch.ops.aten.as_strided.default(view_8, [1234], [1], 0);  view_8 = None

    _foreach_copy_1 = torch.ops.aten._foreach_copy_.default([primals_6, primals_7, primals_8, primals_9], [as_strided, as_strided_1, as_strided_2, as_strided_3]);  as_strided = as_strided_1 = as_strided_2 = as_strided_3 = None

    ... (uses primals_6, primals_7, primals_8, primals_9)

    ->

    wait_tensor: "f32[3047980]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    view_1: "f32[2, 1523990]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_1, [761378, 617, 761378, 617], 1);  view_1 = None

    getitem_28: "f32[2, 761378]" = split_with_sizes_6[0]
    view_2: "f32[1522756]" = torch.ops.aten.reshape.default(getitem_28, [1522756])

    as_strided: "f32[1234, 1234]" = torch.ops.aten.as_strided.default(view_2, [1234, 1234], [1234, 1], 0);  view_2 = None

    getitem_33: "f32[2, 617]" = split_with_sizes_6[1]
    view_4: "f32[1234]" = torch.ops.aten.reshape.default(getitem_33, [1234])

    as_strided_1: "f32[1234]" = torch.ops.aten.as_strided.default(view_4, [1234], [1], 0);  view_4 = None

    getitem_38: "f32[2, 761378]" = split_with_sizes_6[2]
    view_6: "f32[1522756]" = torch.ops.aten.reshape.default(getitem_38, [1522756])

    as_strided_2: "f32[1234, 1234]" = torch.ops.aten.as_strided.default(view_6, [1234, 1234], [1234, 1], 0);  view_6 = None

    getitem_43: "f32[2, 617]" = split_with_sizes_6[3];  split_with_sizes_6 = None
    view_8: "f32[1234]" = torch.ops.aten.reshape.default(getitem_43, [1234])

    as_strided_3: "f32[1234]" = torch.ops.aten.as_strided.default(view_8, [1234], [1], 0);  view_8 = None

    ... (uses as_strided, as_strided_1, as_strided_2, as_strided_3. But make sure to keep the primals in graph output list)


    ====== Pattern #2: len(splits) == 1 ======

    wait_tensor: "f32[1024]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:138 in foreach_all_gather_copy_out, code: torch._foreach_copy_(out, splits_unpadded)
    view_2: "f32[2, 512]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
    view_3: "f32[1024]" = torch.ops.aten.reshape.default(view_2, [1024]);  view_2 = None
    as_strided_1: "f32[32, 32]" = torch.ops.aten.as_strided.default(view_3, [32, 32], [32, 1], 0);  view_3 = None
    _foreach_copy_1 = torch.ops.aten._foreach_copy_.default([primals_3], [as_strided_1]);  as_strided_1 = None

    ... (uses primals_3)

    ->

    wait_tensor: "f32[1024]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:138 in foreach_all_gather_copy_out, code: torch._foreach_copy_(out, splits_unpadded)
    view_2: "f32[2, 512]" = torch.ops.aten.reshape.default(wait_tensor, [2, -1]);  wait_tensor = None
    view_3: "f32[1024]" = torch.ops.aten.reshape.default(view_2, [1024]);  view_2 = None
    as_strided_1: "f32[32, 32]" = torch.ops.aten.as_strided.default(view_3, [32, 32], [32, 1], 0);  view_3 = None

    ... (uses as_strided_1)
    """

    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    node_list = list(mod.graph.nodes)
    return_op = None
    for n in node_list:
        if n.op == "output":
            return_op = n
            break
    for i, n in enumerate(node_list):
        if (
            n.target is torch.ops._c10d_functional.wait_tensor.default \
            and (node_list[i+1].target is torch.ops.aten.reshape.default and node_list[i+1].args[0] == n)
        ):
            if node_list[i+2].target is torch.ops.aten.split_with_sizes.default and node_list[i+2].args[0] == node_list[i+1]:
                # If there is `split_with_sizes` in graph (i.e. len(splits) > 1)
                split_with_sizes_node = node_list[i+2]
                block_seq_start_index = i+3
                num_blocks = len(split_with_sizes_node.args[1])
                for j in range(num_blocks):
                    clone_node = node_list[block_seq_start_index + j * 4 + 1]
                    assert clone_node.target is torch.ops.aten.clone.default, f"mod: {mod}, clone_node.target: {clone_node.target}"
                    clone_input = clone_node.args[0]
                    clone_node.replace_all_uses_with(clone_input)
                    mod.graph.erase_node(clone_node)
                foreach_copy_start_index = block_seq_start_index + num_blocks * 4
                foreach_copy_node = node_list[foreach_copy_start_index]
                assert foreach_copy_node.target is torch.ops.aten._foreach_copy_.default
                for copy_into_node, copy_from_node in zip(foreach_copy_node.args[0], foreach_copy_node.args[1]):
                    assert copy_into_node in primal_inputs_tensor_only
                    # TODO(yf225): since we don't know from the graph that primals are size-0 in graph output,
                    # how do we make sure it is sound that we don't replace primal with as_strided_X in graph output?
                    copy_into_node.replace_all_uses_with(
                        copy_from_node,
                        delete_user_cb=lambda n: n != return_op,
                    )
                mod.graph.erase_node(foreach_copy_node)
            elif (
                node_list[i+2].target is torch.ops.aten.reshape.default and node_list[i+2].args[0] == node_list[i+1]
                and node_list[i+3].target is torch.ops.aten.as_strided.default and node_list[i+3].args[0] == node_list[i+2]
                and node_list[i+4].target is torch.ops.aten._foreach_copy_.default and node_list[i+4].args[1][0] == node_list[i+3]
            ):
                # If there is no `split_with_sizes` in graph (i.e. len(splits) == 1)
                foreach_copy_node = node_list[i+4]
                copy_into_node = foreach_copy_node.args[0][0]
                copy_from_node = foreach_copy_node.args[1][0]
                assert copy_into_node in primal_inputs_tensor_only
                # TODO(yf225): since we don't know from the graph that primals are size-0 in graph output,
                # how do we make sure it is sound that we don't replace primal with as_strided_X in graph output?
                copy_into_node.replace_all_uses_with(
                    copy_from_node,
                    delete_user_cb=lambda n: n != return_op,
                )
                mod.graph.erase_node(foreach_copy_node)
    mod.graph.lint()
    mod.recompile()



def undo_functionalization_for_split_with_sizes_then_inplace_foreach_copy(mod):
    """
    NOTE: replace this pattern in `foreach_all_gather()` in _fsdp_collectives.py:
    ```
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    ```
    """
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:53 in foreach_all_gather, code: foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [761378, 617, 761378, 617]);  slice_1 = None
    getitem: "f32[761378]" = split_with_sizes[0]
    getitem_1: "f32[617]" = split_with_sizes[1]
    getitem_2: "f32[761378]" = split_with_sizes[2]
    getitem_3: "f32[617]" = split_with_sizes[3]

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:55 in foreach_all_gather, code: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    _foreach_copy = torch.ops.aten._foreach_copy_.default([getitem, getitem_1, getitem_2, getitem_3], [primals_2, primals_3, primals_4, primals_5]);  primals_2 = primals_3 = primals_4 = primals_5 = None

    # No stacktrace found for following nodes
    slice_tensor: "f32[1523990]" = torch.ops.aten.slice.Tensor(empty, 0, 1523990, 3047980)
    slice_scatter_default: "f32[1523990]" = torch.ops.aten.slice_scatter.default(slice_tensor, getitem, 0, 0, 761378);  slice_tensor = getitem = None
    slice_scatter_default_1: "f32[3047980]" = torch.ops.aten.slice_scatter.default(empty, slice_scatter_default, 0, 1523990, 3047980);  empty = slice_scatter_default = None
    slice_tensor_1: "f32[1523990]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 1523990, 3047980)
    slice_scatter_default_2: "f32[1523990]" = torch.ops.aten.slice_scatter.default(slice_tensor_1, getitem_1, 0, 761378, 761995);  slice_tensor_1 = getitem_1 = None
    slice_scatter_default_3: "f32[3047980]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 0, 1523990, 3047980);  slice_scatter_default_1 = slice_scatter_default_2 = None
    slice_tensor_2: "f32[1523990]" = torch.ops.aten.slice.Tensor(slice_scatter_default_3, 0, 1523990, 3047980)
    slice_scatter_default_4: "f32[1523990]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, getitem_2, 0, 761995, 1523373);  slice_tensor_2 = getitem_2 = None
    slice_scatter_default_5: "f32[3047980]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_3, slice_scatter_default_4, 0, 1523990, 3047980);  slice_scatter_default_3 = slice_scatter_default_4 = None
    slice_tensor_3: "f32[1523990]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1523990, 3047980)
    slice_scatter_default_6: "f32[1523990]" = torch.ops.aten.slice_scatter.default(slice_tensor_3, getitem_3, 0, 1523373, 1523990);  slice_tensor_3 = full_default = None
    slice_scatter_default_7: "f32[3047980]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_5, slice_scatter_default_6, 0, 1523990, 3047980);  slice_scatter_default_5 = slice_scatter_default_6 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
    slice_10: "f32[1523990]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 1523990, 3047980);  slice_scatter_default_7 = None

    ... (uses slice_10)

    ->

    split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [761378, 617, 761378, 617]);  slice_1 = None
    getitem: "f32[761378]" = split_with_sizes[0]
    getitem_1: "f32[617]" = split_with_sizes[1]
    getitem_2: "f32[761378]" = split_with_sizes[2]
    getitem_3: "f32[617]" = split_with_sizes[3]

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:55 in foreach_all_gather, code: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    _foreach_copy = torch.ops.aten._foreach_copy_.default([getitem, getitem_1, getitem_2, getitem_3], [primals_2, primals_3, primals_4, primals_5]);  primals_2 = primals_3 = primals_4 = primals_5 = None

    ... (uses slice_1)
    """
    node_list = list(mod.graph.nodes)
    for i in range(len(node_list)):
        n = node_list[i]
        if (
            n.target is torch.ops.aten.split_with_sizes.default \
            and node_list[i+len(n.args[1])+1].target is torch.ops.aten._foreach_copy_.default
        ):
            split_with_sizes_node = n
            # s -> ss -> ss -> s
            num_blocks = len(split_with_sizes_node.args[1])
            if i + 2 + num_blocks * 4 >= len(node_list):  # each pattern block contains: split_with_sizes * 1 + _foreach_copy_ * 1 + (getitem + slice + slice_scatter + slice_scatter) * num_block + slice * 1
                continue
            getitem_start_index = i + 1
            if not all(
                node.target is operator.getitem and node.args[0] == split_with_sizes_node
                for node in node_list[getitem_start_index:getitem_start_index+num_blocks]
            ):
                continue
            block_start_index = i+len(n.args[1])+2
            if not all(
                oc[0].target is torch.ops.aten.slice.Tensor \
                and oc[1].target is torch.ops.aten.slice_scatter.default \
                and oc[2].target is torch.ops.aten.slice_scatter.default \
                for oc in _get_chunks(node_list[block_start_index:block_start_index+num_blocks*3], 3)
            ):
                continue
            last_slice_start_index = block_start_index + num_blocks * 3
            last_slice_node = node_list[last_slice_start_index]
            assert last_slice_node.target is torch.ops.aten.slice.Tensor, f"Expected torch.ops.aten.slice.Tensor but got {last_slice_node.target}"
            last_slice_node.replace_all_uses_with(split_with_sizes_node.args[0])
            mod.graph.erase_node(last_slice_node)
            for node in reversed(node_list[block_start_index:block_start_index+num_blocks*3]):
                mod.graph.erase_node(node)
            mod.graph.lint()
            mod.recompile()


def replace_inplace_foreach_copy_with_inplace_copy(mod):
    # NOTE: Inductor seems to fail when encountering `_foreach_copy_` op, so we need this FX pass.
    """
    _foreach_copy__1 = torch.ops.aten._foreach_copy_.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None

    ->

    ... = torch.ops.aten.copy_.default(primal_1, getitem_28)
    ... = torch.ops.aten.copy_.default(primal_2, getitem_33)
    ... = torch.ops.aten.copy_.default(primal_3, getitem_38)
    ... = torch.ops.aten.copy_.default(primal_4, getitem_43)
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten._foreach_copy_.default:
            with mod.graph.inserting_before(n):
                for copy_into, copy_from in zip(n.args[0], n.args[1]):
                    mod.graph.call_function(torch.ops.aten.copy_.default, (copy_into, copy_from), {})
            mod.graph.erase_node(n)
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


def remove_clone_if_input_is_alias_of_graph_input(mod):
    # TODO(yf225): is it safe to assume that graph input will not be mutated in the graph? or do we need to check that the graph input is not copy_ into at end of graph?
    """
    expand: "f32[2, 1234]" = torch.ops.aten.expand.default(arg0_1, [2, 1234]);  arg0_1 = None
    clone: "f32[2, 12340]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    ... (uses clone)

    ->

    ... (uses expand)
    """
    graph_inputs = set()
    for n in mod.graph.nodes:
        if n.op == "placeholder":
            graph_inputs.add(n.target)
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.clone.default:
            clone_inp = n.args[0]
            is_alias, view_chain = _is_alias_of_graph_input(mod.graph, graph_inputs, clone_inp)
            if is_alias:
                n.replace_all_uses_with(clone_inp)
                mod.graph.erase_node(n)
    mod.graph.lint()
    mod.recompile()

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


def replace_empty_then_slice_then_compute_then_foreach_copy_then_slice_scatter_pattern(mod):
    """
    NOTE: pattern from `foreach_reduce_scatter_copy_in`:
    ```
    for grad in unsharded_grads:
        grad_size = grad.size()
        dim0_padded_size = _get_dim0_padded_size(grad_size, world_size)
        if dim0_padded_size != grad_size:
            padded_grad = grad.new_empty(dim0_padded_size)
            padded_grad_slices.append(padded_grad[: grad.size(0)])
            grads_to_copy.append(grad)
            grad = padded_grad
        grad_views.append(grad.view(world_size, -1))
    if padded_grad_slices:
        with torch.no_grad():
            torch._foreach_copy_(padded_grad_slices, grads_to_copy)
    ```
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:252 in foreach_reduce_scatter_copy_in, code: padded_grad = grad.new_empty(dim0_padded_size)
    empty_2: "f32[16, 7]" = torch.ops.aten.empty.memory_format([16, 7], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:253 in foreach_reduce_scatter_copy_in, code: padded_grad_slices.append(padded_grad[: grad.size(0)])
    slice_11: "f32[15, 7]" = torch.ops.aten.slice.Tensor(empty_2, 0, 0, 15)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:252 in foreach_reduce_scatter_copy_in, code: padded_grad = grad.new_empty(dim0_padded_size)
    empty_3: "f32[16]" = torch.ops.aten.empty.memory_format([16], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:253 in foreach_reduce_scatter_copy_in, code: padded_grad_slices.append(padded_grad[: grad.size(0)])
    slice_12: "f32[15]" = torch.ops.aten.slice.Tensor(empty_3, 0, 0, 15)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:252 in foreach_reduce_scatter_copy_in, code: padded_grad = grad.new_empty(dim0_padded_size)
    empty_4: "f32[4, 15]" = torch.ops.aten.empty.memory_format([4, 15], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:253 in foreach_reduce_scatter_copy_in, code: padded_grad_slices.append(padded_grad[: grad.size(0)])
    slice_13: "f32[3, 15]" = torch.ops.aten.slice.Tensor(empty_4, 0, 0, 3)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:252 in foreach_reduce_scatter_copy_in, code: padded_grad = grad.new_empty(dim0_padded_size)
    empty_5: "f32[4]" = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:253 in foreach_reduce_scatter_copy_in, code: padded_grad_slices.append(padded_grad[: grad.size(0)])
    slice_14: "f32[3]" = torch.ops.aten.slice.Tensor(empty_5, 0, 0, 3)

    ...

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:259 in foreach_reduce_scatter_copy_in, code: torch._foreach_copy_(padded_grad_slices, grads_to_copy)
    _foreach_copy_2 = torch.ops.aten._foreach_copy.default([slice_11, slice_12, slice_13, slice_14], [mm_2, view_10, mm_1, view_9]);  slice_11 = slice_12 = slice_13 = slice_14 = mm_2 = view_10 = mm_1 = view_9 = None
    getitem_48: "f32[15, 7]" = _foreach_copy_2[0]
    getitem_49: "f32[15]" = _foreach_copy_2[1]
    getitem_50: "f32[3, 15]" = _foreach_copy_2[2]
    getitem_51: "f32[3]" = _foreach_copy_2[3];  _foreach_copy_2 = None

    # No stacktrace found for following nodes
    slice_scatter_default_8: "f32[16, 7]" = torch.ops.aten.slice_scatter.default(empty_2, getitem_48, 0, 0, 15);  empty_2 = getitem_48 = None

    ... (uses slice_scatter_default_8)

    # No stacktrace found for following nodes
    slice_scatter_default_9: "f32[16]" = torch.ops.aten.slice_scatter.default(empty_3, getitem_49, 0, 0, 15);  empty_3 = getitem_49 = None

    ... (uses slice_scatter_default_9)

    # No stacktrace found for following nodes
    slice_scatter_default_10: "f32[4, 15]" = torch.ops.aten.slice_scatter.default(empty_4, getitem_50, 0, 0, 3);  empty_4 = getitem_50 = None

    ... (uses slice_scatter_default_10)

    # No stacktrace found for following nodes
    slice_scatter_default_11: "f32[4]" = torch.ops.aten.slice_scatter.default(empty_5, getitem_51, 0, 0, 3);  empty_5 = getitem_51 = None

    ... (uses slice_scatter_default_11)

    ->

    empty_2: "f32[16, 7]" = torch.ops.aten.empty.memory_format([16, 7], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
    slice_11: "f32[15, 7]" = torch.ops.aten.slice.Tensor(empty_2, 0, 0, 15)
    empty_3: "f32[16]" = torch.ops.aten.empty.memory_format([16], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
    slice_12: "f32[15]" = torch.ops.aten.slice.Tensor(empty_3, 0, 0, 15)
    empty_4: "f32[4, 15]" = torch.ops.aten.empty.memory_format([4, 15], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
    slice_13: "f32[3, 15]" = torch.ops.aten.slice.Tensor(empty_4, 0, 0, 3)
    empty_5: "f32[4]" = torch.ops.aten.empty.memory_format([4], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), pin_memory = False)
    slice_14: "f32[3]" = torch.ops.aten.slice.Tensor(empty_5, 0, 0, 3)
    ...
    _foreach_copy_2 = torch.ops.aten._foreach_copy_.default([slice_11, slice_12, slice_13, slice_14], [mm_2, view_10, mm_1, view_9]);  slice_11 = slice_12 = slice_13 = slice_14 = mm_2 = view_10 = mm_1 = view_9 = None
    ... (uses empty_2)
    ... (uses empty_3)
    ... (uses empty_4)
    ... (uses empty_5)
    """
    node_list = list(mod.graph.nodes)
    node_type_to_node_set, node_to_node_index = _collect_node_types_and_indices(mod)
    for foreach_copy_node in node_type_to_node_set[torch.ops.aten._foreach_copy.default]:
        foreach_copy_node_index = node_to_node_index[foreach_copy_node]
        foreach_copy_to = foreach_copy_node.args[0]
        num_blocks = len(foreach_copy_to)
        getitem_nodes = node_list[foreach_copy_node_index+1:foreach_copy_node_index+num_blocks+1]
        assert all(node.target is operator.getitem for node in getitem_nodes)
        mismatch = False
        empty_nodes = []
        ss_nodes = []
        for i, slice_node in enumerate(foreach_copy_to):
            if not (slice_node.target is torch.ops.aten.slice.Tensor and slice_node.args[0].target is torch.ops.aten.empty.memory_format):
                mismatch = True
                break
            empty_node = slice_node.args[0]
            getitem_node = getitem_nodes[i]
            ss_node = None
            for node in node_type_to_node_set[torch.ops.aten.slice_scatter.default]:
                if node.args[0] == empty_node and node.args[1] == getitem_node:
                    ss_node = node
                    break
            if not ss_node:
                mismatch = True
                break
            empty_nodes.append(empty_node)
            ss_nodes.append(ss_node)
        if mismatch:
            continue
        # We have a match!
        for node in ss_nodes:
            node.replace_all_uses_with(node.args[0])
            mod.graph.erase_node(node)
        foreach_copy_node.target = torch.ops.aten._foreach_copy_.default
        for node in getitem_nodes:
            node.replace_all_uses_with(foreach_copy_node.args[0][node.args[1]])
            mod.graph.erase_node(node)
    mod.graph.lint()
    mod.recompile()
