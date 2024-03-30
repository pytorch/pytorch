import torch
import operator
import copy
import logging
from collections import defaultdict

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
    TODO(yf225): in eager code, can we replace this entire pattern with an ATen op? e.g. https://github.com/pytorch/pytorch/pull/121081. Simplest is just to add a C++ version.
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
                assert foreach_copy_node.target is torch.ops.aten._foreach_copy_.default, f"got: {foreach_copy_node.target}, from {foreach_copy_node}"
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



def replace_foreach_all_gather_pattern(mod):
    """
    TODO(yf225): in eager code, can we replace this entire pattern with an ATen op? e.g. https://github.com/pytorch/pytorch/pull/121081. Simplest is just to add a C++ version.
    NOTE: replace this pattern in `foreach_all_gather()` in _fsdp_collectives.py:
    ```
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    ```
    """
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:47 in foreach_all_gather, code: all_gather_output = torch.empty(
    empty: "f32[3047980]" = torch.ops.aten.empty.memory_format([3047980], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:50 in foreach_all_gather, code: all_gather_input = all_gather_output.narrow(
    slice_1: "f32[1523990]" = torch.ops.aten.slice.Tensor(empty, 0, 1523990, 3047980)

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

    ... (another _foreach_copy op writing into [getitem, getitem_1, getitem_2, getitem_3], followed by a list of slice and slice_scatter ops, ends with slice_20 that is the same range as slice_1)

    ... (uses slice_20)

    ->

    empty: "f32[3047980]" = torch.ops.aten.empty.memory_format([3047980], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
    slice_1: "f32[1523990]" = torch.ops.aten.slice.Tensor(empty, 0, 1523990, 3047980)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [761378, 617, 761378, 617]);  slice_1 = None
    getitem: "f32[761378]" = split_with_sizes[0]
    getitem_1: "f32[617]" = split_with_sizes[1]
    getitem_2: "f32[761378]" = split_with_sizes[2]
    getitem_3: "f32[617]" = split_with_sizes[3]

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:55 in foreach_all_gather, code: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
    _foreach_copy = torch.ops.aten._foreach_copy_.default([getitem, getitem_1, getitem_2, getitem_3], [primals_2, primals_3, primals_4, primals_5]);  primals_2 = primals_3 = primals_4 = primals_5 = None

    ... (uses slice_1)

    ... (another _foreach_copy op writing into [getitem, getitem_1, getitem_2, getitem_3])

    ... (uses slice_1)
    """
    node_list = list(mod.graph.nodes)
    for i in range(len(node_list)):
        n = node_list[i]
        if (
            n.target is torch.ops.aten.empty.memory_format
            and (node_list[i+1].target is torch.ops.aten.slice.Tensor and node_list[i+1].args[0] == n)
            and (node_list[i+2].target is torch.ops.aten.split_with_sizes.default and node_list[i+2].args[0] == node_list[i+1])
        ):
            empty_node = n
            slice_after_empty_node = node_list[i+1]
            split_with_sizes_node_idx = i+2
            split_with_sizes_node = node_list[split_with_sizes_node_idx]
            num_blocks = len(split_with_sizes_node.args[1])
            getitem_start_index = split_with_sizes_node_idx + 1
            if not all(
                node.target is operator.getitem and node.args[0] == split_with_sizes_node
                for node in node_list[getitem_start_index:getitem_start_index+num_blocks]
            ):
                continue
            getitem_nodes = node_list[getitem_start_index:getitem_start_index+num_blocks]
            for j in range(getitem_start_index + num_blocks, len(node_list)):
                nj = node_list[j]
                if nj.target is torch.ops.aten._foreach_copy_.default and all(arg == getitem_node for arg, getitem_node in zip(nj.args[0], getitem_nodes)):
                    for k in range(j+1, len(node_list)):
                        nk = node_list[k]
                        if nk.target is torch.ops.aten.slice.Tensor and nk.args[0] == empty_node:
                            block_start_index = k
                            if all(
                                oc[0].target is torch.ops.aten.slice.Tensor \
                                and oc[1].target is torch.ops.aten.slice_scatter.default \
                                and oc[2].target is torch.ops.aten.slice_scatter.default \
                                for oc in _get_chunks(node_list[block_start_index:block_start_index+num_blocks*3], 3)
                            ):
                                last_slice_start_index = block_start_index + num_blocks * 3
                                last_slice_node = node_list[last_slice_start_index]
                                if not last_slice_node.target is torch.ops.aten.slice.Tensor or last_slice_node.args[1:] != slice_after_empty_node.args[1:]:
                                    continue
                                last_slice_node.replace_all_uses_with(slice_after_empty_node)
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

def sink_reduce_scatter_wait_to_end_of_graph(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
    reduce_scatter_tensor_1: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_390, 'avg', 8, '0');  view_390 = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
    wait_tensor_20: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None

    # No stacktrace found for following nodes
    as_strided_321: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 0)
    as_strided_322: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 524288)
    as_strided_323: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 1048576)
    as_strided_324: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 1572864)
    as_strided_325: "f32[704, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [704, 2048], [2048, 1], 2097152)
    as_strided_326: "f32[256, 5632]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 5632], [5632, 1], 3538944)
    as_strided_327: "f32[704, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [704, 2048], [2048, 1], 4980736)
    as_strided_328: "f32[256]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256], [1], 6422528)
    as_strided_329: "f32[256]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256], [1], 6422784);  wait_tensor_20 = None

    ... (some other nodes)

    return [as_strided_321, as_strided_322, as_strided_323, as_strided_324, as_strided_325, as_strided_326, as_strided_327, as_strided_328, as_strided_329]

    ->

    reduce_scatter_tensor_1: "f32[6423040]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_390, 'avg', 8, '0');  view_390 = None

    ... (some other nodes)

    wait_tensor_20: "f32[6423040]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
    as_strided_321: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 0)
    as_strided_322: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 524288)
    as_strided_323: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 1048576)
    as_strided_324: "f32[256, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 2048], [2048, 1], 1572864)
    as_strided_325: "f32[704, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [704, 2048], [2048, 1], 2097152)
    as_strided_326: "f32[256, 5632]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256, 5632], [5632, 1], 3538944)
    as_strided_327: "f32[704, 2048]" = torch.ops.aten.as_strided.default(wait_tensor_20, [704, 2048], [2048, 1], 4980736)
    as_strided_328: "f32[256]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256], [1], 6422528)
    as_strided_329: "f32[256]" = torch.ops.aten.as_strided.default(wait_tensor_20, [256], [1], 6422784);  wait_tensor_20 = None
    return [as_strided_321, as_strided_322, as_strided_323, as_strided_324, as_strided_325, as_strided_326, as_strided_327, as_strided_328, as_strided_329]
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
            as_strided_nodes = []
            for j in range(i + 1, len(node_list)):
                if node_list[j].target == torch.ops.aten.as_strided.default and node_list[j].args[0] == reduce_scatter_wait_node:
                    as_strided_nodes.append(node_list[j])
                else:
                    break
            reduce_scatter_wait_blocks.append({
                "reduce_scatter_wait_node": reduce_scatter_wait_node,
                "as_strided_nodes": as_strided_nodes,
            })
    for block in reduce_scatter_wait_blocks:
        reduce_scatter_wait_node = block["reduce_scatter_wait_node"]
        as_strided_nodes = block["as_strided_nodes"]
        with mod.graph.inserting_before(return_op):
            new_reduce_scatter_wait_node = _create_new_node_and_replace(mod, reduce_scatter_wait_node, propagate_meta=True)
            for as_strided_node in as_strided_nodes:
                new_as_strided_node = _create_new_node_and_replace(mod, as_strided_node, propagate_meta=True)
    mod.graph.lint()
    mod.recompile()


def decompose_all_gather_copy_in(mod):
    """
    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:86 in foreach_all_gather, code: all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(16384256, 8, 0, torch.bfloat16, device(type='cuda', index=0), [8192000, 256, 8192000], [convert_element_type, convert_element_type_1, convert_element_type_2]);  convert_element_type = convert_element_type_1 = convert_element_type_2 = None
    getitem: "bf16[16384256]" = all_gather_copy_in[0]
    getitem_1: "bf16[131074048]" = all_gather_copy_in[1];  all_gather_copy_in = None

    # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, 8, '0');  getitem = None

    ->

    empty: "bf16[131074048]" = torch.ops.aten.empty.memory_format([131074048], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
    slice_1: "bf16[16384256]" = torch.ops.aten.slice.Tensor(empty, 0, 0, 16384256);  empty = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [8192000, 256, 8192000])
    getitem: "bf16[8192000]" = split_with_sizes[0]
    getitem_1: "bf16[256]" = split_with_sizes[1]
    getitem_2: "bf16[8192000]" = split_with_sizes[2];  split_with_sizes = None
    copy__default = torch.ops.aten.copy_.default(getitem, convert_element_type);  getitem = convert_element_type = None
    copy__default_1 = torch.ops.aten.copy_.default(getitem_1, convert_element_type_1);  getitem_1 = convert_element_type_1 = None
    copy__default_2 = torch.ops.aten.copy_.default(getitem_2, convert_element_type_2);  getitem_2 = convert_element_type_2 = None
    all_gather_into_tensor: "bf16[131074048]" = torch.ops._c10d_functional.all_gather_into_tensor.default(slice_1, 8, '0');  slice_1 = None
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

            # torch.ops.fsdp.all_gather_copy_in(all_gather_input_numel, world_size, rank, dtype, device, inp_split_sizes, param_all_gather_inputs)
            all_gather_input_numel = all_gather_copy_in_node.args[0]
            world_size = all_gather_copy_in_node.args[1]
            rank = all_gather_copy_in_node.args[2]
            dtype = all_gather_copy_in_node.args[3]
            device = all_gather_copy_in_node.args[4]
            inp_split_sizes = all_gather_copy_in_node.args[5]
            param_all_gather_inputs = all_gather_copy_in_node.args[6]

            # Decompose the `all_gather_copy_in` op
            with mod.graph.inserting_before(all_gather_node):
                # Source: all_gather_output = torch.empty((all_gather_input_numel * world_size,), dtype=dtype, device=device)
                empty_node = mod.graph.call_function(torch.ops.aten.empty.memory_format, ([all_gather_input_numel * world_size],), {"dtype": dtype, "device": device, "pin_memory": False})
                # Source: all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
                slice_node = mod.graph.call_function(torch.ops.aten.slice.Tensor, (empty_node, 0, all_gather_input_numel * rank, all_gather_input_numel))
                # Source: foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
                split_with_sizes_node = mod.graph.call_function(torch.ops.aten.split_with_sizes.default, (slice_node, inp_split_sizes))
                getitem_nodes = []
                for i in range(len(inp_split_sizes)):
                    getitem_node = mod.graph.call_function(operator.getitem, (split_with_sizes_node, i))
                    getitem_nodes.append(getitem_node)
                # Source: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
                inplace_copy_nodes = []
                for i in range(len(inp_split_sizes)):
                    inplace_copy_node = mod.graph.call_function(torch.ops.aten.copy_.default, (getitem_nodes[i], param_all_gather_inputs[i]))
                    inplace_copy_nodes.append(inplace_copy_node)
                getitem_0_node.replace_all_uses_with(slice_node)
                mod.graph.erase_node(getitem_1_node)
                mod.graph.erase_node(getitem_0_node)
                mod.graph.erase_node(all_gather_copy_in_node)
    mod.graph.lint()
    mod.recompile()


def decompose_contiguous_view_as_strided(mod):
    """

    ->

    getitem_22: "bf16[8, 256]" = split_with_sizes_5[1]
    view_4: "bf16[2048]" = torch.ops.aten.reshape.default(getitem_22, [2048]);  getitem_22 = None
    as_strided_1: "bf16[2048]" = torch.ops.aten.as_strided.default(view_4, [2048], [1], 0);  view_4 = None
    """
    pass
