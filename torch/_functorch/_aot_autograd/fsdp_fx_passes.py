import torch
import operator

def if_tensor_is_resized_to_full_then_resize_it_to_0_at_end_of_graph(mod):
    # FSDP graph has this invariant that if a tensor needs to be resized to full during execution of the graph, it *will* be resized to 0 again before exit of graph.
    tensors_resized = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] > 0:
            tensors_resized.add(n.args[0])
    for tensor in list(tensors_resized):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (tensor, 0), {})
        mod.graph.lint()
        mod.recompile()


def move_resize_to_0_to_end_of_graph(mod):
    # This pass is always a good idea to do so to avoid any use-after-free issues.
    resize_to_0_nodes = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] == 0:
            resize_to_0_nodes.add(n)
    for resize_to_0_node in list(resize_to_0_nodes):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (resize_to_0_node.args[0], 0), {})
        mod.graph.erase_node(resize_to_0_node)
        mod.graph.lint()
        mod.recompile()


def replace_primal_clone_at_beginning_of_graph_with_primal(mod):
    # Replace `clone(primal)` at beginning of graph with `primal`.
    # This is only safe if the graph does not have any autograd-affecting mutations and not explicitly cloning the primal through user code.
    # (i.e. only `with no_grad(): foreach_copy_` and `resize_storage_bytes_` is supported now).
    # TODO add checks to make sure the above invariant is maintained.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if n.op != "placeholder" and n.target is not torch.ops.inductor.resize_storage_bytes_.default:
            if n.target is torch.ops.aten.clone.default:
                if n.args[0] in primal_inputs_tensor_only:
                    n.replace_all_uses_with(n.args[0])
                    mod.graph.erase_node(n)
                    mod.graph.lint()
                    mod.recompile()
            else:
                break


def replace_primal_noop_as_strided_with_primal(mod):
    # Replace `as_strided(primal, ...)` with `primal`, if the as_strided is a no-op based on size and stride info. Should be always safe to do.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if (
            n.target is torch.ops.aten.as_strided.default \
            and n.args[0] in primal_inputs_tensor_only \
            and n.meta['val'].shape == n.args[0].meta['val'].shape \
            and n.meta['val'].stride() == n.args[0].meta['val'].stride()
        ):
            n.replace_all_uses_with(n.args[0])
            mod.graph.erase_node(n)
            mod.graph.lint()
            mod.recompile()


def arg_equals_or_contains_input_node(arg, inp_n):
    if isinstance(arg, (list, tuple)):
        return any(arg_equals_or_contains_input_node(a, inp_n) for a in arg)
    else:
        return arg == inp_n


def input_is_used_in_other_ops(ops, inp_n, except_callback):
    for n in ops:
        if (not except_callback(n)) and any(arg_equals_or_contains_input_node(arg, inp_n) for arg in n.args):
            return True
    return False


def reinplace_foreach_copy_if_input_has_no_other_use_in_graph(mod):
    """
    _foreach_copy_1 = torch.ops.aten._foreach_copy.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    getitem_44: "f32[2, 76137800]" = _foreach_copy_1[0]
    getitem_45: "f32[2, 6170]" = _foreach_copy_1[1]
    getitem_46: "f32[2, 76137800]" = _foreach_copy_1[2]
    getitem_47: "f32[2, 6170]" = _foreach_copy_1[3];  _foreach_copy_1 = None

    ->

    _foreach_copy__1 = torch.ops.aten._foreach_copy_.default([primal_1, primal_2, primal_3, primal_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    """
    # TODO: maybe super slow, need optimization
    for n in list(mod.graph.nodes):
        if n.target is torch.ops.aten._foreach_copy.default:
            _foreach_copy_outplace_node = n
            if all(not input_is_used_in_other_ops(
                list(mod.graph.nodes),
                inp_n,
                except_callback=lambda n: (n == _foreach_copy_outplace_node or (n.target is torch.ops.inductor.resize_storage_bytes_.default))  # ignore this op, and ignore resize_storage_bytes_ ops
            ) for inp_n in _foreach_copy_outplace_node.args[0]):
                with mod.graph.inserting_before(_foreach_copy_outplace_node):
                    for i, arg in enumerate(_foreach_copy_outplace_node.args[0]):
                        copy_to = arg
                        copy_from = _foreach_copy_outplace_node.args[1][i]
                        # _foreach_copy_inplace_node = mod.graph.call_function(torch.ops.aten._foreach_copy_.default, _foreach_copy_outplace_node.args, {})
                        # NOTE: Inductor seems to fail when encountering `_foreach_copy_` op. Need more investigation.
                        mod.graph.call_function(torch.ops.aten.copy_.default, (copy_to, copy_from), {})
                for node in list(mod.graph.nodes):
                    if node.target is operator.getitem and node.args[0] == _foreach_copy_outplace_node:
                        node.replace_all_uses_with(_foreach_copy_outplace_node.args[0][node.args[1]])
                        mod.graph.erase_node(node)
                mod.graph.erase_node(_foreach_copy_outplace_node)
                mod.graph.lint()
                mod.recompile()


def replace_as_strided_scatter_with_primal_if_primal_has_no_other_use_after_this_op(mod):
    """
    as_strided_scatter_3: "f32[12340]" = torch.ops.aten.as_strided_scatter.default(primals_4, view_15, [12340], [1], 0);

    ->

    primals_4
    """
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for i, n in enumerate(list(mod.graph.nodes)):
        if n.target is torch.ops.aten.as_strided_scatter.default:
            as_strided_scatter_node = n
            if as_strided_scatter_node.args[0] in primal_inputs_tensor_only:
                primal = as_strided_scatter_node.args[0]
                if (
                    primal.meta['val'].shape == as_strided_scatter_node.meta['val'].shape \
                    and primal.meta['val'].stride() == as_strided_scatter_node.meta['val'].stride() \
                    and not input_is_used_in_other_ops(list(mod.graph.nodes)[i+1:], primal, except_callback=lambda n: n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] == 0)
                ):
                    as_strided_scatter_node.replace_all_uses_with(primal)
                    mod.graph.erase_node(as_strided_scatter_node)
                    mod.graph.lint()
                    mod.recompile()


def use_input_as_output_for_inplace_copy_ops(mod):
    """
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(getitem_2, as_strided)
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(copy_, add_1)
    ->
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(getitem_2, as_strided)
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(getitem_2, add_1)
    """
    for n in list(mod.graph.nodes):
        if n.target is torch.ops.aten.copy_.default:
            left_inp_n = n.args[0]
            n.replace_all_uses_with(left_inp_n)
            mod.graph.lint()
            mod.recompile()


def flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def if_tensor_is_resized_to_0_immediately_after_inplace_copy_then_delete_the_copy(mod):
    """
    copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(primals_6, getitem_44)
    resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  primals_6 = None
    ->
    resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  primals_6 = None
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.copy_.default:
            inplace_copy_inp = n.args[0]
            for j, node in enumerate(node_list[i+1:]):
                if inplace_copy_inp in flatten_arg_list(node.args):
                    if node.target is torch.ops.inductor.resize_storage_bytes_.default and node.args[1] == 0:
                        mod.graph.erase_node(n)
                        mod.graph.lint()
                        mod.recompile()
                        break
                    else:
                        break
