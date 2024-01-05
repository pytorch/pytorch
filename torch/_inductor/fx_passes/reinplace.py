from collections import defaultdict, namedtuple
import torch
from torch._inductor.lowering import (
    inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings,
)


aten = torch.ops.aten
InplaceableOp = namedtuple("InplaceableOp", ["inplace_op", "mutated_arg"])
InplaceOpHandler = namedtuple("InplaceOpHandler", ["handler", "mutated_arg"])


def replace_target_with(graph, node, new_target):
    node.target = new_target


def graph_call_function(graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
        (args, kwargs),
    )
    with V.fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)

    node = graph.call_function(fn, args, kwargs)
    node.meta["val"] = fake_result
    return node


def reinplace_slice_scatter(graph, node):
    def arg_extractor(self, src, dim=0, start=None, end=None, step=1):
        return self, src, dim, start, end, step

    self_, src, dim, start, end, step = arg_extractor(*node.args, **node.kwargs)
    with graph.inserting_before(node):
        view = graph_call_function(
            graph,
            aten.slice.Tensor,
            self_,
            dim,
            start,
            end,
            step,
        )
        graph_call_function(graph, aten.copy_.default, view, src)
    node.replace_all_uses_with(self_)
    graph.erase_node(node)


def reinplace_select_scatter(graph, node):
    def arg_extractor(self, src, dim, index):
        return self, src, dim, index

    self_, src, dim, index = arg_extractor(*node.args, **node.kwargs)
    with graph.inserting_before(node):
        view = graph_call_function(graph, aten.select.int, self_, dim, index)
        graph_call_function(graph, aten.copy_.default, view, src)
    node.replace_all_uses_with(self_)
    graph.erase_node(node)


def reinplace_as_strided_scatter(graph, node):
    def arg_extractor(self, src, size, stride, storage_offset=None):
        return self, src, size, stride, storage_offset

    self_, src, size, stride, storage_offset = arg_extractor(*node.args, **node.kwargs)
    with graph.inserting_before(node):
        view = graph_call_function(
            graph,
            aten.as_strided.default,
            self_,
            size,
            stride,
            storage_offset,
        )
        graph_call_function(graph, aten.copy_.default, view, src)

    node.replace_all_uses_with(self_)
    graph.erase_node(node)

inplaceable_ops = {
    aten.index_put.default: InplaceOpHandler(
        functools.partial(replace_target_with, new_target=aten.index_put_.default),
        0,
    ),
    aten._unsafe_index_put.default: InplaceOpHandler(
        functools.partial(
            replace_target_with, new_target=inductor_prims._unsafe_index_put_
        ),
        0,
    ),
    aten.slice_scatter.default: InplaceOpHandler(reinplace_slice_scatter, 0),
    aten.select_scatter.default: InplaceOpHandler(reinplace_select_scatter, 0),
    aten.as_strided_scatter.default: InplaceOpHandler(reinplace_as_strided_scatter, 0),
}

try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops = {
        c10d_functional.all_reduce.default: InplaceableOp(
            c10d_functional.all_reduce_.default, 0
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

inplaceable_foreach_ops = {}
for outplace_op, inplace_op in inplaceable_foreach_ops_lowerings.items():
    inplaceable_foreach_ops[outplace_op] = InplaceableOp(inplace_op, 0)


inplaceable_triton_ops = {triton_kernel_wrapper_functional}


def reinplace_inplaceable_ops(graph):
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.
    """

    copy_args_to_copy_nodes = {}
    mutated_inputs = set()
    storage_to_nodes = defaultdict(list)
    node_order: Dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target == aten.copy_.default and node.args[0].op == "placeholder":
            dst = node.args[0]
            src = node.args[1]
            # If the target is a getitem and it indexes a possible clone,
            # then skip over it
            if src.target == operator.getitem and (
                (
                    src.args[0].target == triton_kernel_wrapper_functional
                    and src.args[0].kwargs["kwargs"][src.args[1]] == node.args[0]
                )
                or (src.args[0].target in inplaceable_foreach_ops)
            ):
                src = src.args[0]

            copy_args_to_copy_nodes[(dst, src)] = node

            mutated_inputs.add(node.args[0])

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node):
        node_loc = node_order.get(node)
        if node_loc is None:
            # If node has been newly added to the graph, we don't know its location
            # and thus can't check if it's the last use. Cautiosly assume there may
            # be later users.
            return True
        for view in shared_view_nodes:
            for user in view.users:
                # Skip over the copy_ epilogue node that could get reinplaced
                if copy_node == user:
                    continue
                # Skip all users before node
                user_loc = node_order.get(user)
                if user_loc is not None and user_loc <= node_loc:
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        if isinstance(mutated_arg, (list, tuple)):
            return all(can_inplace(node, arg) for arg in mutated_arg)

        if get_node_storage(mutated_arg) is None:
            return False
        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
        if mutated_arg.op == "placeholder":
            if not (
                copy_node := copy_args_to_copy_nodes.get((mutated_arg, node), False)
            ):
                return False

            if any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=copy_node
            ):
                return False

            return True
        elif any(view.op == "placeholder" for view in shared_view_nodes):
            # If mutated arg is view of any of the inputs of the graph,
            # do not allow for inplacing.
            # This would require more sophisticated algorithm to handle
            return False
        else:
            return not any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=None
            )

    replace_list: List[Tuple[Any, Any]] = []
    for node in graph.nodes:
        if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            if can_inplace(node, mutated_arg):
                # TODO(yifu): this doesn't properly remove copy epilogues for
                # ops that mutate multiple inputs. Need to revise the copy
                # node tracking logic to support the case.
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    graph.erase_node(copy_node)
                inplaceable_op.handler(graph, node)
        elif node.target in inplaceable_triton_ops:
            # inplaceable_triton_ops take an additional argument called
            # tensors_to_clone which contain a list of tensors to clone
            # This pass iterates over them and sees which ones are safe
            # to eliminate (i.e. no longer need the clones)
            tensors_to_clone = []
            for arg in node.kwargs["tensors_to_clone"]:
                assert arg in node.kwargs["kwargs"]
                mutated_arg = node.kwargs["kwargs"][arg]
                if can_inplace(node, mutated_arg):
                    copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                    if copy_node is not None:
                        graph.erase_node(copy_node)
                    for user in node.users:
                        if user.target == operator.getitem and user.args[1] == arg:
                            replace_list.append((user, mutated_arg))
                else:
                    tensors_to_clone.append(arg)
            kwargs = dict(node.kwargs)
            kwargs["tensors_to_clone"] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)
        elif (
            inplaceable_op := inplaceable_foreach_ops.get(node.target, None)
        ) is not None:
            mutated_args = node.args[inplaceable_op.mutated_arg]

            if not all((arg, node) in copy_args_to_copy_nodes for arg in mutated_args):
                continue

            if can_inplace(node, mutated_args):
                for arg in mutated_args:
                    copy_node = copy_args_to_copy_nodes[(arg, node)]
                    graph.erase_node(copy_node)

                node.target = inplaceable_op.inplace_op
    for node, replacement in replace_list:
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
