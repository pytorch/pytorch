import torch
from . import config

AOT_PARTITIONER_DEBUG = config.debug_partitioner


def _is_tangent(node):
    return node.op == "placeholder" and "tangents" in node.target


def collect_mul_ops(node, target_name="aten::mul.Tensor", seen=None):
    """
    Collect operations that match 'target_name',

    :yields: lists of nodes sorted topologically
    """
    if seen is None:
        seen = set()

    if node in seen:
        return

    seen.add(node)

    def is_same_target(node):
        if isinstance(node.target, torch._ops.PyOperatorABC):
            return node.target.name() == target_name
        return False

    input_nodes = list(node.users)
    if is_same_target(node):
        group = [node]
        new_input_nodes = []
        while input_nodes:
            input_node = input_nodes.pop(0)
            # If this node has multiple users we can not re-order
            if is_same_target(input_node) and len(input_node.users) == 1:
                input_nodes.extend(input_node.users)
                group.append(input_node)
                seen.add(input_node)
            else:
                new_input_nodes.append(input_node)

        input_nodes = new_input_nodes
        if len(group) > 2:
            yield group

    for input_node in input_nodes:
        yield from collect_mul_ops(input_node, seen=seen)


def reorder_tangents(graph):
    """
    reorder operations on tangent inputs, such that
    if the backwards pass yields

        ((tangents_1 * primals_4) * primals_3) * primals_2

    then the graph will be updated to look like:

       tangents_1 * (primals_4 * (primals_3 * primals_2))

    """
    tangents = {node for node in graph.nodes if _is_tangent(node)}

    groups = []
    for tangent in tangents:
        groups.extend(collect_mul_ops(tangent, seen=set()))

    if AOT_PARTITIONER_DEBUG:
        print(f"Found {len(groups)} groups of pointwise multiplications to reorder")
    for group in groups:
        if AOT_PARTITIONER_DEBUG:
            print(' + will reorder group: ', group)
        inputs = [group[0]._args[0]] + [node._args[1] for node in group]
        end = group[-1]
        while inputs:
            right = inputs.pop(-1)
            left = inputs.pop(-1)
            with graph.inserting_before(end):
                node = graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(left, right)
                )

                # TODO: compute correct meta
                if isinstance(right, torch.fx.node.Node):
                    node.meta = right.meta
                elif isinstance(left, torch.fx.node.Node):
                    node.meta = left.meta

                if len(inputs):
                    inputs.append(node)
        end.replace_all_uses_with(node)
