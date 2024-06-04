import torch

def flatten_arg_list(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat_args.extend(flatten_arg_list(arg))
        else:
            flat_args.append(arg)
    return flat_args


def propagate_node_meta(from_node, to_node):
    for k, v in from_node.meta.items():
        to_node.meta[k] = v


def is_node_used_by_ops(node, ops):
    for op in ops:
        if node in flatten_arg_list(op.args) + flatten_arg_list(op.kwargs.values()):
            return True
