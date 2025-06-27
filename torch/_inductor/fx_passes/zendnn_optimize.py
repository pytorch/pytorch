# mypy: allow-untyped-defs
from .zendnn_op_replacements import replace_with_zendnn_ops


def optimize(graph):
    # replace aten ops with zendnn ops
    opt_graph = replace_with_zendnn_ops(graph)
    return opt_graph
