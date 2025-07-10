# mypy: allow-untyped-defs
from torch._inductor import config

from .zendnn_custom_passes import add_zendnn_weight_prepack_ops
from .zendnn_op_replacements import replace_with_zendnn_ops
from .zendnn_single_post_op_fusions import zendnn_single_post_op_fusions


def optimize(graph):
    # replace aten ops with zendnn ops
    opt_graph = replace_with_zendnn_ops(graph)
    if config.cpp.weight_prepack:
        # replace zendnn ops with zendnn custom passes
        opt_graph = add_zendnn_weight_prepack_ops(opt_graph)
    # single post-op fusion passes
    opt_graph = zendnn_single_post_op_fusions(opt_graph)
    return opt_graph
