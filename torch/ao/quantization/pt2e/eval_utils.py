import torch
import torch.nn.functional as F


def _replace_dropout_for_eval(m: torch.fx.GraphModule):
    """
    Replace the aten training dropout pattern with a noop, intended for eval.

    For models with dropout torch ops (nn.Dropout, F.dropout), calling model.eval()
    effectively turns these dropout ops into noops. For exported models, however,
    this is not done automatically, since the aten dropout patterns previously generated
    for training remain in the graph. Here we rewrite these dropout patterns with noops
    to avoid incorrectly applying further dropout during eval.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # Avoid circular dependencies
    from .utils import get_aten_graph_module

    def dropout_train(x):
        return F.dropout(x, p=0.5, training=True)

    def dropout_eval(x):
        return F.dropout(x, p=0.5, training=False)

    example_inputs = (torch.randn(1),)
    match_pattern = get_aten_graph_module(dropout_train, example_inputs)
    replacement_pattern = get_aten_graph_module(dropout_eval, example_inputs)

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],
        ignore_literals=True,
    )
    m.recompile()


# TODO: also support move_model_to_train
# TODO: also support standalone batchnorm
def _move_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout.
    QAT users should call this before performing inference on the model.
    """
    _replace_dropout_for_eval(model)
    return model
