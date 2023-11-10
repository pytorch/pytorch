import torch
import torch.nn.functional as F


def _replace_dropout(m: torch.fx.GraphModule, for_eval_mode: bool):
    """
    Replace all dropout patterns in the model with the one used in either train
    mode or eval mode, depending on `for_eval_mode`.

    This simulates the behavior of calling `model.train()` and `model.eval()` on a
    model containing `nn.Dropout`, where the eval version of this module becomes
    effectively a noop. For exported models, however, this is not done automatically,
    since dropout is now an aten pattern, so we need to rewrite these dropout
    patterns here manually to produce the same effect.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # Avoid circular dependencies
    from .utils import get_aten_graph_module
    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    def dropout_train(x):
        return F.dropout(x, p=0.5, training=True)

    def dropout_eval(x):
        return F.dropout(x, p=0.5, training=False)

    example_inputs = (torch.randn(1),)

    if for_eval_mode:
        match_pattern = get_aten_graph_module(dropout_train, example_inputs)
        replacement_pattern = get_aten_graph_module(dropout_eval, example_inputs)
    else:
        match_pattern = get_aten_graph_module(dropout_eval, example_inputs)
        replacement_pattern = get_aten_graph_module(dropout_train, example_inputs)

    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],
        ignore_literals=True,
    )
    m.recompile()


# TODO: also support batchnorm
def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout.
    QAT users should call this before performing inference on the model.
    """
    _replace_dropout(model, for_eval_mode=True)
    return model

# TODO: also support batchnorm
def _move_exported_model_to_train(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to train mode.

    This is equivalent to model.train() but only for certain special ops like dropout.
    QAT users should call this before performing training on the model.
    """
    _replace_dropout(model, for_eval_mode=False)
    return model
