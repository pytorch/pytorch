import torch
import torch.nn.functional as F


def _replace_dropout(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch dropout patterns in the model between train and eval modes.

    Dropout has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the dropout behavior between the two modes, so here we need to rewrite the aten
    dropout patterns manually to achieve the same effect.

    See https://github.com/pytorch/pytorch/issues/103681.
    """
    # Avoid circular dependencies
    from .utils import get_aten_graph_module

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    for inplace in [False, True]:

        def dropout_train(x):
            return F.dropout(x, p=0.5, training=True, inplace=inplace)

        def dropout_eval(x):
            return F.dropout(x, p=0.5, training=False, inplace=inplace)

        example_inputs = (torch.randn(1),)
        if train_to_eval:
            match_pattern = get_aten_graph_module(dropout_train, example_inputs)
            replacement_pattern = get_aten_graph_module(dropout_eval, example_inputs)
        else:
            match_pattern = get_aten_graph_module(dropout_eval, example_inputs)
            replacement_pattern = get_aten_graph_module(dropout_train, example_inputs)

        from torch.fx.subgraph_rewriter import replace_pattern_with_filters

        replace_pattern_with_filters(
            m,
            match_pattern,
            replacement_pattern,
            match_filters=[],
            ignore_literals=True,
        )
        m.recompile()


def _replace_batchnorm(m: torch.fx.GraphModule, train_to_eval: bool):
    """
    Switch batchnorm patterns in the model between train and eval modes.

    Batchnorm has different behavior in train vs eval mode. For exported models,
    however, calling `model.train()` or `model.eval()` does not automatically switch
    the batchnorm behavior between the two modes, so here we need to rewrite the aten
    batchnorm patterns manually to achieve the same effect.
    """
    # TODO(Leslie): This function still fails to support custom momentum and eps value.
    # Enable this support in future updates.

    # Avoid circular dependencies
    from .utils import get_aten_graph_module

    # Needed to ensure subgraph matches are self-contained
    m.graph.eliminate_dead_code()
    m.recompile()

    def bn_train(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True
        )

    def bn_eval(
        x: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ):
        return F.batch_norm(
            x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False
        )

    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )
    if train_to_eval:
        match_pattern = get_aten_graph_module(bn_train, example_inputs)
        replacement_pattern = get_aten_graph_module(bn_eval, example_inputs)
    else:
        match_pattern = get_aten_graph_module(bn_eval, example_inputs)
        replacement_pattern = get_aten_graph_module(bn_train, example_inputs)

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],
        ignore_literals=True,
    )
    m.recompile()


def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to eval mode.

    This is equivalent to model.eval() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing inference on the model.
    """
    _replace_dropout(model, train_to_eval=True)
    _replace_batchnorm(model, train_to_eval=True)
    return model


def _move_exported_model_to_train(model: torch.fx.GraphModule):
    """
    Move an exported GraphModule to train mode.

    This is equivalent to model.train() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing training on the model.
    """
    _replace_dropout(model, train_to_eval=False)
    _replace_batchnorm(model, train_to_eval=False)
    return model
