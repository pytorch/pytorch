# mypy: allow-untyped-defs
import types

import torch
import torch.nn.functional as F

from torch.ao.quantization.utils import _assert_and_get_unique_device


__all__ = [
    "model_is_exported",
]


class _WrapperModule(torch.nn.Module):
    """Class to wrap a callable in an :class:`torch.nn.Module`. Use this if you
    are trying to export a callable.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Simple forward that just calls the ``fn`` provided to :meth:`WrapperModule.__init__`."""
        return self.fn(*args, **kwargs)


def model_is_exported(m: torch.nn.Module) -> bool:
    """
    Return True if the `torch.nn.Module` was exported, False otherwise
    (e.g. if the model was FX symbolically traced or not traced at all).
    """
    return isinstance(m, torch.fx.GraphModule) and any(
        "val" in n.meta for n in m.graph.nodes
    )


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
    from .utils import _get_aten_graph_module_for_pattern

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
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train), example_inputs
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval), example_inputs
            )
        else:
            match_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_eval), example_inputs
            )
            replacement_pattern = _get_aten_graph_module_for_pattern(
                _WrapperModule(dropout_train), example_inputs
            )

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
    from .utils import _get_aten_graph_module_for_pattern

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

    device = _assert_and_get_unique_device(m)
    is_cuda = device is not None and device.type == "cuda"
    bn_train_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_train),
        example_inputs,
        is_cuda,
    )
    bn_eval_aten = _get_aten_graph_module_for_pattern(
        _WrapperModule(bn_eval),
        example_inputs,
        is_cuda,
    )

    if train_to_eval:
        match_pattern = bn_train_aten
        replacement_pattern = bn_eval_aten
    else:
        match_pattern = bn_eval_aten
        replacement_pattern = bn_train_aten

    from torch.fx.subgraph_rewriter import replace_pattern_with_filters

    replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern,
        match_filters=[],
        ignore_literals=True,
    )
    m.recompile()


# TODO: expose these under this namespace?
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


def _allow_exported_model_train_eval(model: torch.fx.GraphModule):
    """
    Allow users to call `model.train()` and `model.eval()` on an exported model,
    but with the effect of changing behavior between the two modes limited to special
    ops only, which are currently dropout and batchnorm.

    Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
    does in eager models, but only provides an approximation. In particular, user code
    branching on `training` flag will not function correctly in general because the branch
    is already specialized at export time. Additionally, other ops beyond dropout and batchnorm
    that have different train/eval behavior will also not be converted properly.
    """

    def _train(self, mode: bool = True):
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)

    def _eval(self):
        _move_exported_model_to_eval(self)

    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    return model
