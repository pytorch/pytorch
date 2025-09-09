# mypy: allow-untyped-defs
import functools

import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)

from .zendnn_utils import counters


pass_pattern = PatternMatcherPass()
at_ops = torch.ops.aten


# linear replacement
# aten.linear is present in torch.export path
@register_graph_pattern(
    CallFunction(
        at_ops.linear,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement(match: Match, mat_1, mat_2, bias):
    def repl(mat_1, mat_2, bias):
        counters["zendnn"]["zendnn_linear"] += 1
        return at_ops.zendnn_linear(mat_1, mat_2, bias)

    match.replace_by_example(repl, [mat_1, mat_2, bias])


@register_graph_pattern(
    CallFunction(
        at_ops.linear,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement_no_bias(match: Match, mat_1, mat_2):
    def repl(mat_1, mat_2):
        counters["zendnn"]["zendnn_linear"] += 1
        return at_ops.zendnn_linear(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2])


def is_bias_1d_tensor(match):
    # returns true if bias tensor is 1d
    return match.args[0].meta["val"].ndim == 1


def check_alpha_beta_bias(match):
    # check bias, beta and alpha has desired values or not
    if match.kwargs["beta"] == 1.0 and match.kwargs["alpha"] == 1.0:
        return is_bias_1d_tensor(match)
    return False


# check for weight and bias as placeholder
def is_placeholder(weight_idx, bias_idx=None):
    def fn(match):
        # get_attr is a corner case in export path
        if match.args[weight_idx].op not in ("placeholder", "get_attr"):
            return False
        if bias_idx:
            if match.args[bias_idx].op not in ("placeholder", "get_attr"):
                return False
            else:
                return check_alpha_beta_bias(match)
        return True

    return fn


@register_graph_pattern(
    CallFunction(
        at_ops.mm,
        Arg(),
        CallFunction(at_ops.permute, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(1),  # weight_idx = 1, bias = None
)
def mm_linear_replacement_2d(match: Match, mat_1, mat_2, dims):
    def repl(mat_1, mat_2, dims):
        counters["zendnn"]["zendnn_linear"] += 1
        return at_ops.zendnn_linear(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2, dims])


@register_graph_pattern(
    CallFunction(
        at_ops.view,
        CallFunction(
            at_ops.mm,
            CallFunction(at_ops.view, Arg(), Arg()),
            CallFunction(at_ops.permute, Arg(), Arg()),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2),  # weight_idx = 2, bias = None
)
def mm_linear_replacement_nd(match: Match, mat_1, size, mat_2, dims, size_1):
    def repl(mat_1, size, mat_2, dims, size_1):
        # for n-d case, we will calculate the output size from available info
        # and insert a view op before zendnn_linear, if needed
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = at_ops.view(mat_1, exp_inp_shape)
            counters["zendnn"]["zendnn_linear"] += 1
            return at_ops.zendnn_linear(view_0, mat_2)
        else:
            counters["zendnn"]["zendnn_linear"] += 1
            return at_ops.zendnn_linear(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, size, mat_2, dims, size_1])


@register_graph_pattern(
    CallFunction(
        at_ops.addmm,
        Arg(),
        Arg(),
        CallFunction(at_ops.permute, Arg(), Arg()),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2, 0),  # weight_idx = 2, bias_idx = 0
)
def addmm_linear_replacement_2d(match, bias, mat_1, mat_2, dims, *, beta, alpha):
    def repl(bias, mat_1, mat_2, dims, beta, alpha):
        counters["zendnn"]["zendnn_linear"] += 1
        return at_ops.zendnn_linear(mat_1, mat_2, bias)

    match.replace_by_example(repl, [bias, mat_1, mat_2, dims, beta, alpha])


@register_graph_pattern(
    CallFunction(
        at_ops.view,
        CallFunction(
            at_ops.addmm,
            Arg(),
            CallFunction(at_ops.view, Arg(), Arg()),
            CallFunction(at_ops.permute, Arg(), Arg()),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(3, 0),  # weight_idx = 3, bias_idx = 0
)
def addmm_linear_replacement_nd(
    match, bias, mat_1, size, mat_2, dims, size_1, *, beta, alpha
):
    def repl(bias, mat_1, size, mat_2, dims, size_1, beta, alpha):
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = at_ops.view(mat_1, exp_inp_shape)
            counters["zendnn"]["zendnn_linear"] += 1
            return at_ops.zendnn_linear(view_0, mat_2, bias)
        else:
            counters["zendnn"]["zendnn_linear"] += 1
            return at_ops.zendnn_linear(mat_1, mat_2, bias)

    match.replace_by_example(
        repl, [bias, mat_1, size, mat_2, dims, size_1, beta, alpha]
    )


def replace_with_zendnn_ops(gm):
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="replace_with_zendnn_ops",
    )

    if config.pattern_matcher:
        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm
