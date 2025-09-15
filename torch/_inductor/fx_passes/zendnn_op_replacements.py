import functools
from typing import Any, Callable, Optional

import torch
from torch._dynamo.utils import counters
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
from torch.fx.graph_module import GraphModule


pass_pattern = PatternMatcherPass()
aten = torch.ops.aten


# linear replacement
# aten.linear is present in torch.export path
@register_graph_pattern(
    CallFunction(
        aten.linear,
        Arg(),
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement(match: Match, mat_1: Any, mat_2: Any, bias: Any) -> None:
    def repl(mat_1: Any, mat_2: Any, bias: Any) -> torch.Tensor:
        counters["zendnn"]["zendnn_linear"] += 1
        return aten.zendnn_linear_unary(mat_1, mat_2, bias)

    match.replace_by_example(repl, [mat_1, mat_2, bias])


@register_graph_pattern(
    CallFunction(
        aten.linear,
        Arg(),
        Arg(),
    ),
    pass_dict=pass_pattern,
)
def linear_replacement_no_bias(match: Match, mat_1: Any, mat_2: Any) -> None:
    def repl(mat_1: Any, mat_2: Any) -> torch.Tensor:
        counters["zendnn"]["zendnn_linear"] += 1
        return aten.zendnn_linear_unary(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2])


def is_bias_1d_tensor(match: Match) -> bool:
    # returns true if bias tensor is 1d
    return match.args[0].meta["val"].ndim == 1


def check_alpha_beta_bias(match: Match) -> bool:
    # check bias, beta and alpha has desired values or not
    if match.kwargs["beta"] == 1.0 and match.kwargs["alpha"] == 1.0:
        return is_bias_1d_tensor(match)
    return False


# check for weight and bias as placeholder
def is_placeholder(
    weight_idx: int, bias_idx: Optional[int] = None
) -> Callable[[Match], bool]:
    def fn(match: Match) -> bool:
        # get_attr is a corner case in export path
        if match.args[weight_idx].op not in ("placeholder", "get_attr"):
            return False
        if bias_idx is not None:
            if match.args[bias_idx].op not in ("placeholder", "get_attr"):
                return False
            else:
                return check_alpha_beta_bias(match)
        return True

    return fn


@register_graph_pattern(
    CallFunction(
        aten.mm,
        Arg(),
        CallFunction(aten.permute, Arg(), Arg()),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(1),  # weight_idx = 1, bias = None
)
def mm_linear_replacement_2d(match: Match, mat_1: Any, mat_2: Any, dims: Any) -> None:
    def repl(mat_1: Any, mat_2: Any, dims: Any) -> torch.Tensor:
        counters["zendnn"]["zendnn_linear"] += 1
        return aten.zendnn_linear_unary(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2, dims])


@register_graph_pattern(
    CallFunction(
        aten.view,
        CallFunction(
            aten.mm,
            CallFunction(aten.view, Arg(), Arg()),
            CallFunction(aten.permute, Arg(), Arg()),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2),  # weight_idx = 2, bias = None
)
def mm_linear_replacement_nd(
    match: Match, mat_1: Any, size: Any, mat_2: Any, dims: Any, size_1: Any
) -> None:
    def repl(mat_1: Any, size: Any, mat_2: Any, dims: Any, size_1: Any) -> torch.Tensor:
        # for n-d case, we will calculate the output size from available info
        # and insert a view op before zendnn_linear_unary, if needed
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = aten.view(mat_1, exp_inp_shape)
            counters["zendnn"]["zendnn_linear"] += 1
            return aten.zendnn_linear_unary(view_0, mat_2)
        else:
            counters["zendnn"]["zendnn_linear"] += 1
            return aten.zendnn_linear_unary(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, size, mat_2, dims, size_1])


@register_graph_pattern(
    CallFunction(
        aten.addmm,
        Arg(),
        Arg(),
        CallFunction(aten.permute, Arg(), Arg()),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(2, 0),  # weight_idx = 2, bias_idx = 0
)
def addmm_linear_replacement_2d(
    match: Match,
    bias: Any,
    mat_1: Any,
    mat_2: Any,
    dims: Any,
    *,
    beta: float,
    alpha: float,
) -> None:
    def repl(
        bias: Any, mat_1: Any, mat_2: Any, dims: Any, beta: float, alpha: float
    ) -> torch.Tensor:
        counters["zendnn"]["zendnn_linear"] += 1
        return aten.zendnn_linear_unary(mat_1, mat_2, bias)

    match.replace_by_example(repl, [bias, mat_1, mat_2, dims, beta, alpha])


@register_graph_pattern(
    CallFunction(
        aten.view,
        CallFunction(
            aten.addmm,
            Arg(),
            CallFunction(aten.view, Arg(), Arg()),
            CallFunction(aten.permute, Arg(), Arg()),
            beta=KeywordArg("beta"),
            alpha=KeywordArg("alpha"),
        ),
        Arg(),
    ),
    pass_dict=pass_pattern,
    extra_check=is_placeholder(3, 0),  # weight_idx = 3, bias_idx = 0
)
def addmm_linear_replacement_nd(
    match: Match,
    bias: Any,
    mat_1: Any,
    size: Any,
    mat_2: Any,
    dims: Any,
    size_1: Any,
    *,
    beta: float,
    alpha: float,
) -> None:
    def repl(
        bias: Any,
        mat_1: Any,
        size: Any,
        mat_2: Any,
        dims: Any,
        size_1: Any,
        beta: float,
        alpha: float,
    ) -> torch.Tensor:
        exp_inp_shape = list(size_1)
        exp_inp_shape[-1] = mat_2.shape[-1]
        if mat_1.shape != tuple(exp_inp_shape):
            view_0 = aten.view(mat_1, exp_inp_shape)
            counters["zendnn"]["zendnn_linear"] += 1
            return aten.zendnn_linear_unary(view_0, mat_2, bias)
        else:
            counters["zendnn"]["zendnn_linear"] += 1
            return aten.zendnn_linear_unary(mat_1, mat_2, bias)

    match.replace_by_example(
        repl, [bias, mat_1, size, mat_2, dims, size_1, beta, alpha]
    )


def replace_with_zendnn_ops(gm: GraphModule) -> GraphModule:
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
