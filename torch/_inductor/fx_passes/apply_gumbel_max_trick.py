import torch
from .pre_grad import apply_gumbel_max_trick_pass
from ..pattern_matcher import (
    CallFunction,
    CallMethod,
    Arg,
    KeywordArg,
    CallFunctionVarArgs,
    CallModuleVarArgs,
    Match,
    register_graph_pattern,
)
import operator

@register_graph_pattern(
    CallFunction(
        torch.argmax,
        CallFunction(
            operator.truediv,
            # we don't reply on PatternMatcher to match softmax
            # and exponential_ due to the mutation op
            KeywordArg("softmax"),
            KeywordArg("rand_exp"),
        ),
        dim=-1,
        keepdim=True,
    ),
    pass_dict=apply_gumbel_max_trick_pass,
)
def apply_gumbel_max_trick_old(match: Match, softmax, rand_exp):
    # return
    logits = softmax.args[0]
    if rand_exp.target != "exponential_" or len(rand_exp.users) != 1:
        return

    empty_node, rate = rand_exp.args
    if rate != 1.0:
        return

    if len(empty_node.users) != 1:
        return

    truediv, argmax = match.nodes
    nodes_to_erase = [truediv, softmax]

    graph = match.graph

    with graph.inserting_before(argmax):
        log = graph.call_function(torch.log, (rand_exp,))
        gumbel_noise = graph.call_function(operator.neg, (log,))
        argmax_input = graph.call_function(operator.add, (logits, gumbel_noise))
        argmax.args[0].replace_all_uses_with(argmax_input)

    # erase nodes
    for n in nodes_to_erase:
        match.graph.erase_node(n)
