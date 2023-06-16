import torch
import copy
from typing import Dict
from torch._ops import OpOverload

aten = torch.ops.aten

_NON_FUNCTION_TO_FUNCTIONAL_ASSERTION_FUNCS: Dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten.functional_sym_constrain_range
}


def functionalize(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    gm = copy.deepcopy(gm)
    graph = gm.graph

    inputs_node = next(n for n in graph.nodes if n.op == "placeholder")

    with graph.inserting_after(inputs_node):
        dep_token = graph.call_function(torch.empty, args=(0,))
        dep_token.name = "dep_token"

    none_functional_assertions = [
        n
        for n in graph.nodes
        if n.target in _NON_FUNCTION_TO_FUNCTIONAL_ASSERTION_FUNCS
    ]
    next_dep_token_index = 2
    for n in none_functional_assertions:
        with graph.inserting_after(n):
            args = n.args[1:]
            kwargs = {**n.kwargs, "dep_token": dep_token}
            dep_token = graph.call_function(
                _NON_FUNCTION_TO_FUNCTIONAL_ASSERTION_FUNCS[n.target],
                args=args,
                kwargs=kwargs,
            )
            dep_token.name = f"dep_token_{next_dep_token_index}"
            n.replace_all_uses_with(dep_token)

    for n in none_functional_assertions:
        graph.erase_node(n)

    output_node = next(n for n in graph.nodes if n.op == "output")
    graph.output(output_node.args[0] + (dep_token,))
    graph.erase_node(output_node)

    graph.lint()
    gm.recompile()
    return gm
