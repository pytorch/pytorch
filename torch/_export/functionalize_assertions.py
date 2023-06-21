import torch
import copy
from typing import Dict
from torch._ops import OpOverload

aten = torch.ops.aten

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: Dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten.functional_sym_constrain_range,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


def _functionalize_side_effectful_ops(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```

    Will be transformed to:
    ```
    def f(x):
        dep_token = make_dep_token()
        dep_token_2 = functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token
        )

        return x.add(3), dep_token_2
    ```
    """

    gm = copy.deepcopy(gm)
    graph = gm.graph

    inputs_node = next(n for n in graph.nodes if n.op == "placeholder")

    with graph.inserting_after(inputs_node):
        dep_token = graph.call_function(aten.make_dep_token)
        dep_token.name = "dep_token"

    non_functional_assertions = [
        n
        for n in graph.nodes
        if n.target in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS
    ]
    next_dep_token_index = 2
    for n in non_functional_assertions:
        with graph.inserting_after(n):
            kwargs = {**n.kwargs, "dep_token": dep_token}
            dep_token = graph.call_function(
                _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS[n.target],
                args=n.args,
                kwargs=kwargs,
            )
            dep_token.name = f"dep_token_{next_dep_token_index}"
            next_dep_token_index += 1
            n.replace_all_uses_with(dep_token)
            graph.erase_node(n)

    output_node = next(n for n in graph.nodes if n.op == "output")
    graph.output(output_node.args[0] + (dep_token,))
    graph.erase_node(output_node)

    graph.lint()
    gm.recompile()
    return gm
