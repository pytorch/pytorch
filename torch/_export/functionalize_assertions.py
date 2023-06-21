import torch
import copy
from typing import Dict, Tuple
from torch._ops import OpOverload
from torch._functorch.aot_autograd import FQN
from dataclasses import dataclass

aten = torch.ops.aten

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: Dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten.functional_sym_constrain_range,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


@dataclass
class SideEffectOpsFunctionalizationResult:
    # Graph module with assertions functionalized.
    graph_module: torch.fx.GraphModule
    # FQN of assertion dependency token in output.
    dep_token_output: FQN
    # Index of dependency token in output.
    dep_token_output_index: int

def _functionalize_side_effectful_ops(
    gm: torch.fx.GraphModule,
) -> Tuple[torch.fx.GraphModule, FQN]:
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
    # Always appending `dep_token` node to the end of outputs. If `gm` is after
    # AOT export, the outputs will be in format (updated_inputs, user_outputs, dep_token).
    # NOTE: extra change might be needed if `trace_joint` is enabled while calling
    # `aot_export_module` (since `param_gradients` will be added as well). However
    # ignore this case for now since:
    # - It's always disabled in current export logic as
    #   https://github.com/pytorch/pytorch/blob/def1b57151687abd585e3000dd10907b8be01266/torch/_export/__init__.py#L188
    # - Extra callsites like
    #   https://github.com/pytorch/pytorch/blob/def1b57151687abd585e3000dd10907b8be01266/torch/_export/exported_program.py#L119
    #   will need to be changed.
    graph.output(output_node.args[0] + (dep_token,))
    graph.erase_node(output_node)

    graph.lint()
    gm.recompile()

    return SideEffectOpsFunctionalizationResult(
        graph_module=gm,
        dep_token_output=str(dep_token),
        dep_token_output_index=len(output_node.args[0]),
    )
