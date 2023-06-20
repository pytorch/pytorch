from typing import Dict, List
from dataclasses import dataclass

import torch
from torch._ops import OpOverload
from torch._export.exported_program import _process_constraints
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForConstraintsPass,
)
from torch.fx.passes.pass_manager import PassManager

aten = torch.ops.aten

_NON_FUNCTION_TO_FUNCTIONAL_ASSERTION_FUNCS: Dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten.functional_sym_constrain_range,
    aten._assert_async.default: aten._functional_assert_async.default,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


@dataclass
class FunctionalizedGraphModule:
    graph_module: torch.fx.GraphModule
    assertions_dep_token_index: int


def add_functionalized_runtime_assertions(
    gm: torch.fx.GraphModule,
    parameter_names: List[str],
    buffer_names: List[str],
    example_inputs: List[torch.Tensor],
) -> FunctionalizedGraphModule:
    range_constraints, equality_constraints = _process_constraints(
        gm,
        buffer_names=buffer_names,
        parameter_names=parameter_names,
        example_inputs=example_inputs,
    )
    gm = PassManager(
        [
            _AddRuntimeAssertionsForConstraintsPass(
                range_constraints, equality_constraints
            )
        ]
    )(gm).graph_module
    return functionalize(gm)


def functionalize(gm: torch.fx.GraphModule) -> FunctionalizedGraphModule:
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
            kwargs = {**n.kwargs, "dep_token": dep_token}
            dep_token = graph.call_function(
                _NON_FUNCTION_TO_FUNCTIONAL_ASSERTION_FUNCS[n.target],
                args=n.args,
                kwargs=kwargs,
            )
            dep_token.name = f"dep_token_{next_dep_token_index}"
            next_dep_token_index += 1
            n.replace_all_uses_with(dep_token)

    for n in none_functional_assertions:
        graph.erase_node(n)

    output_node = next(n for n in graph.nodes if n.op == "output")
    graph.output(output_node.args[0] + (dep_token,))
    graph.erase_node(output_node)

    graph.lint()
    gm.recompile()

    return FunctionalizedGraphModule(graph_module=gm, assertions_dep_token_index=1)
