import copy
import dataclasses
import sympy
from collections import defaultdict

from typing import Any, Dict, List, Optional, Tuple, Union

from sympy.logic.boolalg import Boolean as SympyBoolean

import torch
from torch.fx.passes.pass_manager import PassManager
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from . import error
from .pass_base import PassType
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForConstraintsPass,
    ConstraintsContainer,
)


__all__ = ["ExportedProgram"]


LeafValue = Union[
    None,
    bool,
    complex,
    float,
    int,
    str,
    torch.Tensor,
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
]


ConstraintExpr = Union[sympy.Expr, SympyBoolean]


# Information to maintain user calling/returning specs
@dataclasses.dataclass
class CallSpec:
    in_spec: Optional[pytree.TreeSpec] = None
    out_spec: Optional[pytree.TreeSpec] = None


# Extra information for joint graphs
@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    parameters: List[str]
    buffers: List[str]

    user_inputs: List[str]
    user_outputs: List[str]
    inputs_to_parameters: Dict[str, str]
    inputs_to_buffers: Dict[str, str]

    buffers_to_mutate: Dict[str, str]

    backward_signature: Optional[ExportBackwardSignature]


class ExportedProgram:
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        call_spec: CallSpec,
        state_dict: Dict[str, Any],
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self.graph_module = torch.fx.GraphModule(root, graph)

        self.graph_signature: ExportGraphSignature = graph_signature
        self.call_spec: CallSpec = call_spec
        self.state_dict: Dict[str, Any] = state_dict
        self.symbol_to_range: Dict[str, Tuple[int, int]] = {}
        self._input_shape_constraints: Dict[str, ConstraintsContainer] = {}
        self._input_name_to_example_inputs: Dict[str, Any] = {}

    def __call__(self, *args: Any) -> Any:
        if self.call_spec.in_spec is not None:
            try:
                args = fx_pytree.tree_flatten_spec(args, self.call_spec.in_spec)  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(args)
                raise error.InternalError(
                    "Trying to flatten user inputs with exported input tree spec: \n"
                    f"{self.call_spec.in_spec}\n"
                    "but actually got inputs with tree spec of: \n"
                    f"{received_spec}"
                )

        with torch.no_grad():
            res = torch.fx.Interpreter(self.graph_module).run(*args, enable_io_processing=False)

        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
        return res

    def __str__(self) -> str:
        graph_module = self.graph_module.print_readable(print_output=False).replace("\n", "\n    ")
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph Signature: {self.graph_signature}\n"
            f"Symbol to range: {self.symbol_to_range}\n"
        )
        return string

    @property
    def graph(self):
        return self.graph_module.graph

    def transform(self, *passes: PassType) -> "ExportedProgram":
        pm = PassManager(list(passes))
        res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        assert transformed_gm is not None
        transformed_ep = ExportedProgram(
            transformed_gm,
            transformed_gm.graph,
            copy.deepcopy(self.graph_signature),
            copy.deepcopy(self.call_spec),
            self.state_dict,
        )
        return transformed_ep

    def add_runtime_assertions(self) -> "ExportedProgram":
        return self.transform(
            _AddRuntimeAssertionsForConstraintsPass(
                self._input_shape_constraints,
                self._input_name_to_example_inputs,
                # Only add in inline constraints which are unbacked symints (which
                # start with 'i')
                {k: v for (k, v) in self.symbol_to_range.items() if str(k).startswith('i')},
            )
        )


def _set_constraints(
    exported_program: ExportedProgram,
    input_shape_constraints,
    inline_constraints: Dict[str, Tuple[int, int]],
    example_inputs: Any,
):
    # TODO(angelayi): clean this up in a later diff

    exported_program.symbol_to_range = inline_constraints

    tensor_id_to_input_names: Dict[int, List[str]] = defaultdict(list)
    input_name_to_example_inputs: Dict[str, Any] = {}
    if example_inputs is not None:
        input_tracker = 0
        for node in exported_program.graph.nodes:
            if node.op == "placeholder":
                example_input = example_inputs[input_tracker]
                tensor_id_to_input_names[id(example_input)].append(node.name)
                input_name_to_example_inputs[node.name] = example_input
                input_tracker += 1

    input_shape_constraints_by_src_name: Dict[str, ConstraintsContainer] = defaultdict(
        lambda: ConstraintsContainer([], [])
    )
    for constraint in input_shape_constraints:
        for name in tensor_id_to_input_names[constraint["t_id"]]:
            input_shape_constraints_by_src_name[name].ranges.append(
                (constraint["dim"], constraint["min"], constraint["max"])
            )
        if constraint["shared"] is not None:
            for name in tensor_id_to_input_names[constraint["shared"]["t_id"]]:
                for other_name in tensor_id_to_input_names[constraint["t_id"]]:
                    input_shape_constraints_by_src_name[name].equalities.append(
                        (constraint["shared"]["dim"], other_name, constraint["dim"])
                    )
                input_tracker += 1

    exported_program._input_shape_constraints = input_shape_constraints_by_src_name
    exported_program._input_name_to_example_inputs = input_name_to_example_inputs
