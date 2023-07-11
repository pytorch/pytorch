from collections import defaultdict
import copy
import dataclasses
import sympy
from typing import Any, Dict, List, Optional, Tuple, Union
from torch._functorch.aot_autograd import FQN, GraphInputName, GraphOutputName

import torch
from torch.fx.passes.infra.pass_manager import PassManager
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._subclasses.fake_tensor import FakeTensor
from . import error
from .pass_base import PassType
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForConstraintsPass,
    InputDim,
    RangeConstraint,
)
from .passes.functionalize_side_effectful_ops_pass import (
    _FunctionalizeSideEffectfulOpsPass,
    _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS,
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


# Information to maintain user calling/returning specs
@dataclasses.dataclass
class CallSpec:
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec


# Extra information for joint graphs
@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    parameters: List[FQN]
    buffers: List[FQN]

    user_inputs: List[GraphInputName]
    user_outputs: List[GraphOutputName]
    inputs_to_parameters: Dict[GraphInputName, FQN]
    inputs_to_buffers: Dict[GraphInputName, FQN]

    buffers_to_mutate: Dict[GraphOutputName, FQN]

    backward_signature: Optional[ExportBackwardSignature]
    # Map from assertion dependency token index to assertion dep token output
    # name in output. The shape of output after aot_autograd will be like:
    # (updated_inputs, user_outputs, dep_token).
    assertion_dep_token: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token) == 1
        assertion_dep_token_index = list(assertion_dep_token.keys())[0]
        assert (
            len(self.user_outputs) + len(self.buffers_to_mutate)
            == assertion_dep_token_index
        )


class ExportedProgram:
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        call_spec: CallSpec,
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]],
        range_constraints: Dict[sympy.Symbol, RangeConstraint],
        equality_constraints: List[Tuple[InputDim, InputDim]],
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self.graph_module = torch.fx.GraphModule(root, graph)

        self.graph_signature: ExportGraphSignature = graph_signature
        self.call_spec: CallSpec = call_spec
        self.state_dict: Dict[str, Any] = state_dict
        self.range_constraints: Dict[sympy.Symbol, RangeConstraint] = range_constraints
        self.equality_constraints: List[Tuple[InputDim, InputDim]] = equality_constraints

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

        param_buffer_values = (value for _, value in self.state_dict.items())

        with torch.no_grad():
            res = torch.fx.Interpreter(self.graph_module).run(
                *param_buffer_values,
                *args,
                enable_io_processing=False
            )

        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            mutated_buffers = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = list(assertion_dep_token.keys())[0]
                res = res[:assertion_dep_token_index]

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
            finally:
                ix = 0
                for _, buffer in self.graph_signature.buffers_to_mutate.items():
                    self.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res

    def __str__(self) -> str:
        graph_module = self.graph_module.print_readable(print_output=False).replace("\n", "\n    ")
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph Signature: {self.graph_signature}\n"
            f"Symbol to range: {self.range_constraints}\n"
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
            copy.deepcopy(self.range_constraints),
            copy.deepcopy(self.equality_constraints),
        )
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _add_runtime_assertions(
        self,
        functionalize: bool,
    ) -> "ExportedProgram":
        ep = self.transform(
            _AddRuntimeAssertionsForConstraintsPass(
                self.range_constraints,
                self.equality_constraints,
            )
        )
        # Graph signature update should be part of pass run instead of a
        # separate step. However this requires augmenting pass infra at fx level
        # to operate on `ExportedProgram` instead of `fx.GraphModule`.
        # TODO: Integrate graph signature update into pass run.
        ep = _fixup_graph_signature(old_ep=self, new_ep=ep)
        if functionalize:
            ep = ep.transform(_FunctionalizeSideEffectfulOpsPass())
            ep = _update_graph_signature_after_assertions_functionalization(ep)

        return ep


def _update_graph_signature_after_assertions_functionalization(
    ep: ExportedProgram,
) -> ExportedProgram:
    output_node = next(
        n for n in ep.graph_module.graph.nodes if n.op == "output"
    )
    dep_token = next(
        (
            {idx: str(n)}
            for idx, n in enumerate(output_node.args[0])
            if n.target
            in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS.values()
        ),
        None,
    )

    return (
        _update_graph_signature(
            ep=ep,
            gs=dataclasses.replace(
                copy.deepcopy(ep.graph_signature), assertion_dep_token=dep_token
            ),
        )
        if dep_token is not None
        else ep
    )

def _fixup_graph_signature(
    old_ep: ExportedProgram, new_ep: ExportedProgram,
) -> ExportedProgram:
    def _get_output_node_names(gm: torch.fx.GraphModule) -> List[FQN]:
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        return [str(arg) for arg in output_node.args[0]]  # type: ignore[misc]

    # Update output names since after adding run time assertions, the names of
    # outputs could change.
    # The assumption here is that the pass:
    # - Won't change graph outputs order semantically so it's possible to create
    #   map from old to new output names based on position.
    # - Will keep input names unchanged so no need to update inputs related
    #   fields (`user_inputs`, `inputs_to_parameters`, `inputs_to_buffers`, ...)
    # If any pass logic breaks the above assumption, it needs to update the
    # signature accordingly to maintain the assumption.
    outputs = _get_output_node_names(old_ep.graph_module)
    new_outputs = _get_output_node_names(new_ep.graph_module)
    assert len(outputs) == len(new_outputs)
    outputs_map = dict(zip(outputs, new_outputs))
    gs = old_ep.graph_signature
    # Need to update graph signature fields related to output since after adding
    # runtime assertions, the output names could change.
    new_user_outputs = [outputs_map[u] for u in gs.user_outputs]  # type: ignore[index]
    new_buffers_to_mutate = {
        outputs_map[u]: b for u, b in gs.buffers_to_mutate.items()  # type: ignore[index]
    }

    return _update_graph_signature(
        ep=new_ep,
        gs=dataclasses.replace(
            copy.deepcopy(new_ep.graph_signature),
            user_outputs=new_user_outputs,
            buffers_to_mutate=new_buffers_to_mutate,
        ),
    )

def _update_graph_signature(
    ep: ExportedProgram, gs: ExportGraphSignature,
) -> ExportedProgram:
    gm = copy.deepcopy(ep.graph_module)
    return ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=gs,
        call_spec=copy.deepcopy(ep.call_spec),
        state_dict=ep.state_dict,
        range_constraints=copy.deepcopy(ep.range_constraints),
        equality_constraints=copy.deepcopy(ep.equality_constraints),
    )


def _process_constraints(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    example_inputs: List[torch.Tensor],
) -> Tuple[Dict[sympy.Symbol, RangeConstraint], List[Tuple[InputDim, InputDim]]]:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule): GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        range_constraints (Dict[sympy.Symbol, RangeConstraints]): Mapping of
            symbols (from SymInts) appearing in the fake tensors in
            node.meta["val"] to their range constraints, which are a tuple
            containing (lower, upper) constraints.

        equality_constraints (List[Tuple[InputDim, InputDim]]): List of tuples
            of (node, dim) to mark that these dimensions are equal.
    """
    input_shape_constraints = graph_module.meta.get("input_shape_constraints", [])
    inline_constraints = graph_module.meta.get("inline_constraints", [])
    num_params_buffer = len(graph_signature.buffers) + len(graph_signature.parameters)

    # Create dict mapping tensor_id to node names
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)
    # Create dict mapping placeholder node names to their nodes
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes):
        if node.op != "placeholder":
            # All placeholder nodes should be together in the beginning of the
            # graph
            break
        if i >= num_params_buffer:
            example_input = example_inputs[i - num_params_buffer]
            tensor_id_to_nodes[id(example_input)].append(node.name)
            placeholder_nodes[node.name] = node

    # Create list of (node name, dim) tuples to mark that they are equal
    equality_constraints: List[Tuple[InputDim, InputDim]] = []
    # Create dict mapping (node name, dim) a list of range (lower, upper)
    # constraints
    multi_range_constraints: Dict[InputDim, List[RangeConstraint]] = defaultdict(list)
    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint["t_id"]]:
            node_dim = InputDim(node, constraint["dim"])

            # Accumulate range constraints
            multi_range_constraints[node_dim].append(
                RangeConstraint(constraint["min"], constraint["max"])
            )

            # Accumulate equality constraints
            if shared := constraint.get("shared", None):
                for other_node in tensor_id_to_nodes[shared["t_id"]]:
                    other_node_dim = InputDim(other_node, shared["dim"])
                    equality_constraints.append((node_dim, other_node_dim))

    # Create dict mapping symbol to a singular range (lower, upper)
    range_constraints: Dict[sympy.Symbol, RangeConstraint] = {}

    # Add inline constraints to range_constraints
    for symbol, value_range in inline_constraints.items():
        range_constraints[symbol] = RangeConstraint(value_range.lower, value_range.upper)

    # Add input range constraints to range_constraintss
    for input_dim, multi_range_constraint in multi_range_constraints.items():  # type: ignore[assignment]
        # Simplify the range constraints into a single range constraint
        # Ex. ranges [2, 10] and [3, 11] would get merged to [3, 10]
        min_vals = [rc.min_val for rc in multi_range_constraint]
        max_vals = [rc.max_val for rc in multi_range_constraint]
        min_val = max(min_vals)
        max_val = min(max_vals)
        assert min_val <= max_val

        # Add input node range constraints
        val = placeholder_nodes[input_dim.input_name].meta["val"]
        assert isinstance(val, FakeTensor)
        symint = val.shape[input_dim.dim]
        assert isinstance(symint, SymInt)
        symbol = symint.node._expr
        range_constraints[symbol] = RangeConstraint(min_val, max_val)

    return range_constraints, equality_constraints
