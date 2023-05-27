from collections import defaultdict
import copy
import dataclasses
import math
import sympy
from typing import Any, Dict, List, Optional, Tuple, Union


import torch
from torch.fx.passes.pass_manager import PassManager
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._subclasses.fake_tensor import FakeTensor
from . import error
from .pass_base import PassType
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForConstraintsPass,
    NodeDim,
    RangeConstraint,
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
        symbol_to_range: Dict[sympy.Symbol, RangeConstraint],
        equality_constraints: Dict[NodeDim, List[NodeDim]],
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self.graph_module = torch.fx.GraphModule(root, graph)

        self.graph_signature: ExportGraphSignature = graph_signature
        self.call_spec: CallSpec = call_spec
        self.state_dict: Dict[str, Any] = state_dict
        self.symbol_to_range: Dict[sympy.Symbol, RangeConstraint] = symbol_to_range
        self.equality_constraints: Dict[NodeDim, List[NodeDim]] = equality_constraints

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
            copy.deepcopy(self.symbol_to_range),
            copy.deepcopy(self.equality_constraints),
        )
        return transformed_ep

    def add_runtime_assertions(self) -> "ExportedProgram":
        return self.transform(
            _AddRuntimeAssertionsForConstraintsPass(self.symbol_to_range, self.equality_constraints)
        )


def _process_constraints(
    graph_module: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Tuple[Dict[sympy.Symbol, RangeConstraint], Dict[NodeDim, List[NodeDim]]]:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule): GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        symbol_to_constraints (Dict[sympy.Symbol, List[Constraints]]): Mapping of
            symbols (SymInt/SymFloat/SymBool) appearing in fake tensors to their
            constraints, which are either their range (lower, upper) constraint,
            or an equality constraint with another symbol.

        symbol_to_fx_source (Dict[sympy.Symbol, Tuple[torch.fx.Node, int]]): Mapping
            of symbols to the FX node in which they appear in, and their
            dimension.
    """
    input_shape_constraints = graph_module.meta.get("input_shape_constraints", [])
    inline_constraints = graph_module.meta.get("inline_constraints", [])

    # Create dict mapping tensor_id to node names
    # And another dict mapping placeholder node names to their nodes
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes):
        if node.op != "placeholder":
            # All placeholder nodes should be together in the beginning of the
            # graph
            break
        example_input = example_inputs[i]
        tensor_id_to_nodes[id(example_input)].append(node.name)
        placeholder_nodes[node.name] = node

    # Create dict mapping (node name, dim) to a list of other (node name, dim)
    # to mark that they are equal
    equality_constraints: Dict[NodeDim, List[NodeDim]] = defaultdict(list)
    # Create dict mapping (node name, dim) a list of range (lower, upper)
    # constraints
    range_constraints: Dict[NodeDim, List[RangeConstraint]] = defaultdict(list)
    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint["t_id"]]:
            node_dim = NodeDim(node, constraint["dim"])

            # Accumulate range constraints
            range_constraints[node_dim].append(
                RangeConstraint(constraint["min"], constraint["max"])
            )

            # Accumulate equality constraints
            if shared := constraint.get("shared", None):
                for other_node in tensor_id_to_nodes[shared["t_id"]]:
                    other_node_dim = NodeDim(other_node, shared["dim"])
                    equality_constraints[node_dim].append(other_node_dim)

    # Create dict mapping symbol to a range (lower, upper)
    symbol_to_range: Dict[sympy.Symbol, RangeConstraint] = {}

    # Convert simple sympy Integers into concrete int to
    # insert into graph
    def _convert_to_int(val):
        if val == sympy.oo:
            return math.inf
        if val == -sympy.oo:
            return -math.inf
        if isinstance(val, sympy.Integer):
            return int(val)
        raise RuntimeError(
            "Export constraints cannot be non-integer expressions"
        )

    # Add inline constraints to symbol_to_ranges
    for symbol, (min_val, max_val) in inline_constraints.items():
        symbol_to_range[symbol] = RangeConstraint(
            _convert_to_int(min_val), _convert_to_int(max_val)
        )

    # Add input range constraints to symbol_ro_ranges
    for (node_name, dim), range_constraints in range_constraints.items():
        # Simplify the range constraints into a single range constraint
        # Ex. ranges [2, 10] and [3, 11] would get merged to [3, 10]
        min_vals, max_vals = zip(*range_constraints)
        min_val = max(min_vals)
        max_val = min(max_vals)
        assert min_val <= max_val

        # Add input node range constraints
        val = placeholder_nodes[node_name].meta["val"]
        assert isinstance(val, FakeTensor)
        symint = val.shape[dim]
        assert isinstance(symint, SymInt)
        symbol = symint.node._expr
        symbol_to_range[symbol] = RangeConstraint(
            _convert_to_int(min_val), _convert_to_int(max_val)
        )

    return symbol_to_range, equality_constraints
