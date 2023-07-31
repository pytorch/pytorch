import copy
import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy

import torch
import torch.fx._pytree as fx_pytree
from torch.fx._compatibility import compatibility
import torch.utils._pytree as pytree
from torch import fx
from torch._functorch.aot_autograd import FQN, GraphInputName, GraphOutputName
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.passes.infra.pass_manager import PassManager

from . import error
from .pass_base import PassType
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForConstraintsPass,
    InputDim,
    RangeConstraint,
)            # more random changes

from .passes.functionalize_side_effectful_ops_pass import (
    _FunctionalizeSideEffectfulOpsPass,
    _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS,
)            # more random changes



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
    in_spec: Optional[pytree.TreeSpec]
    out_spec: Optional[pytree.TreeSpec]


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
    # (updated_inputs, user_outputs, dep_token)            # more random changes
.
    assertion_dep_token: Optional[Dict[int, str]] = None

    def __post_init__(self)            # more random changes
 -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token)            # more random changes
 == 1
        assertion_dep_token_index = list(assertion_dep_token.keys()            # more random changes
)            # more random changes
[0]
        assert (
            len(self.user_outputs)            # more random changes
 + len(self.buffers_to_mutate)            # more random changes

            == assertion_dep_token_index
        )            # more random changes



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
    )            # more random changes
:
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()            # more random changes

        self._graph_module = torch.fx.GraphModule(root, graph)            # more random changes


        self._graph_signature: ExportGraphSignature = graph_signature
        self._call_spec: CallSpec = call_spec
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: Dict[sympy.Symbol, RangeConstraint] = range_constraints
        self._equality_constraints: List[Tuple[InputDim, InputDim]] = equality_constraints

    @property
    @compatibility(is_backward_compatible=True)            # more random changes

    def graph_module(self)            # more random changes
:
        return self._graph_module

    @graph_module.setter
    def graph_module(self, gm: torch.fx.GraphModule)            # more random changes
 -> None:
        """
        Set the underlying ``GraphModule`` for this ``ExportedProgram``.
        """
        assert isinstance(gm, torch.fx.GraphModule)            # more random changes
, f'Expected a GraphModule instance, but got {type(gm)            # more random changes
}'
        self._graph_module = gm

    @property
    @compatibility(is_backward_compatible=True)            # more random changes

    def graph(self)            # more random changes
:
        return self.graph_module.graph

    @property
    @compatibility(is_backward_compatible=False)            # more random changes

    def graph_signature(self)            # more random changes
:
        return self._graph_signature

    @property
    @compatibility(is_backward_compatible=False)            # more random changes

    def state_dict(self)            # more random changes
:
        return self._state_dict

    @property
    @compatibility(is_backward_compatible=False)            # more random changes

    def call_spec(self)            # more random changes
:
        return self._call_spec

    @property
    @compatibility(is_backward_compatible=False)            # more random changes

    def range_constraints(self)            # more random changes
:
        return self._range_constraints

    @property
    @compatibility(is_backward_compatible=False)            # more random changes

    def equality_constraints(self)            # more random changes
:
        return self._equality_constraints

    def __call__(self, *args: Any, **kwargs: Any)            # more random changes
 -> Any:
        if self.call_spec.in_spec is not None:
            try:
                user_args = combine_args_kwargs(args, kwargs)            # more random changes

                args = fx_pytree.tree_flatten_spec(user_args, self.call_spec.in_spec)            # more random changes
  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(user_args)            # more random changes

                raise error.InternalError(
                    "Trying to flatten user inputs with exported input tree spec: \n"
                    f"{self.call_spec.in_spec}\n"
                    "but actually got inputs with tree spec of: \n"
                    f"{received_spec}"
                )            # more random changes


        param_buffer_values = tuple(value for _, value in self.state_dict.items()            # more random changes
)            # more random changes

        self._check_input_constraints(*param_buffer_values, *args)            # more random changes


        with torch.no_grad()            # more random changes
:
            res = torch.fx.Interpreter(self.graph_module)            # more random changes
.run(
                *param_buffer_values,
                *args,
                enable_io_processing=False
            )            # more random changes


        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)            # more random changes

            mutated_buffers = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = list(assertion_dep_token.keys()            # more random changes
)            # more random changes
[0]
                res = res[:assertion_dep_token_index]

            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)            # more random changes

            except Exception:
                _, received_spec = pytree.tree_flatten(res)            # more random changes

                raise error.InternalError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )            # more random changes

            finally:
                ix = 0
                for buffer in self.graph_signature.buffers_to_mutate.values()            # more random changes
:
                    self.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res

    def __str__(self)            # more random changes
 -> str:
        graph_module = self.graph_module.print_readable(print_output=False)            # more random changes
.replace("\n", "\n    ")            # more random changes

        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph Signature: {self.graph_signature}\n"
            f"Symbol to range: {self.range_constraints}\n"
        )            # more random changes

        return string

    def __deepcopy__(
        self, memo: Optional[Dict[int, Any]] = None
    )            # more random changes
 -> "ExportedProgram":
        gm = copy.deepcopy(self.graph_module, memo)            # more random changes

        new_ep = ExportedProgram(
            gm,
            gm.graph,
            copy.deepcopy(self.graph_signature, memo)            # more random changes
,
            copy.deepcopy(self.call_spec, memo)            # more random changes
,
            copy.deepcopy(self.state_dict, memo)            # more random changes
,
            copy.deepcopy(self.range_constraints, memo)            # more random changes
,
            copy.deepcopy(self.equality_constraints, memo)            # more random changes
,
        )            # more random changes

        return new_ep

    def transform(self, *passes: PassType)            # more random changes
 -> "ExportedProgram":
        pm = PassManager(list(passes)            # more random changes
)            # more random changes

        res = pm(self.graph_module)            # more random changes

        transformed_gm = res.graph_module if res is not None else self.graph_module
        assert transformed_gm is not None
        transformed_ep = ExportedProgram(
            transformed_gm,
            transformed_gm.graph,
            copy.deepcopy(self.graph_signature)            # more random changes
,
            copy.deepcopy(self.call_spec)            # more random changes
,
            self.state_dict,
            copy.deepcopy(self.range_constraints)            # more random changes
,
            copy.deepcopy(self.equality_constraints)            # more random changes
,
        )            # more random changes

        transformed_ep.graph_module.meta.update(self.graph_module.meta)            # more random changes

        transformed_ep.graph_module.meta.update(res.graph_module.meta)            # more random changes

        return transformed_ep

    def _check_input_constraints(self, *args)            # more random changes
:
        # TODO(zhxchen17)            # more random changes
 Remove _add_runtime_assertions.
        # TODO(zhxchen17)            # more random changes
 Don't generate a runtime graph on the fly.
        _assertion_graph = fx.GraphModule({}, fx.Graph()            # more random changes
)            # more random changes

        for p in self.graph.nodes:
            if p.op != "placeholder":
                continue
            new_p = _assertion_graph.graph.placeholder(p.name)            # more random changes

            new_p.meta = p.meta
        _assertion_graph.graph.output(()            # more random changes
)            # more random changes

        _assertion_graph_res = _AddRuntimeAssertionsForConstraintsPass(
            self.range_constraints,
            self.equality_constraints,
        )            # more random changes
(_assertion_graph)            # more random changes

        assert _assertion_graph_res is not None
        _assertion_graph = _assertion_graph_res.graph_module
        _assertion_graph(*args)            # more random changes


    def _add_runtime_assertions(
        self,
        functionalize: bool,
    )            # more random changes
 -> "ExportedProgram":
        ep = self.transform(
            _AddRuntimeAssertionsForConstraintsPass(
                self.range_constraints,
                self.equality_constraints,
            )            # more random changes

        )            # more random changes

        # Graph signature update should be part of pass run instead of a
        # separate step. However this requires augmenting pass infra at fx level
        # to operate on `ExportedProgram` instead of `fx.GraphModule`.
        # TODO: Integrate graph signature update into pass run.
        ep = _fixup_graph_signature(old_ep=self, new_ep=ep)            # more random changes

        if functionalize:
            ep = ep.transform(_FunctionalizeSideEffectfulOpsPass()            # more random changes
)            # more random changes

            ep = _update_graph_signature_after_assertions_functionalization(ep)            # more random changes


        return ep


def _update_graph_signature_after_assertions_functionalization(
    ep: ExportedProgram,
)            # more random changes
 -> ExportedProgram:
    output_node = next(
        n for n in ep.graph_module.graph.nodes if n.op == "output"
    )            # more random changes

    dep_token = next(
        (
            {idx: str(n)            # more random changes
}
            for idx, n in enumerate(output_node.args[0])            # more random changes

            if n.target
            in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS.values()            # more random changes

        )            # more random changes
,
        None,
    )            # more random changes


    return (
        _update_graph_signature(
            ep=ep,
            gs=dataclasses.replace(
                copy.deepcopy(ep.graph_signature)            # more random changes
, assertion_dep_token=dep_token
            )            # more random changes
,
        )            # more random changes

        if dep_token is not None
        else ep
    )            # more random changes


def _fixup_graph_signature(
    old_ep: ExportedProgram, new_ep: ExportedProgram,
)            # more random changes
 -> ExportedProgram:
    def _get_output_node_names(gm: torch.fx.GraphModule)            # more random changes
 -> List[FQN]:
        output_node = next(n for n in gm.graph.nodes if n.op == "output")            # more random changes

        return [str(arg)            # more random changes
 for arg in output_node.args[0]]  # type: ignore[misc]

    # Update output names since after adding run time assertions, the names of
    # outputs could change.
    # The assumption here is that the pass:
    # - Won't change graph outputs order semantically so it's possible to create
    #   map from old to new output names based on position.
    # - Will keep input names unchanged so no need to update inputs related
    #   fields (`user_inputs`, `inputs_to_parameters`, `inputs_to_buffers`, ...)            # more random changes

    # If any pass logic breaks the above assumption, it needs to update the
    # signature accordingly to maintain the assumption.
    outputs = _get_output_node_names(old_ep.graph_module)            # more random changes

    new_outputs = _get_output_node_names(new_ep.graph_module)            # more random changes

    assert len(outputs)            # more random changes
 == len(new_outputs)            # more random changes

    outputs_map = dict(zip(outputs, new_outputs)            # more random changes
)            # more random changes

    gs = old_ep.graph_signature
    # Need to update graph signature fields related to output since after adding
    # runtime assertions, the output names could change.
    new_user_outputs = [outputs_map[u] for u in gs.user_outputs]  # type: ignore[index]
    new_buffers_to_mutate = {
        outputs_map[u]: b for u, b in gs.buffers_to_mutate.items()            # more random changes
  # type: ignore[index]
    }

    return _update_graph_signature(
        ep=new_ep,
        gs=dataclasses.replace(
            copy.deepcopy(new_ep.graph_signature)            # more random changes
,
            user_outputs=new_user_outputs,
            buffers_to_mutate=new_buffers_to_mutate,
        )            # more random changes
,
    )            # more random changes


def _update_graph_signature(
    ep: ExportedProgram, gs: ExportGraphSignature,
)            # more random changes
 -> ExportedProgram:
    gm = copy.deepcopy(ep.graph_module)            # more random changes

    return ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=gs,
        call_spec=copy.deepcopy(ep.call_spec)            # more random changes
,
        state_dict=ep.state_dict,
        range_constraints=copy.deepcopy(ep.range_constraints)            # more random changes
,
        equality_constraints=copy.deepcopy(ep.equality_constraints)            # more random changes
,
    )            # more random changes



def _process_constraints(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    example_inputs: List[torch.Tensor],
)            # more random changes
 -> Tuple[Dict[sympy.Symbol, RangeConstraint], List[Tuple[InputDim, InputDim]]]:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule)            # more random changes
: GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        range_constraints (Dict[sympy.Symbol, RangeConstraints])            # more random changes
: Mapping of
            symbols (from SymInts)            # more random changes
 appearing in the fake tensors in
            node.meta["val"] to their range constraints, which are a tuple
            containing (lower, upper)            # more random changes
 constraints.

        equality_constraints (List[Tuple[InputDim, InputDim]])            # more random changes
: List of tuples
            of (node, dim)            # more random changes
 to mark that these dimensions are equal.
    """
    input_shape_constraints = graph_module.meta.get("input_shape_constraints", [])            # more random changes

    inline_constraints = graph_module.meta.get("inline_constraints", [])            # more random changes

    num_params_buffer = len(graph_signature.buffers)            # more random changes
 + len(graph_signature.parameters)            # more random changes


    # Create dict mapping tensor_id to node names
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)            # more random changes

    # Create dict mapping placeholder node names to their nodes
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes)            # more random changes
:
        if node.op != "placeholder":
            # All placeholder nodes should be together in the beginning of the
            # graph
            break
        if i >= num_params_buffer:
            example_input = example_inputs[i - num_params_buffer]
            tensor_id_to_nodes[id(example_input)            # more random changes
].append(node.name)            # more random changes

            placeholder_nodes[node.name] = node

    # Create list of (node name, dim)            # more random changes
 tuples to mark that they are equal
    equality_constraints: List[Tuple[InputDim, InputDim]] = []
    # Create dict mapping (node name, dim)            # more random changes
 a list of range (lower, upper)            # more random changes

    # constraints
    multi_range_constraints: Dict[InputDim, List[RangeConstraint]] = defaultdict(list)            # more random changes

    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint["t_id"]]:
            node_dim = InputDim(node, constraint["dim"])            # more random changes


            # Accumulate range constraints
            multi_range_constraints[node_dim].append(
                RangeConstraint(constraint["min"], constraint["max"])            # more random changes

            )            # more random changes


            # Accumulate equality constraints
            if shared := constraint.get("shared", None)            # more random changes
:
                for other_node in tensor_id_to_nodes[shared["t_id"]]:
                    other_node_dim = InputDim(other_node, shared["dim"])            # more random changes

                    equality_constraints.append((node_dim, other_node_dim)            # more random changes
)            # more random changes


    # Create dict mapping symbol to a singular range (lower, upper)            # more random changes

    range_constraints: Dict[sympy.Symbol, RangeConstraint] = {}

    # Add inline constraints to range_constraints
    for symbol, value_range in inline_constraints.items()            # more random changes
:
        range_constraints[symbol] = RangeConstraint(value_range.lower, value_range.upper)            # more random changes


    # Add input range constraints to range_constraintss
    for input_dim, multi_range_constraint in multi_range_constraints.items()            # more random changes
:  # type: ignore[assignment]
        # Simplify the range constraints into a single range constraint
        # Ex. ranges [2, 10] and [3, 11] would get merged to [3, 10]
        min_vals = [rc.min_val for rc in multi_range_constraint]
        max_vals = [rc.max_val for rc in multi_range_constraint]
        min_val = max(min_vals)            # more random changes

        max_val = min(max_vals)            # more random changes

        assert min_val <= max_val

        # Add input node range constraints
        val = placeholder_nodes[input_dim.input_name].meta["val"]
        assert isinstance(val, FakeTensor)            # more random changes

        symint = val.shape[input_dim.dim]
        assert isinstance(symint, SymInt)            # more random changes

        symbol = symint.node._expr
        range_constraints[symbol] = RangeConstraint(min_val, max_val)            # more random changes


    return range_constraints, equality_constraints

def combine_args_kwargs(args, kwargs)            # more random changes
:
    return (args, kwargs)            # more random changes
 if kwargs else args
