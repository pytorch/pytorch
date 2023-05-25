import copy
import dataclasses
import pickle
import warnings
import sympy

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sympy.logic.boolalg import Boolean as SympyBoolean

import torch
from torch.fx.passes.pass_manager import PassManager
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from . import error
from .pass_base import PassType
from .passes.add_runtime_assertions_for_constraints_pass import AddRuntimeAssertionsForConstraintsPass


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


# We specialize this class so that the graph module pickles differently than how
# the FX.GraphModule is pickled in order to save the metadata on each node.
ExportGraphModule = torch.fx.GraphModule


def reduce_graph_module(state_bytes: bytes) -> "ExportGraphModule":
    """
    Function used to deserialize a graph module.
    To serialize the graph, we mapped all of the targets within nodes to their
    string names since we cannot serialize the operations themselves. During
    deserialization, we will then replace the string target names with their
    actual operations.

    Args:
        body: Dictionary of properties for a graph module

    Returns:
        A loaded ExportGraphModule.
    """

    def str_to_op(str_op: str):
        if not isinstance(str_op, str):
            return str_op

        # Some source_fn values are just a string
        if not str_op.startswith("torch.ops."):
            return str_op

        # Get the torch op
        target = None
        for name in str_op.split(".")[2:]:
            target = (
                getattr(torch.ops, name) if target is None else getattr(target, name)
            )
        return target

    body = pickle.loads(state_bytes)

    # Get the target ops since we serialized the targets with just their name
    graph = body["_graph"]
    for node in graph.nodes:

        # Given the name of an operation, find the actual Op object
        # Ex. Given `aten.add.Tensor` we will return `OpOverload(op='aten.add', overload='Tensor')`
        if node.op == "call_function" and isinstance(node.target, str):
            node.target = str_to_op(node.target)

        if (original_aten := node.meta.get("original_aten", None)) is not None:
            node.meta["original_aten"] = str_to_op(original_aten)

        if (source_fn := node.meta.get("source_fn", None)) is not None:
            node.meta["source_fn"] = str_to_op(source_fn)

    # fx.GraphModule constructor expects the totally flattened dictionary for
    # attributes but directly passing the module dict doesn't comply with that
    # format. (some attributes are saved under `_buffers` in module dict etc)
    # So, we work around this by creating a dummy module which contains the
    # original module's attributes
    root = torch.nn.Module()
    root.__dict__ = body

    gm = torch.fx.GraphModule(root, graph, class_name="ExportGraphModule")
    gm.__class__ = type(type(gm).__name__, (ExportGraphModuleMixin, type(gm)), {})

    gm.recompile()
    return gm


class ExportGraphModuleMixin:
    def __reduce__(self) -> Tuple[Callable[..., "ExportGraphModule"], Tuple[bytes]]:
        """
        Serialization of the ExportGraphModule. The FX serialization does not
        serialize the underlying graph to preserve backwards-compatiblity and
        instead retraces the graph module when loading.  This results in loss of
        metadata that is later used for optimizations directly on the graph
        module.  Since we want to preserve this metadata and we do not care that
        much about BC, we will write our own serialization method.
        """

        def op_to_str(op):
            try:
                pickle.dumps(op)
            except TypeError:
                if isinstance(op, torch._ops.HigherOrderOperator):
                    return f"torch.ops.{op.__name__}"
                elif "torch" in op.__module__:
                    return f"torch.ops.{str(op)}"
                else:
                    raise pickle.PickleError(f"Unable to pickle op {op}")
            return op

        gm_dict = self.__dict__.copy()
        gm_dict["_graphmodule_cls_name"] = self.__class__.__name__

        graph = copy.deepcopy(gm_dict["_graph"])
        for node in graph.nodes:
            # Replace the ops with their names since we cannot serialize the ops
            if node.op == "call_function":
                node.target = op_to_str(node.target)

            if (original_aten := node.meta.get("original_aten", None)) is not None:
                node.meta["original_aten"] = op_to_str(original_aten)

            if (source_fn := node.meta.get("source_fn", None)) is not None:
                node.meta["source_fn"] = (source_fn[0], op_to_str(source_fn[1]))

            # Check if other metadata are pickleable
            unpickleable_keys = []
            for key, val in node.meta.items():
                try:
                    pickle.dumps(val)
                except (pickle.PickleError, TypeError):
                    warnings.warn(
                        f"Cannot pickle node {node}'s metadata {key} with value {val}."
                    )
                    unpickleable_keys.append(key)

            for key in unpickleable_keys:
                del node.meta[key]

        gm_dict["_graph"] = graph
        pickled_state = pickle.dumps(gm_dict)

        return (reduce_graph_module, (pickled_state,))


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
        gm = torch.fx.GraphModule(root, graph, class_name="ExportGraphModule")
        gm.__class__ = type(type(gm).__name__, (ExportGraphModuleMixin, type(gm)), {})
        self.graph_module: ExportGraphModule = gm

        self.graph_signature: ExportGraphSignature = graph_signature
        self.call_spec: CallSpec = call_spec
        self.state_dict: Dict[str, Any] = state_dict
        self.symbol_to_range: Dict[str, Tuple[int, int]] = {}
        self._input_shape_constraints: Dict[str, List[Tuple[int, ConstraintExpr, ConstraintExpr]]] = {}
        self._input_name_to_example_inputs: Dict[str, Any] = {}

    def __call__(self, *args: Any) -> Any:
        if self.call_spec.in_spec is not None:
            try:
                args = fx_pytree.tree_flatten_spec(args, self.call_spec.in_spec)  # type: ignore[assignment]
            except Exception as e:
                raise error.InternalError("The in_spec is not correctly maintained.") from e

        with torch.fx.traceback.preserve_node_meta(), torch.no_grad():
            res = torch.fx.Interpreter(self.graph_module).run(*args, enable_io_processing=False)

        if self.call_spec.out_spec is not None:
            try:
                mutation = self.graph_signature.buffers_to_mutate
                num_mutated = len(mutation)
                res = pytree.tree_unflatten(res[num_mutated:], self.call_spec.out_spec)
                return res
            except Exception as e:
                raise error.InternalError("The out_spec is not correctly maintained.") from e
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
        res = AddRuntimeAssertionsForConstraintsPass(
            self._input_shape_constraints,
            self._input_name_to_example_inputs,
            # Only add in inline constraints which are unbacked symints (which
            # start with 'i')
            {k: v for (k, v) in self.symbol_to_range.items() if str(k).startswith('i')},
        )(self.graph_module)
        assert res is not None
        graph_module = res.graph_module
        self.graph_module = graph_module
        return self


def _get_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, arg_index: int
) -> Tuple[str, torch.nn.Module, torch.fx.Node]:
    submod_node = node.args[arg_index]
    assert isinstance(submod_node, torch.fx.Node)
    assert submod_node.op == "get_attr"
    assert isinstance(submod_node.target, str)
    submodule = graph_module.get_submodule(submod_node.target)
    return submod_node.target, submodule, node


def get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, torch.fx.GraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target is torch.ops.cond:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))
            control_flow_submodules.append(_get_submodule(graph_module, node, 2))
        if node.target is torch.ops.map:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))

    return control_flow_submodules  # type: ignore[return-value]
