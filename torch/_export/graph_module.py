import copy
import dataclasses
import pickle
import warnings
import sympy

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from sympy.logic.boolalg import Boolean as SympyBoolean

import torch
import torch.fx as fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from . import error

ExportGraphModule = fx.GraphModule


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


@dataclasses.dataclass
class ExportMetadata:
    """The fields in this class are what used to be extra data from ExportGraphModule."""

    in_spec: Optional[pytree.TreeSpec] = None
    out_spec: Optional[pytree.TreeSpec] = None
    update_spec: int = 0  # TODO more information here.
    # TODO(gmagogsfm): Expose constraints in Metadata
    # Mapping from output name to mutated buffer names.
    mutation: List[Tuple[str, List[str]]] = dataclasses.field(default_factory=list)
    input_shape_constraints: Dict[str, Any] = dataclasses.field(default_factory=dict)
    inline_constraints: Dict[str, Any] = dataclasses.field(default_factory=dict)
    input_name_to_example_inputs: Dict[str, Any] = dataclasses.field(default_factory=dict)


EXPORT_METADATA = "_export_metadata_key"


def get_export_meta(gm: fx.GraphModule) -> ExportMetadata:
    if EXPORT_METADATA not in gm.meta:
        raise AssertionError(
            "GraphModule does not have EXPORT metadata associated with it."
        )
    return gm.meta[EXPORT_METADATA]


def is_export_graph_module(gm: fx.GraphModule) -> bool:
    return EXPORT_METADATA in gm.meta



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

    exir_meta = body["meta"][EXPORT_METADATA]
    gm = make_export_graph_module(
        root, graph, in_spec=exir_meta.in_spec, out_spec=exir_meta.out_spec, class_name=body["_graphmodule_cls_name"]
    )

    gm.recompile()
    return gm


class ExportGraphModuleMixin:
    def forward(self, *args: Any) -> Any:
        meta = self.meta[EXPORT_METADATA]  # type: ignore[attr-defined]
        if getattr(meta, "in_spec", None) is not None:
            try:
                args = fx_pytree.tree_flatten_spec(args, meta.in_spec)  # type: ignore[assignment]
            except Exception as e:
                raise error.InternalError("The in_spec is not correctly maintained.") from e

        with torch.fx.traceback.preserve_node_meta(), torch.no_grad():
            res = torch.fx.Interpreter(self).run(*args, enable_io_processing=False)

        if getattr(meta, "out_spec", None) is not None:
            try:
                mutation = meta.mutation
                num_mutated = len(mutation) if mutation is not None else 0
                res = pytree.tree_unflatten(
                    res[num_mutated:],
                    meta.out_spec,
                )
                return res
            except Exception as e:
                raise error.InternalError("The out_spec is not correctly maintained.") from e
        return res

    def recompile(self) -> torch.fx.graph_module.PythonCode:
        """
        Generates the code for this GraphModule from its ``graph`` attribute for
        testing (with FileCheck) and debugging purposes.
        """
        python_code = self._graph.python_code(root_module="self")  # type: ignore[attr-defined]
        self._code = python_code.src
        return python_code

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


def attach_export_graph_metadata(gm: fx.GraphModule, meta: ExportMetadata) -> None:
    gm.meta[EXPORT_METADATA] = meta
    gm.__class__ = type(type(gm).__name__, (ExportGraphModuleMixin, type(gm)), {})


def make_export_graph_module(
    root: Union[torch.nn.Module, Dict[str, Any]],
    graph: fx.Graph,
    in_spec: Optional[pytree.TreeSpec] = None,
    out_spec: Optional[pytree.TreeSpec] = None,
    mutation: Optional[List[Tuple[str, List[str]]]] = None,
    example_inputs: Any = None,
    class_name: str = "ExportGraphModule",
) -> fx.GraphModule:

    input_shape_constraints = []
    inline_constraints = {}
    if isinstance(root, torch.fx.GraphModule):
        input_shape_constraints = root.meta.get("input_shape_constraints", [])
        inline_constraints = root.meta.get("inline_constraints", {})

    gm = fx.GraphModule(root, graph, class_name)

    input_tracker = 0

    # group by input id
    input_shape_constraints_by_tensor_id = defaultdict(list)
    for constraint in input_shape_constraints:
        input_shape_constraints_by_tensor_id[constraint["t_id"]].append((constraint["dim"], constraint["min"], constraint["max"]))

    input_shape_constraints_by_src_name: Dict[str, List[Tuple[int, ConstraintExpr, ConstraintExpr]]] = {}
    input_name_to_example_inputs: Dict[str, Any] = {}
    if example_inputs is not None:
        input_tracker = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                example_input = example_inputs[input_tracker]
                if id(example_input) in input_shape_constraints_by_tensor_id:
                    input_shape_constraints_by_src_name[node.name] = input_shape_constraints_by_tensor_id[id(example_input)]
                input_name_to_example_inputs[node.name] = example_input
                input_tracker += 1

    meta = ExportMetadata(
        in_spec=in_spec,
        out_spec=out_spec,
        update_spec=0,
        mutation=mutation if mutation else [],
        input_shape_constraints=input_shape_constraints_by_src_name,
        inline_constraints=inline_constraints,
        input_name_to_example_inputs=input_name_to_example_inputs,
    )
    attach_export_graph_metadata(gm, meta)
    return gm


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
    graph_module: ExportGraphModule,
) -> List[Tuple[str, ExportGraphModule, torch.fx.Node]]:
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
