# mypy: allow-untyped-defs
from collections import OrderedDict
import contextlib
from typing import Dict, Any

from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef

import torch
from ._proto_graph import node_proto

methods_OP = [
    "attributeNames",
    "hasMultipleOutputs",
    "hasUses",
    "inputs",
    "kind",
    "outputs",
    "outputsSize",
    "scopeName",
]
# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ["node", "offset", "debugName"]

GETATTR_KIND = "prim::GetAttr"
CLASSTYPE_KIND = "ClassType"


class NodeBase:
    def __init__(
        self,
        debugName=None,
        inputs=None,
        scope=None,
        tensor_size=None,
        op_type="UnSpecified",
        attributes="",
    ):
        # TODO; Specify a __slots__ for this class or potentially
        # used namedtuple instead
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if "__" not in m:
                repr.append(
                    m + ": " + str(getattr(self, m)) + str(type(getattr(self, m)))
                )
        return "\n".join(repr) + "\n\n"


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super().__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []

        for m in valid_methods:
            if m == "inputs" or m == "outputs":
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                setattr(self, m, io_unique_names)
                setattr(self, m + "tensor_size", io_tensor_sizes)

            else:
                setattr(self, m, getattr(node_cpp, m)())


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super().__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [
                1,
            ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        # Kind attribute string is purely descriptive and will be shown
        # in detailed information for the node in TensorBoard's graph plugin.
        #
        # NodePyOP nodes get this from their kind() method.
        self.kind = "Parameter"
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = "IO Node"


class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super().__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        self.attributes = str(
            {k: _node_get(node_cpp, k) for k in node_cpp.attributeNames()}
        ).replace("'", " ")
        self.kind = node_cpp.kind()


class GraphPy:
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """

    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = "default"
        self.scope_name_appeared = []

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)

    def printall(self):
        print("all nodes")
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split("/")[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for node_output, outputSize in zip(node.outputs, node.outputstensor_size):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(
                    node_output,
                    node.inputs,
                    node.scopeName,
                    outputSize,
                    op_type=node.kind,
                    attributes=node.attributes,
                )

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = (
                    node.scopeName + "/" + input_node_id
                )

        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
            if hasattr(node, "input_or_output"):
                self.unique_name_to_scoped_name[key] = (
                    node.input_or_output + "/" + node.debugName
                )

            if hasattr(node, "scope") and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
                if node.scope == "" and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = (
                        self.shallowest_scope_name + "/" + node.debugName
                    )

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
            ]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[
                    node.debugName
                ]

    def to_proto(self):
        """Convert graph representation of GraphPy object to TensorBoard required format."""
        # TODO: compute correct memory usage and CPU time once
        # PyTorch supports it
        nodes = []
        for v in self.nodes_io.values():
            nodes.append(
                node_proto(
                    v.debugName,
                    input=v.inputs,
                    outputsize=v.tensor_size,
                    op=v.kind,
                    attributes=v.attributes,
                )
            )
        return nodes


def parse(graph, trace, args=None, omit_useless_nodes=True):
    """Parse an optimized PyTorch model graph and produces a list of nodes and node stats.

    Useful for eventual conversion to TensorBoard protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    nodes_py = GraphPy()
    for node in graph.inputs():
        if omit_useless_nodes:
            if (
                len(node.uses()) == 0
            ):  # number of user of the node (= number of outputs/ fanout)
                continue

        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, "input"))

    attr_to_scope: Dict[Any, str] = {}
    for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            attr_name = node.s("name")
            attr_key = node.output().debugName()
            parent = node.input().node()
            if (
                parent.kind() == GETATTR_KIND
            ):  # If the parent node is not the top-level "self" node
                parent_attr_key = parent.output().debugName()
                parent_scope = attr_to_scope[parent_attr_key]
                attr_scope = parent_scope.split("/")[-1]
                attr_to_scope[attr_key] = f"{parent_scope}/{attr_scope}.{attr_name}"
            else:
                attr_to_scope[attr_key] = f"__module.{attr_name}"
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[attr_key]  # type: ignore[attr-defined]
                nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))

    for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
        node_pyio = NodePyIO(node, "output")
        node_pyio.debugName = f"output.{i + 1}"
        node_pyio.inputs = [node.debugName()]
        nodes_py.append(node_pyio)

    def parse_traced_name(module):
        if isinstance(module, torch.jit.TracedModule):
            module_name = module._name
        else:
            module_name = getattr(module, "original_name", "Module")
        return module_name

    alias_to_name = {}
    base_name = parse_traced_name(trace)
    for name, module in trace.named_modules(prefix="__module"):
        mod_name = parse_traced_name(module)
        attr_name = name.split(".")[-1]
        alias_to_name[name] = f"{mod_name}[{attr_name}]"

    for node in nodes_py.nodes_op:
        module_aliases = node.scopeName.split("/")
        replacements = [
            alias_to_name[alias] if alias in alias_to_name else alias.split(".")[-1]
            for alias in module_aliases
        ]
        node.scopeName = base_name
        if any(replacements):
            node.scopeName += "/" + "/".join(replacements)

    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def graph(model, args, verbose=False, use_strict_trace=True):
    """
    Process a PyTorch model and produces a `GraphDef` proto that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
      use_strict_trace (bool): Whether to pass keyword argument `strict` to
        `torch.jit.trace`. Pass False when you want the tracer to
        record your mutable container types (list, dict)
    """
    with _set_model_to_eval(model):
        try:
            trace = torch.jit.trace(model, args, strict=use_strict_trace)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print("Error occurs, No graph saved")
            raise e

    if verbose:
        print(graph)
    list_of_nodes = parse(graph, trace, args)
    # We are hardcoding that this was run on CPU even though it might have actually
    # run on GPU. Note this is what is shown in TensorBoard and has no bearing
    # on actual execution.
    # TODO: See if we can extract GPU vs CPU information from the PyTorch model
    # and pass it correctly to TensorBoard.
    #
    # Definition of StepStats and DeviceStepStats can be found at
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/graph/tf_graph_common/test/graph-test.ts
    # and
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/step_stats.proto
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
    # The producer version has been reverse engineered from standard
    # TensorBoard logged data.


@contextlib.contextmanager
def _set_model_to_eval(model):
    """Context manager to temporarily set the training mode of ``model`` to eval."""
    if not isinstance(model, torch.jit.ScriptFunction):
        originally_training = model.training
        model.train(False)
        try:
            yield
        finally:
            model.train(originally_training)
    else:
        # Do nothing for ScriptFunction
        try:
            yield
        finally:
            pass


def _node_get(node: torch._C.Node, key: str):
    """Get attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)
