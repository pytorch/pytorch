import numpy as np
import time
from collections import OrderedDict

from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from tensorboard.compat.proto.versions_pb2 import VersionDef

import torch
from ._proto_graph import node_proto
from torch.onnx.utils import OperatorExportTypes


methods_OP = ['attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs',
              'kind', 'outputs', 'outputsSize', 'scopeName']
# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ['node', 'offset', 'uniqueName']


class NodeBase(object):
    def __init__(self, uniqueName=None, inputs=None, scope=None, tensor_size=None, op_type='UnSpecified', attributes=''):
        # TODO; Specify a __slots__ for this class or potentially
        # used namedtuple instead
        self.uniqueName = uniqueName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if '__' not in m:
                repr.append(m + ': ' + str(getattr(self, m)) + str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super(NodePy, self).__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []

        for m in valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.uniqueName())
                    if n.type().kind() == 'CompleteTensorType':
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)

            else:
                setattr(self, m, getattr(node_cpp, m)())


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [1, ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'


class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
        self.kind = node_cpp.kind()


class GraphPy(object):
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.uniqueName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)
            for node_output, outputSize in zip(x.outputs, x.outputstensor_size):
                self.scope_name_appeared.append(x.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output,
                                                      x.inputs,
                                                      x.scopeName,
                                                      outputSize,
                                                      op_type=x.kind,
                                                      attributes=x.attributes)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_io.items():
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.uniqueName
            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.uniqueName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.uniqueName] = self.shallowest_scope_name + '/' + node.uniqueName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.uniqueName in self.unique_name_to_scoped_name:
                self.nodes_io[key].uniqueName = self.unique_name_to_scoped_name[node.uniqueName]

    def to_proto(self):
        nodes = []
        node_stats = []
        for v in self.nodes_io.values():
            nodes.append(node_proto(v.uniqueName,
                                    input=v.inputs,
                                    outputsize=v.tensor_size,
                                    op=v.kind,
                                    attributes=v.attributes))

            if v.tensor_size and len(v.tensor_size) > 0:  # assume data is float32, only parameter is counted
                node_stats.append(
                    NodeExecStats(node_name=v.uniqueName,
                                  all_start_micros=int(time.time() * 1e7),
                                  all_end_rel_micros=42,
                                  memory=[AllocatorMemoryUsed(allocator_name="cpu",
                                                              total_bytes=int(np.prod(v.tensor_size)) * 4)]))

        return nodes, node_stats


def parse(graph, args=None, omit_useless_nodes=True):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.

    Args:
      graph (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = len(args)

    scope = {}
    nodes_py = GraphPy()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))  # parameter

    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))

    for node in graph.outputs():  # must place last.
        NodePyIO(node, 'output')
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def graph(model, args, verbose=False, operator_export_type='ONNX', omit_useless_nodes=True):
    """
    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
      operator_export_type (str): One of 'ONNX', 'ONNX_ATEN', or 'RAW'.
        Defaults to 'ONNX' format  because it outputs the most visually
        understandable format.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    operator_export_type = getattr(OperatorExportTypes, operator_export_type)

    # This code is similar to torch/onnx/utils.py, but adjusted to provide
    # the most visually understandable output.
    #
    # For example, the commented out line
    #
    #    # torch._C._jit_pass_onnx_peephole(graph).
    #
    # This pass removes a lot of scope information. The amount of optimization
    # cannot be too much (lots of information lost) or too little (too much
    # useless information), therefore I copy-pasted the code so that it will
    # not be affected by torch/onnx/utils.py changes.
    def _optimize_trace(trace, operator_export_type):
        trace.set_graph(_optimize_graph(trace.graph(), operator_export_type))

    def _optimize_graph(graph, operator_export_type):
        # torch._C._jit_pass_remove_inplace_ops(graph)
        # we record now record some ops like ones/zeros
        # into a trace where we previously recorded constants
        # use constant prop to maintain our current level of onnx support
        # without implementing symbolics for all of them
        torch._C._jit_pass_constant_propagation(graph)
        torch.onnx.utils._split_tensor_list_constants(graph, graph)
        # run dce to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        # torch._C._jit_pass_canonicalize_ops(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)

        # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
        torch._C._jit_pass_prepare_division_for_onnx(graph)
        # onnx only supports tensors, so we turn all out number types into tensors
        torch._C._jit_pass_erase_number_types(graph)
        # onnx does not support tuples, so try to remove them
        torch._C._jit_pass_lower_all_tuples(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)

        if operator_export_type != OperatorExportTypes.RAW:
            graph = torch._C._jit_pass_onnx(graph, operator_export_type)
            torch._C._jit_pass_lint(graph)
            # torch._C._jit_pass_onnx_peephole(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_fixup_onnx_loops(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph

    with torch.onnx.set_training(model, False):
        try:
            trace, _ = torch.jit.get_trace_graph(model, args)
        except RuntimeError:
            print('Error occurs, No graph saved')
            _ = model(*args)  # don't catch, just print the error message
            print("Checking if it's onnx problem...")
            try:
                import tempfile
                torch.onnx.export(
                    model, args, tempfile.TemporaryFile(), verbose=True)
            except RuntimeError:
                print("Your model fails onnx too, please report to onnx team")
            # Create an object matching
            # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/graph.proto
            # The producer version has been reverse engineered from standard
            # TensorBoard logged data.
            return GraphDef(versions=VersionDef(producer=22))

    try:
        # An optimized graph helps debug at a higher level. Users can focus
        # on connections between big modules such as Linear instead of W, x,
        # bias, matmul, etc. Honestly, most users don't care about those
        # detailed nodes information.
        _optimize_trace(trace, operator_export_type)
    except RuntimeError as e:
        # Optimize trace might fail (due to bad scopes in some cases we've seen)
        # and we don't want graph visualization to fail in this case. In this
        # case we'll log the warning and display the non-optimized graph.
        logging.warn(ImportError(e))
    graph = trace.graph()
    if verbose:
        print(graph)
    list_of_nodes, node_stats = parse(graph, args, omit_useless_nodes)
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
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0",
                                                                            node_stats=node_stats)]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
