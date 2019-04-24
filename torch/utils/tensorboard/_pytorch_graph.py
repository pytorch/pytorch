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
methods_IO = ['node', 'offset', 'uniqueName']  # 'unique' <int> , 'type' <Tensor<class 'torch._C.Type'>>


class NodeBase(object):
    def __init__(self, uniqueName=None, inputs=None, scope=None, tensor_size=None, op_type='UnSpecified', attributes=''):
        self.uniqueName = uniqueName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        if scope is not None:
            self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if '__' not in m:
                repr.append(m + ': ' + str(getattr(self, m)) + str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'


class NodePy(NodeBase):
    def __init__(self, NodeCpp, valid_methods):
        super(NodePy, self).__init__(NodeCpp)
        self.valid_methods = valid_methods[:]
        self.inputs = []

        for m in self.valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(NodeCpp, m)())
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
                setattr(self, m, getattr(NodeCpp, m)())


class NodePyIO(NodePy):
    def __init__(self, NodeCpp, input_or_output=None):
        super(NodePyIO, self).__init__(NodeCpp, methods_IO)
        try:
            tensor_size = NodeCpp.type().sizes()
        except RuntimeError:
            tensor_size = [1, ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'


class NodePyOP(NodePy):
    def __init__(self, NodeCpp):
        super(NodePyOP, self).__init__(NodeCpp, methods_OP)
        self.attributes = str({k: NodeCpp[k] for k in NodeCpp.attributeNames()}).replace("'", ' ')
        self.kind = NodeCpp.kind()


class GraphPy(object):
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.uniqueNameToScopedName = {}
        self.shallowestScopeName = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if type(x) == NodePyIO:
            self.nodes_io[x.uniqueName] = x
        if type(x) == NodePyOP:
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
                self.shallowestScopeName = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.uniqueNameToScopedName[input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_io.items():
            if type(node) == NodeBase:
                self.uniqueNameToScopedName[key] = node.scope + '/' + node.uniqueName
            if hasattr(node, 'input_or_output'):
                self.uniqueNameToScopedName[key] = node.input_or_output + '/' + node.uniqueName
            if hasattr(node, 'scope'):
                if node.scope == '' and self.shallowestScopeName:
                    self.uniqueNameToScopedName[node.uniqueName] = self.shallowestScopeName + '/' + node.uniqueName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.uniqueNameToScopedName[node_input_id] for node_input_id in node.inputs]
            if node.uniqueName in self.uniqueNameToScopedName:
                self.nodes_io[key].uniqueName = self.uniqueNameToScopedName[node.uniqueName]

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


# one argument: 'hasAttribute', 'hasAttributes',
def parse(graph, args=None, omit_useless_nodes=True):
    """This method parses a PyTorch graph.
    Args:
      graph (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean) whether to remove nodes from the graph.
    """
    n_inputs = len(args)  # not sure...

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
    operator_export_type = getattr(OperatorExportTypes, operator_export_type)

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
            return GraphDef(versions=VersionDef(producer=22))

    try:
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
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0",
                                                                            node_stats=node_stats)]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
