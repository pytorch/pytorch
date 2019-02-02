import time
import warnings
import itertools
from distutils.version import LooseVersion
from collections import OrderedDict
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.versions_pb2 import VersionDef
from .proto_graph import Node_proto

methods_OP = ['attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs',
              'kind', 'outputs', 'outputsSize', 'scopeName']
methods_IO = ['node', 'offset', 'uniqueName']  # 'unique' <int> , 'type' <Tensor<class 'torch._C.Type'>>


class Node_base(object):
    def __init__(self, uniqueName=None, inputs=None, scope=None, tensorSize=None, op_type='UnSpecified', attributes=''):
        self.uniqueName = uniqueName
        self.inputs = inputs
        self.tensorSize = tensorSize
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


class Node_py(Node_base):
    def __init__(self, Node_cpp, valid_mothods):
        super(Node_py, self).__init__(Node_cpp)
        self.valid_mothods = valid_mothods[:]
        self.inputs = []

        for m in self.valid_mothods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(Node_cpp, m)())
                io_uniqueName_list = []
                io_tensorSize_list = []
                for n in list_of_node:
                    io_uniqueName_list.append(n.uniqueName())
                    if n.type().kind() in ['DynamicType', 'ListType']:  # segfault
                        io_tensorSize_list.append(None)
                    else:
                        io_tensorSize_list.append(n.type().sizes())

                setattr(self, m, io_uniqueName_list)
                setattr(self, m + 'TensorSize', io_tensorSize_list)

            else:
                setattr(self, m, getattr(Node_cpp, m)())


class Node_py_IO(Node_py):
    def __init__(self, Node_cpp, input_or_output=None):
        super(Node_py_IO, self).__init__(Node_cpp, methods_IO)
        try:
            tensorsize = Node_cpp.type().sizes()
        except RuntimeError:
            tensorsize = [1, ]  # fail when constant model is used.
        self.tensorSize = tensorsize
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'


class Node_py_OP(Node_py):
    def __init__(self, Node_cpp):
        super(Node_py_OP, self).__init__(Node_cpp, methods_OP)
        self.attributes = str({k: Node_cpp[k] for k in Node_cpp.attributeNames()}).replace("'", ' ')
        self.kind = Node_cpp.kind()


class Graph_py(object):
    def __init__(self):
        self.nodes_OP = []
        self.nodes_IO = OrderedDict()
        self.uniqueNameToScopedName = {}
        self.shallowestScopeName = 'default'
        self.scope_name_appeared = []

    def append(self, x):
        if type(x) == Node_py_IO:
            self.nodes_IO[x.uniqueName] = x
        if type(x) == Node_py_OP:
            self.nodes_OP.append(x)
            for node_output, outputSize in zip(x.outputs, x.outputsTensorSize):
                self.scope_name_appeared.append(x.scopeName)
                self.nodes_IO[node_output] = Node_base(node_output,
                                                       x.inputs,
                                                       x.scopeName,
                                                       outputSize,
                                                       op_type=x.kind,
                                                       attributes=x.attributes)

    def printall(self):
        print('all nodes')
        for node in self.nodes_OP:
            print(node)
        for key in self.nodes_IO:
            print(self.nodes_IO[key])

    def findCommonRoot(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowestScopeName = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_OP:
            for input_node_id in node.inputs:
                    self.uniqueNameToScopedName[input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_IO.items():
            if type(node) == Node_base:
                self.uniqueNameToScopedName[key] = node.scope + '/' + node.uniqueName
            if hasattr(node, 'input_or_output'):
                self.uniqueNameToScopedName[key] = node.input_or_output + '/' + node.uniqueName
            if hasattr(node, 'scope'):
                if node.scope == '' and self.shallowestScopeName:
                    self.uniqueNameToScopedName[node.uniqueName] = self.shallowestScopeName + '/' + node.uniqueName
        # replace name
        # print(self.uniqueNameToScopedName)
        for key, node in self.nodes_IO.items():
            self.nodes_IO[key].inputs = [self.uniqueNameToScopedName[node_input_id] for node_input_id in node.inputs]
            if node.uniqueName in self.uniqueNameToScopedName:
                self.nodes_IO[key].uniqueName = self.uniqueNameToScopedName[node.uniqueName]

    def to_proto(self):
        import numpy as np
        nodes = []
        node_stats = []
        for v in self.nodes_IO.values():
            nodes.append(Node_proto(v.uniqueName,
                                    input=v.inputs,
                                    outputsize=v.tensorSize,
                                    op=v.kind,
                                    attributes=v.attributes))

            if v.tensorSize and len(v.tensorSize) > 0:  # assume data is float32, only parameter is counted
                node_stats.append(NodeExecStats(node_name=v.uniqueName,
                                                all_start_micros=int(time.time() * 1e7),
                                                all_end_rel_micros=42,
                                                memory=[AllocatorMemoryUsed(allocator_name="cpu",
                                                                            total_bytes=np.prod(v.tensorSize) * 4)]))

        return nodes, node_stats


# one argument: 'hasAttribute', 'hasAttributes',
def parse(graph, args=None, omit_useless_nodes=True):
    import torch
    n_inputs = len(args)  # not sure...

    scope = {}
    nodes_py = Graph_py()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:
            nodes_py.append(Node_py_IO(node, 'input'))
        else:
            nodes_py.append(Node_py_IO(node))  # parameter

    for node in graph.nodes():
        nodes_py.append(Node_py_OP(node))

    for node in graph.outputs():  # must place last.
        Node_py_IO(node, 'output')
    nodes_py.findCommonRoot()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def graph(model, args, verbose=False, omit_useless_nodes=True):
    import torch
    assert LooseVersion(torch.__version__) >= LooseVersion("1.0.0"),\
        'This version of tensorboardX requires pytorch>=1.0.0.'

    with torch.onnx.set_training(model, False):
        try:
            trace, _ = torch.jit.get_trace_graph(model, args)
        except RuntimeError:
            print('Error occurs, No graph saved')
            _ = model(args)  # don't catch, just print the error message
            print("Checking if it's onnx problem...")
            try:
                import tempfile
                torch.onnx.export(
                    model, args, tempfile.TemporaryFile(), verbose=True)
            except RuntimeError:
                print("Your model fails onnx too, please report to onnx team")
            return GraphDef(versions=VersionDef(producer=22))

    torch.onnx._optimize_trace(trace, torch.onnx.utils.OperatorExportTypes.ONNX)

    graph = trace.graph()
    if verbose:
        print(graph)
    list_of_nodes, node_stats = parse(graph, args, omit_useless_nodes)
    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0",
                                                                            node_stats=node_stats)]))
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
