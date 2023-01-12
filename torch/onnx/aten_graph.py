import torch
import numpy as np
from os_graph_builder import GraphBuilder

class ATENNode(object):
    def __init__(self, op_name, inputs, outputs, value=None):
        """Create Node.
        
        %26 : Long(device=cpu) = prim::Constant[value={5}]()
        %24 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::add(%16, %26, %25), scope: __main__.MixedOpsModel::

        """
        self._op = op_name
        self._op_domain = op_name.split("::")[0]
        self._inputs = inputs
        self._outputs = outputs
        self._value = value

    def _list_to_str(self, dim_list):
        return ", ".join([str(dim) for dim in dim_list])

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def op(self):
        return self._op

    @property
    def value(self):
        return self._value

    @property
    def is_prim(self):
        return self._op_domain == "prim"

    @property
    def input_names(self):
        if self._inputs is not None and self.inputs:
            return self._list_to_str([input.name for input in self.inputs])
        return ""

    def to_str(self):
        if self.is_prim:
            return "{} = {}[value={{{}}}]()".format(self._outputs[0].to_str(), self.op, self.value)
        else:
            return "{} = {}({})".format(self._outputs[0].to_str(), self.op, self.input_names)


class IOIdentity(ATENNode):
    def __init__(self, name, dtype, shape=[], device="cpu"):
        self._name = name
        self._dtype = dtype
        self._shape = shape
        self._strides = self._get_strides(self._shape)
        self._requires_grad = 0
        self._device = device

    def _get_strides(self, shape):
        strides = []
        for i in range(len(shape)-1):
            strides.append(np.prod(shape[i+1:]))
        strides.append(1)
        return strides

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        # This is an array, for example, [2, 3, 4]
        return self._shape

    @property
    def shape_str(self):
        return self._list_to_str(self._shape)

    @property
    def strides_str(self):
        return self._list_to_str(self._strides)

    def to_str(self):
        """
        %26 : Long(device=cpu) = prim::Constant[value={5}]()
        %24 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::add(%16, %26, %25), scope: __main__.MixedOpsModel::
        """
        if self._shape is not None and self._shape:
            return "{} : {}({}, strides=[{}], requires_grad={}, device={})".format(self.name, str(self.dtype), self.shape_str, self.strides_str, 0, self.device)
        else:
            return "{} : {}(device={})".format(self.name, str(self.dtype), self.device)


class ATENGraph(object):
    def __init__(self, inputs, outputs):
        self._nodes = []
        self._inputs = list(inputs)
        self._outputs = list(outputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def nodes(self):
        return self._nodes

    def add_node(self, aten_node):
        self._nodes.append(aten_node)

    def print(self):
        result = []
        for input in self._inputs:
            result.append(input.to_str())
        result = ["graph({}):".format("\n".join(result))]

        for node in self._nodes:
            result.append("\t" + node.to_str())
        result = "\n".join(result)
        print(result)

def aten_add(gb, node):
    gb.add_node(node)
    print("====== aten_add works. call graph builder.")


def aten_add(gb, node):
    gb.add_node(node)
    print("====== aten_add works. call graph builder.")
func_mapping = {
    "aten::add":aten_add,}

def create_example_graph():
    """
    Create an ATEN example graph.
    graph(%0 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu),
          %1 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu)):
        %25 : Long(device=cpu) = prim::Constant[value={1}](), scope: __main__.MixedOpsModel::
        %16 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::add(%0, %1, %25), scope: __main__.MixedOpsModel::
        %26 : Long(device=cpu) = prim::Constant[value={5}]()
        %24 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::add(%16, %26, %25), scope: __main__.MixedOpsModel::
        %21 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::sub(%0, %1, %25), scope: __main__.MixedOpsModel::
        %22 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu) = aten::mul(%24, %21), scope: __main__.MixedOpsModel::
        return (%22)
    """
    input_0 = IOIdentity("%0", torch.float, [2, 3, 4])
    input_1 = IOIdentity("%1", torch.float, [2, 3, 4])

    graph_inputs = [input_0, input_1]

    output_0 = IOIdentity("%22", torch.float, [2, 3, 4])
    graph_outputs = [output_0]

    graph = ATENGraph(graph_inputs, graph_outputs)
    node_25_output = IOIdentity("%25", torch.long)
    node_25 = ATENNode("prim::Constant", None, [node_25_output], 1)
    graph.add_node(node_25)

    node_16_inputs = graph_inputs + [node_25_output]
    node_16_output = IOIdentity("%16", torch.float, [2, 3, 4])
    node_16 = ATENNode("aten::add", node_16_inputs, [node_16_output])
    graph.add_node(node_16)

    node_26_output = IOIdentity("%26", torch.long)
    node_26 = ATENNode("prim::Constant", None, [node_26_output], 5)
    graph.add_node(node_26)

    node_24_output = IOIdentity("%24", torch.float, [2, 3, 4])
    node_24 = ATENNode("aten::add", [node_16_output, node_26_output, node_25_output], [node_24_output])
    graph.add_node(node_24)

    node_21_output = IOIdentity("%21", torch.float, [2, 3, 4])
    node_21 = ATENNode("aten::sub", [input_0, input_1, node_25_output], [node_21_output])
    graph.add_node(node_21)

    node_22 = ATENNode("aten::mul", [node_24_output, node_21_output], [output_0])
    graph.add_node(node_22)

    return graph

gb : GraphBuilder = GraphBuilder()

example = create_example_graph()
example.print()

for node in example.nodes:
    if node.op in func_mapping:
        func_mapping[node.op](gb, node)

model_name = "os_test_model.onnx"
onnx_model = gb.make_model(model_name)

print("====== End ======")
