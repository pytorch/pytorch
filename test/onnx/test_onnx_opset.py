from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.nn import Module

import onnx

import io

from torch.onnx.symbolic_helper import _export_onnx_opset_version
from torch.onnx import ir_version, producer_name, producer_version


def check_onnx_opset_operator(model, ops, opset_version=_export_onnx_opset_version):
    # check_onnx_components
    assert model.ir_version == ir_version and \
        model.producer_name == producer_name and \
        model.producer_version == producer_version and \
        model.opset_import[0].version == opset_version

    # check the schema with the onnx checker
    onnx.checker.check_model(model)

    # check target type and attributes 
    graph = model.graph
    # ops should contain an object for each node
    # in graph.node, in the right order.
    # At least the op_name should be specified,
    # but the op's attributes can optionally be
    # specified as well
    assert len(ops) == len(graph.node)
    for i in range(0, len(ops)):
        assert graph.node[i].op_type == ops[i]['op_name']
        if "attributes" in ops[i] :
            attributes = ops[i]['attributes']
            assert len(attributes) == len(graph.node[i].attribute)
            for j in range(0, len(attributes)):
                for attribute_field in attributes[j].keys():
                    assert attributes[j][attribute_field] == getattr(graph.node[i].attribute[j], attribute_field)


def check_onnx_opsets_operator(module, x, ops, opset_versions):
    for opset_version in opset_versions:
        f = io.BytesIO()
        torch.onnx.export(module, x, f, opset_version=opset_version)
        model = onnx.load(io.BytesIO(f.getvalue()))
        check_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOpset(TestCase):

    def test_opset_fallback(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.isnan(x)

        ops = [{"op_name" : "IsNaN"},
               {"op_name" : "Cast", "attributes" : [{"name" : "to", "i" : 2, "type" : 2}]}]
        ops = {9 : ops, 10 : ops}
        x = torch.tensor([1.0, float('nan'), 2.0])
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_topk(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.topk(x, 3)

        ops_9 = [{"op_name" : "TopK", "attributes" : [{"name" : "axis", "i" : -1, "type" : 2},
                 {"name" : "k", "i" : 3, "type" : 2}]}]
        ops_10 = [{"op_name" : "Constant", "attributes" : [{"name" : "value", "type" : 4}]},
                  {"op_name" : "Unsqueeze", "attributes" : [{"name" : "axes", "ints" : [0], "type" : 7}]},
                  {"op_name" : "TopK", "attributes" : [{"name" : "axis", "i" : -1, "type" : 2}]}]
        ops = {9 : ops_9, 10 : ops_10}
        x = torch.arange(1., 6., requires_grad=True)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_maxpool(self):
        module = torch.nn.MaxPool1d(2, stride=1)

        ops_9 = [{"op_name" : "MaxPool",
                  "attributes" :
                  [{"name": "kernel_shape", "ints": [2], "type": 7},
                   {"name": "pads", "ints": [0, 0], "type": 7},
                   {"name": "strides", "ints": [1], "type": 7}]}]
        ops_10 = [{"op_name" : "MaxPool",
                   "attributes" :
                   [{"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7}]}]
        ops = {9 : ops_9, 10 : ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

        # add test with dilations
        module = torch.nn.MaxPool1d(2, stride=1, dilation=2)

        ops_10 = [{"op_name" : "MaxPool",
                   "attributes" :
                   [{"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [2], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7}]}]
        ops = {9 : ops_9, 10 : ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

if __name__ == '__main__':
    run_tests()
