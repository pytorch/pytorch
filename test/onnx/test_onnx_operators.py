from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.nn import Module

import onnx

import io

from torch.onnx.symbolic_helper import _export_onnx_opset_version
from torch.onnx import ir_version, producer_name, producer_version


def test_onnx_opset_operator(model, ops, opset_version=_export_onnx_opset_version):
    # check_onnx_components
    assert model.ir_version == ir_version and \
        model.producer_name == producer_name and \
        model.producer_version == producer_version and \
        model.opset_import[0].version == opset_version

    # check the schema with the onnx checker
    onnx.checker.check_model(model)

    # check target type and attributes
    graph = model.graph
    assert len(ops) == len(graph.node)
    for i in range(0, len(ops)):
        assert graph.node[i].op_type == ops[i]['op_name']
        attributes = ops[i]['attributes']
        assert len(attributes) == len(graph.node[i].attribute)
        for j in range(0, len(attributes)):
            for attribute_field in attributes[j].keys():
                assert attributes[j][attribute_field] == getattr(graph.node[i].attribute[j], attribute_field)


def test_onnx_operator(module, x, ops, opset_versions):
    for opset_version in opset_versions:
        f = io.BytesIO()
        torch.onnx.export(module, x, f, opset_version=opset_version)
        model = onnx.load(io.BytesIO(f.getvalue()))
        test_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOperators(TestCase):

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
        test_onnx_operator(MyModule(), x, ops, opset_versions=[9, 10])

if __name__ == '__main__':
    run_tests()
