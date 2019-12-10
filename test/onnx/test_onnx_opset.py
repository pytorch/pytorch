from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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


def check_onnx_opsets_operator(module, x, ops, opset_versions, training=False, example_outputs=None):
    for opset_version in opset_versions:
        f = io.BytesIO()
        torch.onnx.export(module, x, f,
                          opset_version=opset_version,
                          training=training,
                          example_outputs=example_outputs)
        model = onnx.load(io.BytesIO(f.getvalue()))
        check_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOpset(TestCase):

    def test_opset_fallback(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.isnan(x)

        ops = [{"op_name" : "IsNaN"}]
        ops = {9 : ops, 10 : ops}
        x = torch.tensor([1.0, float('nan'), 2.0])
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_topk(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.topk(x, 3)

        ops_9 = [{"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2},
                 {"name": "k", "i": 3, "type": 2}]}]
        ops_10 = [{"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]}]
        ops = {9: ops_9, 10: ops_10}
        x = torch.arange(1., 6., requires_grad=True)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        # test with dynamic k
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, k):
                return torch.topk(input, k)

        ops_10 = [{"op_name": "Constant", "attributes": [{"name": "value", "type": 4}]},
                  {"op_name": "Reshape"},
                  {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]}]
        ops = {10: ops_10}
        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        module = MyModuleDynamic()
        example_output = module(x, k)
        check_onnx_opsets_operator(module, [x, k], ops,
                                   opset_versions=[10],
                                   example_outputs=example_output)

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
        check_onnx_opsets_operator(module, x, ops, opset_versions=[9, 10])

        # add test with dilations
        module = torch.nn.MaxPool1d(2, stride=1, dilation=2)

        ops_10 = [{"op_name" : "MaxPool",
                   "attributes" :
                   [{"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [2], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7}]}]
        ops = {10 : ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_upsample(self):
        class MyModule(Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x, size=size, mode='nearest')

        module = MyModule()
        ops8 = [{"op_name" : "Upsample", "attributes" : [{"name": "mode", "s": ("nearest").encode(), "type": 3},
                {"name": "scales", "floats": [1.0, 1.0, 2.0, 2.0], "type": 6}]}]
        ops9 = [{"op_name" : "Constant"},
                {"op_name" : "Upsample", "attributes" : [{"name": "mode", "s": ("nearest").encode(), "type": 3}]}]
        ops = {8 : ops8, 9 : ops9}
        x = torch.randn(2, 2, 2, 2)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_cast_constant(self):
        class MyModule(Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, x):
                return x - 1

        module = MyModule()
        ops_8 = [{"op_name" : "Constant"},
                 {"op_name" : "Cast", "attributes": [{"name": "to", "i": 7, "type": 2}]},
                 {"op_name" : "Sub"}]
        ops_9 = [{"op_name" : "Constant"}, {"op_name" : "Sub"}]
        ops = {8 : ops_8, 9 : ops_9}
        x = torch.ones(5, 6, dtype=torch.long)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_slice(self):
        class MyModule(Module):
            def forward(self, x):
                return x[0:1]

        ops_9 = [{"op_name" : "Slice",
                  "attributes" :
                  [{"name": "axes", "ints": [0], "type": 7},
                   {"name": "ends", "ints": [1], "type": 7},
                   {"name": "starts", "ints": [0], "type": 7}]}]
        ops_10 = [{"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Slice",
                   "attributes" : []}]
        ops = {9 : ops_9, 10 : ops_10}
        x = torch.randn(3)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1:x.size(0)]

        ops_10 = [{"op_name" : "Shape"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Gather",
                   "attributes" : [{"name" : "axis", "i" : 0, "type" : 2}]},
                  {"op_name" : "Unsqueeze",
                   "attributes" : [{"name" : "axes", "i" : 0, "type" : 7}]},
                  {"op_name": "Constant"},
                  {"op_name" : "Slice",
                   "attributes" : []}]
        ops = {10 : ops_10}
        module = DynamicSliceModel()
        x = torch.rand(1, 2)
        example_output = module(x)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10], example_outputs=example_output)

    def test_flip(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        ops_10 = [{"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Slice",
                   "attributes" : []}]
        ops = {10 : ops_10}
        import numpy
        x = torch.tensor(numpy.arange(6.0).reshape(2, 3))
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[10])

    def test_dropout(self):
        class MyModule(Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        x = torch.randn(1, 2, 3)

        # we should only export the onnx Dropout op in training mode; test both modes

        # test training mode
        ops = [{"op_name" : "Dropout", "attributes" : [{"name" : "ratio", "f" : 0.5, "type" : 1}]}]
        ops = {9 : ops, 10 : ops}
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10], training=True)

        # test eval mode
        ops = []
        ops = {9 : ops, 10 : ops}
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10], training=False)

    def test_full(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.full((3, 4), x)

        ops = [{"op_name" : "Constant"},
               {"op_name" : "ConstantOfShape"},
               {"op_name" : "Cast"},
               {"op_name" : "Add"}]
        ops = {9 : ops, 10 : ops}
        x = torch.tensor(12)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_interpolate(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                return torch.nn.functional.interpolate(x,
                                                       size=size,
                                                       mode='nearest')
        ops_9 = [{"op_name" : "Shape"},
                 {"op_name" : "Constant"},
                 {"op_name" : "Gather"},
                 {"op_name" : "Shape"},
                 {"op_name" : "Constant"},
                 {"op_name" : "Gather"},
                 {"op_name" : "Constant"},
                 {"op_name" : "Mul"},
                 {"op_name" : "Constant"},
                 {"op_name" : "Mul"},
                 {"op_name" : "Unsqueeze"},
                 {"op_name" : "Unsqueeze"},
                 {"op_name" : "Concat"},
                 {"op_name" : "Constant"},
                 {"op_name" : "Cast"},
                 {"op_name" : "Shape"},
                 {"op_name" : "Slice"},
                 {"op_name" : "Cast"},
                 {"op_name" : "Div"},
                 {"op_name" : "Concat"},
                 {"op_name" : "Upsample",
                  "attributes" :
                  [{"name": "mode", "s": ("nearest").encode(), "type": 3}]}]
        ops_10 = [{"op_name" : "Shape"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Gather"},
                  {"op_name" : "Shape"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Gather"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Mul"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Mul"},
                  {"op_name" : "Unsqueeze"},
                  {"op_name" : "Unsqueeze"},
                  {"op_name" : "Concat"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Cast"},
                  {"op_name" : "Shape"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Constant"},
                  {"op_name" : "Slice"},
                  {"op_name" : "Cast"},
                  {"op_name" : "Div"},
                  {"op_name" : "Concat"},
                  {"op_name" : "Resize",
                   "attributes" :
                   [{"name": "mode", "s": ("nearest").encode(), "type": 3}]}]

        ops = {9 : ops_9, 10 : ops_10}
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        check_onnx_opsets_operator(MyModel(), x, ops, opset_versions=[9, 10])

        class MyDynamicModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                # work around for now: turn the dynamic sizes into constant
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x,
                                                       size=size,
                                                       mode='nearest')
        ops_9 = [{"op_name" : "Constant"},
                 {"op_name" : "Upsample",
                  "attributes" :
                  [{"name": "mode", "s": ("nearest").encode(), "type": 3}]}]
        ops_10 = [{"op_name" : "Constant"},
                  {"op_name" : "Resize",
                   "attributes" :
                   [{"name": "mode", "s": ("nearest").encode(), "type": 3}]}]
        ops = {9 : ops_9, 10 : ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(MyDynamicModel(), x, ops, opset_versions=[9, 10])

    def test_std(self):
        class MyModule(Module):
            def forward(self, input):
                return torch.std(input, unbiased=False)

        ops = [{"op_name": "Mul"},
               {"op_name": "ReduceMean", "attributes": [{"name": "keepdims", "i": 0, "type": 2}]},
               {"op_name": "ReduceMean", "attributes": [{"name": "keepdims", "i": 0, "type": 2}]},
               {"op_name": "Mul"},
               {"op_name": "Sub"},
               {"op_name": "Abs"},
               {"op_name": "Sqrt"}]
        ops = {9: ops, 10: ops}
        x = torch.randn(2, 3, 4)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_std_along_dims(self):
        class MyModule(Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        ops = [{"op_name": "Mul"},
               {"op_name": "ReduceMean",
                "attributes": [{"name": "axes", "ints": [0, 1], "type": 7}, {"name": "keepdims", "i": 1, "type": 2}]},
               {"op_name": "ReduceMean",
                "attributes": [{"name": "axes", "ints": [0, 1], "type": 7}, {"name": "keepdims", "i": 1, "type": 2}]},
               {"op_name": "Mul"},
               {"op_name": "Sub"},
               {"op_name": "Abs"},
               {"op_name": "Constant"},
               {"op_name": "Mul"},
               {"op_name": "Constant"},
               {"op_name": "Div"},
               {"op_name": "Sqrt"}
               ]
        ops = {9: ops, 10: ops}
        x = torch.randn(2, 3, 4)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])


if __name__ == '__main__':
    run_tests()
