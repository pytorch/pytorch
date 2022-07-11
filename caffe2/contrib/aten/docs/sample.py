import tempfile

import numpy as np

from torch import nn
from torch.autograd import Variable, Function
import torch.onnx

import onnx
import caffe2.python.onnx.backend

class MyFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x * x + y

    @staticmethod
    def symbolic(graph, x, y):
        x2 = graph.at("mul", x, x)
        r = graph.at("add", x2, y)
        # x, y, x2, and r are 'Node' objects
        # print(r) or print(graph) will print out a textual representation for debugging.
        # this representation will be converted to ONNX protobufs on export.
        return r

class MyModule(nn.Module):
    def forward(self, x, y):
        # you can combine your ATen ops with standard onnx ones
        x = nn.ReLU()(x)
        return MyFunction.apply(x, y)

f = tempfile.NamedTemporaryFile()
torch.onnx.export(MyModule(),
                  (Variable(torch.ones(3, 4)), Variable(torch.ones(3, 4))),
                  f, verbose=True)

# prints the graph for debugging:
# graph(%input : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
#       %y : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
#   %2 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu) = onnx::Relu(%input)
#   %3 : Tensor = aten::ATen[operator="mul"](%2, %2)
#   %4 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::ATen[operator="add"](%3, %y)
#   return (%4)

graph = onnx.load(f.name)

a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)

prepared_backend = caffe2.python.onnx.backend.prepare(graph)
W = {graph.graph.input[0].name: a, graph.graph.input[1].name: b}
c2_out = prepared_backend.run(W)[0]

x = np.maximum(a, 0)
r = x * x + b
np.testing.assert_array_almost_equal(r, c2_out)
