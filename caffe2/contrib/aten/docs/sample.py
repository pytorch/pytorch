import numpy as np

from torch import nn
from torch.autograd import Variable, Function
import torch.onnx

import onnx
import caffe2.python.onnx.backend

class MyFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x*x + y
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

torch.onnx.export(MyModule(),
                  (Variable(torch.ones(3,4)), Variable(torch.ones(3,4))),
                  "output.onnx",
                  verbose=True)

# prints the graph for debugging:
# graph(%1 : Float(3, 4)
#       %2 : Float(3, 4)) {
#   %3 : Float(3, 4) = Relu(%1), uses = [%4.i0, %4.i1];
#   %4 : UNKNOWN_TYPE = ATen[operator=mul](%3, %3), uses = [%5.i0];
#   %5 : Float(3, 4) = ATen[operator=add](%4, %2), uses = [%0.i0];
#   return (%5);
# }

graph = onnx.load("output.onnx")

a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)

prepared_backend = caffe2.python.onnx.backend.prepare(graph)
W = {graph.graph.input[0].name: a, graph.graph.input[1].name: b}
c2_out = prepared_backend.run(W)[0]

x = np.maximum(a, 0)
r = x*x + b
np.testing.assert_array_almost_equal(r, c2_out)
