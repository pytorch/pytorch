from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace
from caffe2.quantization.server import dnnlowp_pybind11

net = core.Net("test_net")

X = np.array([[1, 2], [3, 4]]).astype(np.float32)
W = np.array([[5, 6], [7, 8]]).astype(np.float32)
b = np.array([0, 1]).astype(np.float32)

workspace.FeedBlob("X", X)
workspace.FeedBlob("W", W)
workspace.FeedBlob("b", b)

Y = net.FC(["X", "W", "b"], ["Y"])

dnnlowp_pybind11.ObserveMinMaxOfOutput("test_net.minmax", 1)
workspace.CreateNet(net)
workspace.RunNet(net)
print(workspace.FetchBlob("Y"))

workspace.ResetWorkspace()

workspace.FeedBlob("X", X)
workspace.FeedBlob("W", W)
workspace.FeedBlob("b", b)

dnnlowp_pybind11.ObserveHistogramOfOutput("test_net.hist", 1)
workspace.CreateNet(net)
workspace.RunNet(net)
