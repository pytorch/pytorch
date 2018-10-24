from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, test_util
from caffe2.proto import caffe2_pb2
import caffe2.python.nomnigraph as ng

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import random


class TestNomnigraphTransformations(test_util.TestCase):
    def test_simple_replace(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        fc = nn.controlFlow[0]
        add = nn.createNode(core.CreateOperator("Add", ["X"], ["Y"], engine="CUDNN"))
        nn.replaceNode(fc, add)
        nn.deleteNode(fc)

        # Test it out
        new_netdef = nn.convertToCaffe2Proto()
        workspace.FeedBlob("X", np.array([1, 2, 3]))
        workspace.FeedBlob("W", np.array([1, 2, 3]))
        workspace.RunNetOnce(new_netdef)
        out = workspace.FetchBlob("Y")
        expected_out = np.array([2, 4, 6])
        np.allclose(out, expected_out)

    def test_simple_rewire(self):
        net = core.Net("name")
        # Rewire this so that we get
        # c = Add(a, d)
        # e = Mul(c, b)
        #
        # if a = 1, b = 2, d = 3
        # we get 8: (1 + 3) * 2
        # as opposed to 7: 1 + (3 * 2)
        net.Mul(["a", "b"], ["c"])
        net.Add(["c", "d"], ["e"])
        nn = ng.NNModule(net)

        mul = nn.controlFlow[0]
        add = nn.controlFlow[1]
        a = mul.inputs[0]
        b = mul.inputs[1]
        c = mul.outputs[0]
        d = add.inputs[1]
        e = add.outputs[0]

        nn.deleteEdge(a, mul)
        nn.deleteEdge(b, mul)
        nn.deleteEdge(mul, c)
        nn.deleteEdge(c, add)
        nn.deleteEdge(d, add)
        nn.deleteEdge(add, e)

        nn.createEdge(a, add)
        nn.createEdge(d, add)
        nn.createEdge(add, c)
        nn.createEdge(c, mul)
        nn.createEdge(b, mul)
        nn.createEdge(mul, e)

        # Test it out
        new_netdef = nn.convertToCaffe2Proto()
        workspace.FeedBlob("a", np.array([1, 1, 1]))
        workspace.FeedBlob("b", np.array([2, 2, 2]))
        workspace.FeedBlob("d", np.array([3, 3, 3]))
        workspace.RunNetOnce(new_netdef)
        out = workspace.FetchBlob("e")
        expected_out = np.array([8, 8, 8])
        np.allclose(out, expected_out)
