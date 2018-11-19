from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python import test_util as tu
import caffe2.python.nomnigraph as ng
from caffe2.python.nomnigraph_transformations import transpose_network

import numpy as np
from hypothesis import given
import hypothesis.strategies as st


class TestNomnigraphTransformations(tu.TestCase):
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
        workspace.ResetWorkspace()
        workspace.FeedBlob("X", np.array([1, 2, 3]))
        workspace.FeedBlob("W", np.array([1, 2, 3]))
        workspace.RunNetOnce(new_netdef)
        out = workspace.FetchBlob("Y")
        expected_out = np.array([2, 4, 6])
        np.testing.assert_almost_equal(out, expected_out)

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
        workspace.ResetWorkspace()
        workspace.FeedBlob("a", np.array([1, 1, 1]))
        workspace.FeedBlob("b", np.array([2, 2, 2]))
        workspace.FeedBlob("d", np.array([3, 3, 3]))
        workspace.RunNetOnce(new_netdef)
        out = workspace.FetchBlob("e")
        expected_out = np.array([8, 8, 8])
        np.testing.assert_almost_equal(out, expected_out)

    @given(
        batch_size=st.integers(16, 20),
        channels=st.integers(1, 10),
        height=st.integers(10, 15),
        width=st.integers(10, 15),
        seed=st.integers(0, 65535),
        kernel=st.integers(3, 5),
    )
    def test_transpose_network(self, batch_size, channels, height, width, seed,
                               kernel):
        net = core.Net("net")
        net.Conv(["X", "w1", "b1"], ["c1"], stride=1, pad=0, kernel=kernel)
        net.Conv(["X", "w2", "b2"], ["c2"], stride=1, pad=0, kernel=kernel)
        # c1 and c2: batch_size, 2*channels, height - kernel + 1, width - kernel + 1
        net.Conv(["c1", "w3", "b3"], ["c3"], stride=1, pad=0, kernel=kernel)
        net.Conv(["c1", "w4", "b4"], ["c4"], stride=1, pad=0, kernel=kernel)
        # c3 and c4: batch_size, 2*channels, height - 2*kernel + 2, width - 2*kernel + 2
        net.Flatten(["c3"], "c3f")
        net.Flatten(["c4"], "c4f")
        net.Flatten(["X"], "Xf")
        net.Concat(["c3f", "c4f", "Xf"], ["out", "split_info"], axis=1, add_axis=0)
        np.random.seed(seed)
        workspace.ResetWorkspace()
        tu.randBlobFloat32("X", batch_size, channels, height, width)
        tu.randBlobsFloat32(["w1", "w2"], 2 * channels, channels, kernel, kernel)
        tu.randBlobsFloat32(["b1", "b2"], 2 * channels)
        tu.randBlobsFloat32(["w3", "w4"], 4 * channels, 2 * channels, kernel, kernel)
        tu.randBlobsFloat32(["b3", "b4"], 4 * channels)
        all_inp_names = ["X", "w1", "w2", "b1", "b2", "w3", "w4", "b3", "b4"]
        all_input = workspace.FetchBlobs(all_inp_names)
        workspace.RunNetOnce(net)
        preTransformC1 = workspace.FetchBlob("c1")
        preTransformC3 = workspace.FetchBlob("c3")
        preTransformOut = workspace.FetchBlob("out")
        nn = ng.NNModule(net)
        preTransformNumOperators = len(nn.operators)
        preTransformNumTensors = len(nn.tensors)
        transpose_network(nn)
        new_netdef = nn.convertToCaffe2Proto()
        postTransformNumOperators = len(nn.operators)
        postTransformNumTensors = len(nn.tensors)
        # The minimal number of additional operators and tensors is at least one
        # NCHW2NHWC operator and tensor for each channel-based input tensor
        # and a NHWC2NCHW operator and tensor for the output of each convolution
        # X, w1, w2, w3, w4 are channel-based inputs
        # c1, c2, c3, c4 are the outputs of convolutions
        # i.e. a total of 9.
        self.assertEqual(postTransformNumOperators,
                         preTransformNumOperators + 9,
                         "expected 9 additional operators")
        self.assertEqual(postTransformNumTensors,
                         preTransformNumTensors + 9,
                         "expected 9 additional tensors")
        workspace.ResetWorkspace()
        for name, val in zip(all_inp_names, all_input):
            workspace.FeedBlob(name, val)
        workspace.RunNetOnce(new_netdef)
        postTransformC1 = workspace.FetchBlob("c1")
        postTransformC3 = workspace.FetchBlob("c3")
        postTransformOut = workspace.FetchBlob("out")
        np.testing.assert_almost_equal(postTransformC1, preTransformC1, 1)
        np.testing.assert_almost_equal(postTransformC3, preTransformC3, 1)
        np.testing.assert_almost_equal(postTransformOut, preTransformOut, 1)
