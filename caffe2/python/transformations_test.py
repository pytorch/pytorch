# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python.transformations import Transformer
from caffe2.python import core, workspace, test_util

transformer = Transformer()


def str_compare(a, b, encoding="utf8"):
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b


class TestTransformations(test_util.TestCase):
    def test_transformer_AddNNPACK(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        transformer.AddNNPACK(net)
        assert str_compare(net.Proto().op[0].engine, "NNPACK")

    def test_transformer_FuseNNPACKConvRelu(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 1
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg

    def test_noFuseNNPACKConvRelu(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        net.Relu(["Y"], ["Y3"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 3
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation") and str_compare(arg.s, "Relu"):
                has_activation_arg = True
        assert not has_activation_arg

    def test_transformer_FuseNNPACKConvReluNoInplace(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["X"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 1
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_transformer_FuseNNPACKConvReluInplaceRelu(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 1
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_transformer_FuseNNPACKConvReluPingPongNaming(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["X"])
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 2
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

    def test_transformer_FuseNNPACKConvReluFollowedByMultipleInputOp(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        net.Conv(["Y2", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 2
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

    def test_transformer_FuseNNPACKConvReluInplaceFollowedByMultipleInputOp(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y"])
        net.Conv(["Y", "w", "b"], ["Y2"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y2"], ["Y2"])
        transformer.AddNNPACK(net)  # get the NNPACK engine
        assert str_compare(net.Proto().op[0].engine, "NNPACK")
        transformer.FuseNNPACKConvRelu(net)
        assert len(net.Proto().op) == 2
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if str_compare(arg.name, "activation"):
                assert str_compare(arg.s, "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

    def test_transformer_SinkMaxPool(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.MaxPool(["Y"], ["Y1"], kernel=3)
        net.Relu(["Y1"], ["Y1"])
        transformer.SinkMaxPool(net)
        assert str_compare(net.Proto().op[1].type, "Relu")
        assert str_compare(net.Proto().op[2].type, "MaxPool")

    @given(
        size=st.integers(7, 10),
        input_channels=st.integers(1, 10),
        seed=st.integers(0, 65535),
        order=st.sampled_from(["NCHW", "NHWC"]),
        epsilon=st.floats(min_value=1e-5, max_value=1e-2),
    )
    def test_transformer_FuseConvBN(self, size, input_channels, seed, order, epsilon):
        workspace.ResetWorkspace()
        net = core.Net("net")
        c = input_channels
        h = size
        w = size
        k = 3
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=k, order=order)
        net.SpatialBN(
            ["Y", "scale", "bias", "mean", "var"],
            ["Y2"],
            is_test=True,
            order=order,
            epsilon=epsilon,
        )

        np.random.seed(seed)
        if order == "NCHW":
            workspace.FeedBlob("X", np.random.rand(1, c, h, w).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, c, k, k).astype(np.float32))
        else:
            workspace.FeedBlob("X", np.random.rand(1, h, w, c).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, k, k, c).astype(np.float32))
        workspace.FeedBlob("b", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("scale", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("bias", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("mean", np.random.rand(c).astype(np.float32))
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        workspace.FeedBlob("var", np.random.rand(c).astype(np.float32) + 0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert len(net.Proto().op) == 1
        workspace.RunNetOnce(net)
        postTransformOutput = workspace.FetchBlob("Y2").flatten()
        # Check that there is no numerical difference
        assert np.allclose(
            preTransformOutput,
            postTransformOutput,
            rtol=5e-02,
            atol=1e-03
        )

    @given(
        size=st.integers(7, 10),
        input_channels=st.integers(1, 10),
        seed=st.integers(0, 65535),
        order=st.sampled_from(["NCHW", "NHWC"]),
        epsilon=st.floats(min_value=1e-5, max_value=1e-2),
    )
    def test_transformer_FuseConvBNNoConvBias(self, size, input_channels, seed, order, epsilon):
        workspace.ResetWorkspace()
        net = core.Net("net")
        c = input_channels
        h = size
        w = size
        k = 3
        net.Conv(["X", "w"], ["Y"], stride=1, pad=0, kernel=k, order=order)
        net.SpatialBN(
            ["Y", "scale", "bias", "mean", "var"],
            ["Y2"],
            is_test=True,
            order=order,
            epsilon=epsilon,
        )

        np.random.seed(seed)
        if order == "NCHW":
            workspace.FeedBlob("X", np.random.rand(1, c, h, w).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, c, k, k).astype(np.float32))
        else:
            workspace.FeedBlob("X", np.random.rand(1, h, w, c).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, k, k, c).astype(np.float32))
        workspace.FeedBlob("scale", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("bias", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("mean", np.random.rand(c).astype(np.float32))
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        workspace.FeedBlob("var", np.random.rand(c).astype(np.float32) + 0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert len(net.Proto().op) == 1
        workspace.RunNetOnce(net)
        postTransformOutput = workspace.FetchBlob("Y2").flatten()
        # Check that there is no numerical difference
        assert np.allclose(
            preTransformOutput,
            postTransformOutput,
            rtol=5e-02,
            atol=1e-03
        )

    @given(
        size=st.integers(7, 10),
        input_channels=st.integers(1, 10),
        seed=st.integers(0, 65535),
        order=st.sampled_from(["NCHW", "NHWC"]),
        epsilon=st.floats(min_value=1e-5, max_value=1e-2),
    )
    def test_transformer_FuseConvBNNoConvBiasDuplicatedName(self, size, input_channels, seed, order, epsilon):
        workspace.ResetWorkspace()
        net = core.Net("net")
        c = input_channels
        h = size
        w = size
        k = 3
        net.Conv(["X", "w"], ["Y"], stride=1, pad=0, kernel=k, order=order)
        net.SpatialBN(
            ["Y", "scale", "_bias0", "mean", "var"],
            ["Y2"],
            is_test=True,
            order=order,
            epsilon=epsilon,
        )

        np.random.seed(seed)
        if order == "NCHW":
            workspace.FeedBlob("X", np.random.rand(1, c, h, w).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, c, k, k).astype(np.float32))
        else:
            workspace.FeedBlob("X", np.random.rand(1, h, w, c).astype(np.float32))
            workspace.FeedBlob("w", np.random.rand(c, k, k, c).astype(np.float32))
        workspace.FeedBlob("scale", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("_bias0", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("mean", np.random.rand(c).astype(np.float32))
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        workspace.FeedBlob("var", np.random.rand(c).astype(np.float32) + 0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert len(net.Proto().op) == 1
        workspace.RunNetOnce(net)
        postTransformOutput = workspace.FetchBlob("Y2").flatten()
        print("pre")
        print(preTransformOutput)
        print("after")
        print(postTransformOutput)
        # Check that there is no numerical difference
        assert np.allclose(
            preTransformOutput,
            postTransformOutput,
            rtol=5e-02,
            atol=1e-03
        )
