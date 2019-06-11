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
from caffe2.python import core, workspace
from caffe2.python import test_util as tu

transformer = Transformer()


class TestTransformations(tu.TestCase):
    def _base_test_net(self):
        net = core.Net("net")
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        return net

    def _add_nnpack(self, net):
        transformer.AddNNPACK(net)
        assert tu.str_compare(net.Proto().op[0].engine, "NNPACK")

    def _fuse_nnpack_convrelu(self, net, expected_result_num_ops,
    expected_activation_arg=True):
        self._add_nnpack(net)
        transformer.FuseNNPACKConvRelu(net)
        self.assertEquals(tu.numOps(net), expected_result_num_ops)
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if tu.str_compare(arg.name, "activation"):
                assert tu.str_compare(arg.s, "Relu")
                has_activation_arg = True
        if expected_activation_arg:
            assert has_activation_arg
        else:
            assert not has_activation_arg

    def test_transformer_AddNNPACK(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y2"])
        self._add_nnpack(net)

    def test_transformer_FuseNNPACKConvRelu(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y2"])
        self._fuse_nnpack_convrelu(net, 1)

    def test_noFuseNNPACKConvRelu(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y2"])
        net.Relu(["Y"], ["Y3"])
        self._fuse_nnpack_convrelu(net, 3, expected_activation_arg=False)

    def test_transformer_FuseNNPACKConvReluNoInplace(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["X"])
        self._fuse_nnpack_convrelu(net, 1)
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_transformer_FuseNNPACKConvReluInplaceRelu(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y"])
        self._fuse_nnpack_convrelu(net, 1)
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_transformer_FuseNNPACKConvReluPingPongNaming(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["X"])
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        self._fuse_nnpack_convrelu(net, 2)
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

    def test_transformer_FuseNNPACKConvReluFollowedByMultipleInputOp(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y2"])
        net.Conv(["Y2", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        self._fuse_nnpack_convrelu(net, 2)
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

    def test_transformer_FuseNNPACKConvReluInplaceFollowedByMultipleInputOp(self):
        net = self._base_test_net()
        net.Relu(["Y"], ["Y"])
        net.Conv(["Y", "w", "b"], ["Y2"], stride=1, pad=0, kernel=3, order="NCHW")
        net.Relu(["Y2"], ["Y2"])
        self._fuse_nnpack_convrelu(net, 2)
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]
        assert net.Proto().op[1].output[0] != net.Proto().op[1].input[0]

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
            tu.randBlobFloat32("X", 1, c, h, w)
            tu.randBlobFloat32("w", c, c, k, k)
        else:
            tu.randBlobFloat32("X", 1, h, w, c)
            tu.randBlobFloat32("w", c, k, k, c)
        tu.randBlobsFloat32(["b", "scale", "bias", "mean"], c)

        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        tu.randBlobFloat32("var", c, offset=0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert tu.numOps(net) == 1
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
            tu.randBlobFloat32("X", 1, c, h, w)
            tu.randBlobFloat32("w", c, c, k, k)
        else:
            tu.randBlobFloat32("X", 1, h, w, c)
            tu.randBlobFloat32("w", c, k, k, c)
        tu.randBlobsFloat32(["scale", "bias", "mean"], c)
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        tu.randBlobFloat32("var", c, offset=0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert tu.numOps(net) == 1
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
            tu.randBlobFloat32("X", 1, c, h, w)
            tu.randBlobFloat32("w", c, c, k, k)
        else:
            tu.randBlobFloat32("X", 1, h, w, c)
            tu.randBlobFloat32("w", c, k, k, c)
        tu.randBlobsFloat32(["scale", "_bias0", "mean"], c)
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        tu.randBlobFloat32("var", c, offset=0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert tu.numOps(net) == 1
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

    @given(
        size=st.integers(7, 10),
        input_channels=st.integers(1, 10),
        kt=st.integers(3, 5),
        kh=st.integers(3, 5),
        kw=st.integers(3, 5),
        seed=st.integers(0, 65535),
        epsilon=st.floats(min_value=1e-5, max_value=1e-2),
    )
    def test_transformer_FuseConv3DBN(
        self, size, input_channels, kt, kh, kw, seed, epsilon
    ):
        workspace.ResetWorkspace()
        net = core.Net("net")
        c = input_channels
        t = size
        h = size
        w = size
        net.Conv(
            ["X", "w", "b"],
            ["Y"],
            kernels=[kt, kh, kw],
        )
        net.SpatialBN(
            ["Y", "scale", "bias", "mean", "var"],
            ["Y2"],
            is_test=True,
            epsilon=epsilon,
        )

        np.random.seed(seed)
        tu.randBlobFloat32("X", 1, c, t, h, w)
        tu.randBlobFloat32("w", c, c, kt, kh, kw)
        tu.randBlobsFloat32(["b", "scale", "bias", "mean"], c)
        # This is necessary because 1/sqrt(var) is used and if var is too small
        # we get floating point artifacts that cause test failures
        tu.randBlobFloat32("var", c, offset=0.5)
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2").flatten()
        workspace.FeedBlob("Y2", np.zeros((1, 1)))
        transformer.FuseConvBN(net)

        # Ensure fusion
        assert tu.numOps(net) == 1
        workspace.RunNetOnce(net)
        postTransformOutput = workspace.FetchBlob("Y2").flatten()
        # Check that there is no numerical difference
        assert np.allclose(
            preTransformOutput,
            postTransformOutput,
            rtol=1e-02,
            atol=1e-04
        )

    def test_converterDontEnforceUnusedInputs(self):
        net = core.Net("net")
        net.Relu(["X"], ["Y"])
        net.Proto().external_input.extend(["fake"])
        # This should now work
        transformer.AddNNPACK(net)  # just testing the converter

    def test_converterDontEnforceUnusedOutputs(self):
        net = core.Net("net")
        net.Relu(["X"], ["Y"])
        net.Proto().external_output.extend(["fake"])
        transformer.AddNNPACK(net)  # just testing the converter
