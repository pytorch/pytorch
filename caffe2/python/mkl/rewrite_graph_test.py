from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import copy
from hypothesis import given
import hypothesis.strategies as st

from caffe2.python.model_helper import ModelHelper
from caffe2.python.models import resnet
from caffe2.python import workspace, brew
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl.rewrite_graph as rewrite_graph


def deterministic_io(model):
    model = copy.deepcopy(model)
    for i, op in enumerate(model.InitProto().op):
        op.device_option.random_seed = i + 1
    model.Proto().external_output.extend(
        [model.Proto().op[-1].output[0]])
    return model

def simple_fc():
    model = ModelHelper(name="r")
    brew.fc(model, "data", "fc", 10, 10)
    return model, (1, 10)


def simple_relu():
    model = ModelHelper(name="r")
    brew.relu(model, "data", "fc")
    return model, (1, 10)


def simple_mlp():
    model = ModelHelper(name="r")
    brew.relu(
        model,
        brew.fc(
            model,
            brew.relu(
                model,
                brew.fc(
                    model,
                    "data",
                    "fc1",
                    10,
                    10),
                "rl1"),
            "fc2",
            10,
            10),
        "rl2")
    return model, (1, 10)


def simple_cnn():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    brew.conv(
        model, "data", 'conv1', 3, 16, kernel=3, stride=1
    )
    brew.spatial_bn(
        model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3
    )
    brew.relu(model, 'conv1_spatbn', 'relu1')
    return model, (1, 3, 32, 32)


def simple_resnet():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    resnet.create_resnet_32x32(
        model, "data", num_input_channels=1, num_groups=1, num_labels=5,
        is_test=True)
    return model, (1, 1, 32, 32)


def complex_resnet():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    resnet.create_resnet50(
        model, "data", num_input_channels=1, num_labels=5, is_test=True,
        no_loss=True)
    return model, (1, 1, 224, 224)


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLRewriteTest(hu.HypothesisTestCase):
    @given(gen=st.sampled_from([simple_relu, simple_fc,
                                simple_mlp, simple_cnn]))
    def test_mkl_simple_rewrite(self, gen):
        cpu_model, shape = gen()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        X = np.random.randn(*shape).astype(np.float32)

        def run(model):
            self.ws.run(model.InitProto())
            self.ws.create_blob(model.Proto().external_input[0]).feed(X)
            self.ws.run(model.Proto())
            return self.ws.blobs[model.Proto().external_output[0]].fetch()

        np.testing.assert_allclose(run(cpu_model), run(mkl_model),
                                   atol=1e-4, rtol=1e-4)

    def test_mkl_resnet_rewrite(self):
        cpu_model, shape = complex_resnet()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        np.random.seed(1701)
        X = np.random.randn(*shape).astype(np.float32)

        def run(model):
            self.ws.run(model.InitProto())
            self.ws.create_blob(model.Proto().external_input[0]).feed(X)
            self.ws.run(model.Proto())
            return self.ws.blobs[model.Proto().external_output[0]].fetch()
        np.testing.assert_allclose(run(cpu_model), run(mkl_model),
                                   atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    import unittest
    unittest.main()
