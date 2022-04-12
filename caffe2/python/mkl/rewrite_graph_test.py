




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
    if not model.Proto().external_output:
        model.Proto().external_output.extend([model.Proto().op[-1].output[0]])
    return model

def simple_fc():
    model = ModelHelper(name="r")
    brew.fc(model, "data", "fc", 10, 10)
    return model, [(1, 10)]

def double_matmul():
    model = ModelHelper(name="r")
    fc0 = brew.fc(model, "data", "fc0", 10, 10)
    fc1 = brew.fc(model, fc0, "fc1", 10, 10)
    model.Proto().external_output[:] = [str(fc0), str(fc1)]
    return model, [(1, 10)]

def simple_relu():
    model = ModelHelper(name="r")
    brew.relu(model, "data", "fc")
    return model, [(1, 10)]


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
    return model, [(1, 10)]


def simple_cnn():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    brew.conv(
        model, "data", 'conv1', 3, 16, kernel=3, stride=1
    )
    brew.spatial_bn(
        model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3
    )
    brew.relu(model, 'conv1_spatbn', 'relu1')
    return model, [(1, 3, 32, 32)]


def alexnet():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    conv1 = brew.conv(
        model,
        "data",
        "conv1",
        3,
        64,
        11, ('XavierFill', {}), ('ConstantFill', {}),
        stride=4,
        pad=2
    )
    relu1 = brew.relu(model, conv1, "conv1")
    pool1 = brew.max_pool(model, relu1, "pool1", kernel=3, stride=2, pad=0,
                          legacy_pad=3)
    lrn1 = brew.lrn(
        model, pool1, "pool1_lrn", size=5, alpha=1.0e-4, beta=0.75, bias=1.0)
    conv2 = brew.conv(
        model,
        lrn1,
        "conv2",
        64,
        192,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=2
    )
    relu2 = brew.relu(model, conv2, "conv2")
    pool2 = brew.max_pool(model, relu2, "pool2", kernel=3, stride=2)
    lrn2 = brew.lrn(
        model, pool2, "pool2_lrn", size=5, alpha=1.0e-4, beta=0.75, bias=1.0)
    conv3 = brew.conv(
        model,
        lrn2,
        "conv3",
        192,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = brew.relu(model, conv3, "conv3")
    conv4 = brew.conv(
        model,
        relu3,
        "conv4",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu4 = brew.relu(model, conv4, "conv4")
    conv5 = brew.conv(
        model,
        relu4,
        "conv5",
        256,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu5 = brew.relu(model, conv5, "conv5")
    pool5 = brew.max_pool(model, relu5, "pool5", kernel=3, stride=2)
    fc6 = brew.fc(
        model,
        pool5, "fc6", 256 * 6 * 6, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu6 = brew.relu(model, fc6, "fc6")
    fc7 = brew.fc(
        model, relu6, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu7 = brew.relu(model, fc7, "fc7")
    drop7 = brew.dropout(model, relu7, "fc7_dropout", is_test=1, ratio=0.5)
    fc8 = brew.fc(
        model, drop7, "fc8", 4096, 1000, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu8 = brew.relu(model, fc8, "fc8")
    brew.dropout(model, relu8, "fc8_dropout", is_test=1, ratio=0.5)
    return model, [(1, 3, 224, 224)]


def simple_resnet():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    resnet.create_resnet_32x32(
        model, "data", num_input_channels=1, num_groups=1, num_labels=5,
        is_test=True)
    return model, [(1, 1, 32, 32)]


def complex_resnet():
    model = ModelHelper(name="r", arg_scope={"order": "NCHW", "is_test": True})
    resnet.create_resnet50(
        model, "data", num_input_channels=1, num_labels=5, is_test=True,
        no_loss=True)
    return model, [(1, 1, 224, 224)]


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class MKLRewriteTest(hu.HypothesisTestCase):
    @given(gen=st.sampled_from([simple_relu, simple_fc,
                                simple_mlp, simple_cnn]))
    def test_mkl_simple_rewrite(self, gen):
        cpu_model, (shape,) = gen()
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
        cpu_model, (shape,) = complex_resnet()
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

    def test_mkl_multi_output_rewrite(self):
        cpu_model, shapes = double_matmul()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        np.random.seed(1701)
        Xs = [np.random.randn(*shape).astype(np.float32) for shape in shapes]

        def run(model):
            self.ws.run(model.InitProto())
            for (name, X) in zip(model.Proto().external_input, Xs):
                self.ws.create_blob(name).feed(X)
            print(model.Proto())
            self.ws.run(model.Proto())
            return [self.ws.blobs[name].fetch()
                    for name in model.Proto().external_output]

        run(mkl_model)

        np.testing.assert_allclose(run(cpu_model), run(mkl_model),
                                   atol=1e-4, rtol=1e-4)

    def test_mkl_alexnet_rewrite(self):
        cpu_model, (shape,) = alexnet()
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
