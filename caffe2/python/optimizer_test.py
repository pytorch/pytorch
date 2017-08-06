from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python.optimizer import (
    build_sgd, build_multi_precision_sgd, build_ftrl,
    build_adagrad, build_adam, add_weight_decay, SgdOptimizer)
from caffe2.python.optimizer_context import UseOptimizer
from caffe2.python.optimizer_test_util import OptimizerTestBase
from caffe2.python.test_util import TestCase
from caffe2.python import workspace
from caffe2.python.core import DataType
import numpy as np
import unittest


class TestSgd(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        self._skip_gpu = False
        return build_sgd(model, base_learning_rate=0.1)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertFalse(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)


class TestMultiPrecisionSgd(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        self._skip_gpu = False
        return build_multi_precision_sgd(model, base_learning_rate=0.1)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertFalse(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)

    @unittest.skipIf(not workspace.has_gpu_support, "No GPU support")
    def testGPUDense(self):
        super(TestMultiPrecisionSgd, self).testGPUDense(DataType.FLOAT16)


class TestFtrl(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        self._skip_gpu = True
        return build_ftrl(
            model, engine=None, alpha=1.0, beta=0.1, lambda1=0.0, lambda2=0.0)

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestAdagrad(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        self._skip_gpu = False
        return build_adagrad(model, base_learning_rate=1.0)

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestAdam(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        self._skip_gpu = False
        return build_adam(model, base_learning_rate=0.1)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        self.assertTrue(workspace.HasBlob("optimizer_iteration"))
        iteration_tensor = workspace.FetchBlob("optimizer_iteration")
        np.testing.assert_allclose(np.array([2000]),
                                   iteration_tensor,
                                   atol=1e-5)
        for param in optimizer.get_auxiliary_parameters().shared:
            workspace.FetchBlob(param)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestMultiOptimizers(TestCase):
    def test_multiple_optimizers(self):
        from caffe2.python import brew, core, optimizer
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test")
        fc1 = brew.fc(model, 'data', 'fc1', 100, 50)
        fc2 = brew.fc(model, fc1, 'fc2', 50, 25)
        pred = brew.fc(model, fc2, 'fc3', 25, 10)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        param_to_device = optimizer._get_param_to_device(model)

        def infer_blob_device(blob_name):
            return optimizer.get_param_device(
                blob_name, "{}_grad".format(blob_name), param_to_device
            )

        sgd_1 = optimizer.SgdOptimizer(base_learning_rate=0.1)
        sgd_2 = optimizer.SgdOptimizer(base_learning_rate=0.2)
        adagrad = optimizer.AdagradOptimizer()

        # Check same optimizer share the same learning rate.
        with core.DeviceScope(infer_blob_device("fc1_w")):
            sgd_1(model.net, model.param_init_net, "fc1_w", "fc1_w_grad")
        with core.DeviceScope(infer_blob_device("fc1_b")):
            sgd_1(model.net, model.param_init_net, "fc1_b", "fc1_b_grad")
        fc1_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and op.input[0] == 'fc1_w' or \
                    op.input[0] == 'fc1_b':
                        fc1_lr_blobs.append(op.input[3])
        self.assertEqual(fc1_lr_blobs[0], fc1_lr_blobs[1])

        # Check different instance of the same optimizer has a different lr.
        with core.DeviceScope(infer_blob_device("fc2_w")):
            sgd_2(model.net, model.param_init_net, "fc2_w", "fc2_w_grad")
        with core.DeviceScope(infer_blob_device("fc2_b")):
            sgd_2(model.net, model.param_init_net, "fc2_b", "fc2_b_grad")
        fc2_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and op.input[0] == 'fc2_w' or \
                    op.input[0] == 'fc2_b':
                        self.assertTrue(op.input[3] not in fc1_lr_blobs)
                        fc2_lr_blobs.append(op.input[3])
        self.assertEqual(fc2_lr_blobs[0], fc2_lr_blobs[1])

        # Check different optimizer type case
        with core.DeviceScope(infer_blob_device("fc3_w")):
            adagrad(model.net, model.param_init_net, "fc3_w", "fc3_w_grad")
        with core.DeviceScope(infer_blob_device("fc3_b")):
            adagrad(model.net, model.param_init_net, "fc3_b", "fc3_b_grad")
        fc3_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'Adagrad' and op.input[0] == 'fc3_w' or \
                    op.input[0] == 'fc3_b':
                        self.assertTrue(op.input[3] not in fc2_lr_blobs)
                        self.assertTrue(op.input[3] not in fc1_lr_blobs)
                        fc3_lr_blobs.append(op.input[3])
        self.assertEqual(fc3_lr_blobs[0], fc3_lr_blobs[1])


class TestWeightDecay(TestCase):

    def test_weight_decay(self):
        from caffe2.python import brew
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        cnv = brew.conv(model, 'data', 'cnv', 32, 32, 4)
        a = brew.fc(model, cnv, 'a', 100, 200)
        pred = brew.fc(model, a, 'b', 200, 5)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        add_weight_decay(model, weight_decay=1e-4)
        build_sgd(model, 0.11)

        expected_weight_grad = {'b_w_grad', 'a_w_grad', 'cnv_w_grad'}

        # Check the proto that all weights are decayed and not non-weights
        # are decayed.
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and 'wd_0_0' in op.input:
                if op.output[0] not in expected_weight_grad:
                    print(
                        "Unexpected param for weight_decay: {}".
                        format(op.output[0])
                    )
                self.assertTrue(op.output[0] in expected_weight_grad)
                expected_weight_grad.remove(op.output[0])

        self.assertEqual(
            expected_weight_grad,
            set(),
            "Not all weights were decayed: {}".format(expected_weight_grad)
        )


class TestOptimizerContext(TestCase):

    def test_optimizer_context(self):
        from caffe2.python import brew, optimizer
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        count = optimizer._optimizer_instance_count['SgdOptimizer']
        cnv_optim = SgdOptimizer(0.15)
        weight_optim = SgdOptimizer(0.2)
        bias_optim = SgdOptimizer(0.1)

        with UseOptimizer(cnv_optim):
            cnv = brew.conv(model, 'data', 'cnv', 32, 32, 4)
        with UseOptimizer({'WEIGHT': weight_optim, 'BIAS': bias_optim}):
            a = brew.fc(model, cnv, 'a', 100, 200)
        pred = brew.fc(model, a, 'b', 200, 5)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        add_weight_decay(model, weight_decay=1e-4)
        # use the following optimizer if none specified in param_info
        build_sgd(model, 0.11)
        expected_weight_grad = {'b_w_grad', 'a_w_grad', 'cnv_w_grad'}
        expected_learning_rate = {
            "SgdOptimizer_{}_lr_cpu".format(count): -0.15,
            "SgdOptimizer_{}_lr_cpu".format(count + 1): -0.2,
            "SgdOptimizer_{}_lr_cpu".format(count + 2): -0.1,
            "SgdOptimizer_{}_lr_cpu".format(count + 3): -0.11
        }

        for op in model.net.Proto().op:
            # Check the proto that all weights are decayed and not non-weights
            # are decayed.
            if op.type == 'WeightedSum' and 'wd_0_0' in op.input:
                if op.output[0] not in expected_weight_grad:
                    print(
                        "Unexpected param for weight_decay: {}".
                        format(op.output[0])
                    )
                self.assertTrue(op.output[0] in expected_weight_grad)
                expected_weight_grad.remove(op.output[0])
            # Check the learning rate for each parameter
            if op.type == 'LearningRate':
                val = 0
                for arg in op.arg:
                    if arg.name == 'base_lr':
                        val = arg.f
                self.assertAlmostEqual(
                    val,
                    expected_learning_rate[op.output[0]]
                )

        self.assertEqual(
            expected_weight_grad,
            set(),
            "Not all weights were decayed: {}".format(expected_weight_grad)
        )
