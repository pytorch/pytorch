from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python.optimizer import (
        build_sgd, build_multi_precision_sgd,
        build_ftrl, build_adagrad, build_adam, add_weight_decay)
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
