from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python.optimizer import build_sgd, build_ftrl, build_adagrad, build_adam
from caffe2.python.optimizer_test_util import OptimizerTestBase
from caffe2.python.test_util import TestCase
from caffe2.python import workspace
import numpy as np


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
