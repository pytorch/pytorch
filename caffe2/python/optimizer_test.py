from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python.optimizer import build_sgd, build_ftrl, build_adagrad, build_adam
from caffe2.python.optimizer_test_util import OptimizerTestBase
from caffe2.python.test_util import TestCase


class TestSgd(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        build_sgd(model, base_learning_rate=0.1)


class TestFtrl(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        build_ftrl(
            model, engine=None, alpha=1.0, beta=0.1, lambda1=0.0, lambda2=0.0)


class TestAdagrad(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        build_adagrad(model, base_learning_rate=1.0)


class TestAdam(OptimizerTestBase, TestCase):
    def build_optimizer(self, model):
        build_adam(model, base_learning_rate=0.1)
