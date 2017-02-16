from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python.sgd import build_sgd, build_adagrad, build_adam
from caffe2.python.sgd_test_util import TestBase
from caffe2.python.test_util import TestCase


class TestSgd(TestBase, TestCase):
    def build_optimizer(self, model):
        build_sgd(model, base_learning_rate=0.1)


class TestAdagrad(TestBase, TestCase):
    def build_optimizer(self, model):
        build_adagrad(model, base_learning_rate=1.0)


class TestAdam(TestBase, TestCase):
    def build_optimizer(self, model):
        build_adam(model, base_learning_rate=0.1)
