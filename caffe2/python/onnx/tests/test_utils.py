## @package onnx
# Module caffe2.python.onnx.tests.test_utils

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import numpy as np

class TestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(seed=0)

    def assertSameOutputs(self, outputs1, outputs2, decimal=7):
        self.assertEqual(len(outputs1), len(outputs2))
        for o1, o2 in zip(outputs1, outputs2):
            self.assertEqual(o1.dtype, o2.dtype)
            np.testing.assert_almost_equal(o1, o2, decimal=decimal)

    def add_test_case(self, name, test_func):
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(self, name, test_func)

