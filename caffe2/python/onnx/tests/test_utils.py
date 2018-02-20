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

## @package onnx
# Module caffe2.python.onnx.tests.test_utils

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np


class TestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=0)

    def assertSameOutputs(self, outputs1, outputs2, decimal=7):
        self.assertEqual(len(outputs1), len(outputs2))
        for o1, o2 in zip(outputs1, outputs2):
            np.testing.assert_almost_equal(o1, o2, decimal=decimal)

    def add_test_case(name, test_func):
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(self, name, test_func)
