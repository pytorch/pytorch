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
# Module caffe2.python.onnx.tests.helper_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from caffe2.python.onnx.helper import dummy_name

from caffe2.python.onnx.tests.test_utils import TestCase


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        dummy_name([])
        names_1 = [dummy_name() for _ in range(3)]
        dummy_name([])
        names_2 = [dummy_name() for _ in range(3)]
        self.assertEqual(names_1, names_2)

        dummy_name(names_1)
        names_3 = [dummy_name() for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_3))

        dummy_name(set(names_1))
        names_4 = [dummy_name() for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_4))


if __name__ == '__main__':
    unittest.main()
