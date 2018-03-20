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
# Module caffe2.python.onnx.tests.onnx_backend_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import caffe2.python.onnx.backend as c2

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)

backend_test.exclude(r'(test_hardsigmoid'  # Does not support Hardsigmoid.
                     '|test_mean|test_hardmax'  # Does not support Mean and Hardmax.
                     '|test_cast.*FLOAT16.*'  # Does not support Cast on Float16.
                     '|test_averagepool_.*same.*)')  # Does not support Cast on Float16.

# Skip vgg to speed up CI
if 'JENKINS_URL' in os.environ:
    backend_test.exclude(r'(test_vgg19|test_vgg)')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
