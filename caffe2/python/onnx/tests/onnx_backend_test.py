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
                     '|test_depthtospace.*'  # Does not support DepthToSpace.
                     '|test_.*pool_.*same.*)')  # Does not support pool same.

# Skip vgg to speed up CI
if 'JENKINS_URL' in os.environ:
    backend_test.exclude(r'(test_vgg19|test_vgg)')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
