## @package onnx
# Module caffe2.python.onnx.tests.helper_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from caffe2.python.onnx.helper import dummy_name

from caffe2.python.onnx.tests.test_utils import TestCase
import caffe2.python._import_c_extension as C


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        g = C.DummyName()
        dummy_name(g, [])
        names_1 = [dummy_name(g) for _ in range(3)]
        dummy_name(g, [])
        names_2 = [dummy_name(g) for _ in range(3)]
        self.assertEqual(names_1, names_2)

        dummy_name(g, names_1)
        names_3 = [dummy_name(g) for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_3))

        dummy_name(g, set(names_1))
        names_4 = [dummy_name(g) for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_4))


if __name__ == '__main__':
    unittest.main()
