## @package onnx
# Module caffe2.python.onnx.tests.helper_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from caffe2.python.onnx.tests.test_utils import TestCase
import caffe2.python._import_c_extension as C


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        g = C.DummyName()
        g.reset()
        names_1 = [g.new_dummy_name() for _ in range(3)]
        g.reset()
        names_2 = [g.new_dummy_name() for _ in range(3)]
        self.assertEqual(names_1, names_2)

        g.reset(set(names_1))
        names_3 = [g.new_dummy_name() for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_3))

        g.reset(set(names_1))
        names_4 = [g.new_dummy_name() for _ in range(3)]
        self.assertFalse(set(names_1) & set(names_4))


if __name__ == '__main__':
    unittest.main()
