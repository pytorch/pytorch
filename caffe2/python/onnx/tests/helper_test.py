## @package onnx
# Module caffe2.python.onnx.tests.helper_test






import unittest

from caffe2.python.onnx.tests.test_utils import TestCase
import caffe2.python._import_c_extension as C


class TestCaffe2Basic(TestCase):
    def test_dummy_name(self):
        g = C.DummyName()
        g.reset()  # type: ignore[attr-defined]
        names_1 = [g.new_dummy_name() for _ in range(3)]  # type: ignore[attr-defined]
        g.reset()  # type: ignore[attr-defined]
        names_2 = [g.new_dummy_name() for _ in range(3)]  # type: ignore[attr-defined]
        self.assertEqual(names_1, names_2)

        g.reset(set(names_1))  # type: ignore[attr-defined]
        names_3 = [g.new_dummy_name() for _ in range(3)]  # type: ignore[attr-defined]
        self.assertFalse(set(names_1) & set(names_3))

        g.reset(set(names_1))  # type: ignore[attr-defined]
        names_4 = [g.new_dummy_name() for _ in range(3)]  # type: ignore[attr-defined]
        self.assertFalse(set(names_1) & set(names_4))


if __name__ == '__main__':
    unittest.main()
