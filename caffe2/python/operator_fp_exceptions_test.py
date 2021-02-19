



from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

import numpy as np
import unittest


def setThrowIfFpExceptions(enabled):
    core.GlobalInit(["caffe2", "--caffe2_operator_throw_if_fp_exceptions=%d" % (1 if enabled else 0)])


class OperatorFPExceptionsTest(TestCase):
    def test_fp_exception_divbyzero(self):
        # This test asserts the followings
        # - If flag caffe2_operator_throw_if_fp_exceptions is set,
        # floating point exceptions will be thrown
        # - If flag caffe2_operator_throw_if_fp_exceptions is not set,
        # floating point exceptions will not be thrown
        workspace.blobs["0"] = np.array([0.0], dtype=np.float32)
        workspace.blobs["1"] = np.array([1.0], dtype=np.float32)

        net = core.Net("test_fp")
        net.Div(["1", "0"], "out")

        for throw_if_fp_exceptions in (True, False):
            setThrowIfFpExceptions(throw_if_fp_exceptions)
            exception_raised = False
            try:
                workspace.RunNetOnce(net)
            except Exception as e:
                exception_raised = True
            self.assertEquals(exception_raised, throw_if_fp_exceptions)


if __name__ == '__main__':
    unittest.main()
