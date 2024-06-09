




import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestDuplicateOperands(TestCase):
    def test_duplicate_operands(self):
        net = core.Net('net')
        shape = (2, 4)
        x_in = np.random.uniform(size=shape)
        x = net.GivenTensorFill([], 'X', shape=shape,
                                values=x_in.flatten().tolist())
        xsq = net.Mul([x, x])
        y = net.DotProduct([xsq, xsq])
        net.AddGradientOperators([y])
        workspace.RunNetOnce(net)
        self.assertTrue(np.allclose(workspace.FetchBlob('X_grad'),
                                    4 * x_in**3))

if __name__ == "__main__":
    import unittest
    unittest.main()
