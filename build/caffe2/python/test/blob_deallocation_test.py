from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, workspace
import unittest

core.GlobalInit(['python'])


class BlobDeallocationTest(unittest.TestCase):
    def test(self):
        net = core.Net('net')

        x = net.GivenTensorStringFill([], ['x'], shape=[3], values=['a', 'b', 'c'])
        y = net.GivenTensorStringFill([], ['y'], shape=[3], values=['d', 'e', 'f'])
        net.Concat([x, y], ['concated', '_'], axis=0)

        workspace.ResetWorkspace()
        workspace.RunNetOnce(net)

        workspace.ResetWorkspace()
        workspace.RunNetOnce(net)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
