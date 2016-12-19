from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestTensorPackOps(TestCase):
    def test_pack_ops(self):
        workspace.FeedBlob('l', np.array([1, 2, 3], dtype=np.int32))
        workspace.FeedBlob(
            'd',
            np.array([
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0]],
                dtype=np.float32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t']))
        workspace.RunOperatorOnce(core.CreateOperator(
            'UnpackSegments', ['l', 't'], ['newd']))
        assert((workspace.FetchBlob('newd') == workspace.FetchBlob('d')).all())
        workspace.FeedBlob('l', np.array([1, 2, 3], dtype=np.int64))
        strs = np.array([
            ["a", "a"],
            ["b", "b"],
            ["bb", "bb"],
            ["c", "c"],
            ["cc", "cc"],
            ["ccc", "ccc"]],
            dtype='|S')
        workspace.FeedBlob('d', strs)
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t']))
        workspace.RunOperatorOnce(core.CreateOperator(
            'UnpackSegments', ['l', 't'], ['newd']))
        assert((workspace.FetchBlob('newd') == workspace.FetchBlob('d')).all())

    def test_pad_minf(self):
        workspace.FeedBlob('l', np.array([1, 2, 3], dtype=np.int32))
        workspace.FeedBlob(
            'd',
            np.array([
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0]],
                dtype=np.float32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'PackSegments', ['l', 'd'], ['t'], pad_minf=True))
        workspace.RunOperatorOnce(core.CreateOperator(
            'Exp', ['t'], ['r']
        ))
        result = workspace.FetchBlob('t')
        assert(result[0, -1, 0] < -1000.0)

        # The whole point of padding with -inf is that when we exponentiate it
        # then it should be zero.
        exponentiated = workspace.FetchBlob('r')
        assert(exponentiated[0, -1, 0] == 0.0)

if __name__ == "__main__":
    import unittest
    unittest.main()
