from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


lengths = [[0], [1, 2], [1, 0, 2, 0]]
features1 = [[],
             [1, 2, 2],
             [[1, 1], [2, 2], [2, 2]]
             ]
features2 = [[],
             [2, 4, 4],
             [[2, 2], [4, 4], [4, 4]]
             ]

lengths_exp = [[1], [1, 2], [1, 1, 2, 1]]
features1_exp = [[0],
                 [1, 2, 2],
                 [[1, 1], [0, 0], [2, 2], [2, 2], [0, 0]]]
features2_exp = [[0],
                 [2, 4, 4],
                 [[2, 2], [0, 0], [4, 4], [4, 4], [0, 0]]]


class TestEmptySampleOps(TestCase):
    def test_emptysample(self):
        for i in range(0, 3):
            PadEmptyTest = core.CreateOperator(
                'PadEmptySamples',
                ['lengths', 'features1', 'features2'],
                ['out_lengths', 'out_features1', 'out_features2'],
            )
            workspace.FeedBlob(
                'lengths',
                np.array(lengths[i], dtype=np.int32))
            workspace.FeedBlob(
                'features1',
                np.array(features1[i], dtype=np.int64))
            workspace.FeedBlob(
                'features2',
                np.array(features2[i], dtype=np.int64))
            workspace.RunOperatorOnce(PadEmptyTest)
            np.testing.assert_allclose(
                lengths_exp[i],
                workspace.FetchBlob('out_lengths'),
                atol=1e-4, rtol=1e-4, err_msg='Mismatch in lengths')
            np.testing.assert_allclose(
                features1_exp[i],
                workspace.FetchBlob('out_features1'),
                atol=1e-4, rtol=1e-4, err_msg='Mismatch in features1')
            np.testing.assert_allclose(
                features2_exp[i],
                workspace.FetchBlob('out_features2'),
                atol=1e-4, rtol=1e-4, err_msg='Mismatch in features2')

if __name__ == "__main__":
    import unittest
    unittest.main()
