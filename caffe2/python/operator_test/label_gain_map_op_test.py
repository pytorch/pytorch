from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestMapLookupOp(TestCase):

    def test_int_float(self):

        map_lengths = np.array([2, 3, 2], dtype=np.int32)
        map_keys = np.array([11, 12, 21, 22, 23, 31, 32], dtype=np.float32)
        map_values = np.array(
            [-11.0, -12.0, -21.0, -22.0, -23.0, -31.0, -32.0], dtype=np.float32)
        op = core.CreateOperator(
            "LabelGainMap",
            ["keys"],
            ["values"],
            map_lengths=map_lengths,
            map_keys=map_keys,
            map_values=map_values,
        )

        original_keys = np.array(
            [[11, 22, 31], [11.5, 22.5, 31.2]], dtype=np.float32)

        workspace.FeedBlob("keys", original_keys)
        workspace.RunOperatorOnce(op)
        np.testing.assert_almost_equal(
            workspace.FetchBlob("values"), -1 * original_keys, decimal=3)
