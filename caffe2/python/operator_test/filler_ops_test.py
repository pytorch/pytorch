from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import numpy as np


class TestFillerOperator(hu.HypothesisTestCase):

    @given(**hu.gcs)
    def test_gaussian_fill_op(self, gc, dc):
        op = core.CreateOperator(
            'GaussianFill',
            [],
            'out',
            shape=[17, 3, 3],  # sample odd dimensions
            mean=0.0,
            std=1.0,
        )

        for device_option in dc:
            op.device_option.CopyFrom(device_option)
            assert workspace.RunOperatorOnce(op), "GaussianFill op did not run "
            "successfully"

            blob_out = workspace.FetchBlob('out')
            assert np.count_nonzero(blob_out) > 0, "All generated elements are "
            "zeros. Is the random generator functioning correctly?"

    @given(**hu.gcs)
    def test_msra_fill_op(self, gc, dc):
        op = core.CreateOperator(
            'MSRAFill',
            [],
            'out',
            shape=[15, 5, 3],  # sample odd dimensions
        )
        for device_option in dc:
            op.device_option.CopyFrom(device_option)
            assert workspace.RunOperatorOnce(op), "MSRAFill op did not run "
            "successfully"

            blob_out = workspace.FetchBlob('out')
            assert np.count_nonzero(blob_out) > 0, "All generated elements are "
            "zeros. Is the random generator functioning correctly?"


if __name__ == "__main__":
    import unittest
    unittest.main()
