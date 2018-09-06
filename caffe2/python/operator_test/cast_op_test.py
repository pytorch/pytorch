from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
import numpy as np


class TestCastOp(hu.HypothesisTestCase):

    @given(**hu.gcs)
    def test_cast_int_float(self, gc, dc):
        data = np.random.rand(5, 5).astype(np.int32)
        # from int to float
        op = core.CreateOperator('Cast', 'data', 'data_cast', to=1, from_type=2)
        self.assertDeviceChecks(dc, op, [data], [0])
        # This is actually 0
        self.assertGradientChecks(gc, op, [data], 0, [0])

    @given(**hu.gcs)
    def test_cast_int_float_empty(self, gc, dc):
        data = np.random.rand(0).astype(np.int32)
        # from int to float
        op = core.CreateOperator('Cast', 'data', 'data_cast', to=1, from_type=2)
        self.assertDeviceChecks(dc, op, [data], [0])
        # This is actually 0
        self.assertGradientChecks(gc, op, [data], 0, [0])
