from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core, dyndep
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
import numpy as np

dyndep.InitOpsLibrary("@/caffe2:csrc")


class TestATen2Op(hu.HypothesisTestCase):

    @given(**hu.gcs)
    def test_aten2_op(self, gc, dc):
        self_ = np.random.rand(5, 5).astype(np.float)
        non_blocking = np.array([False])
        # from int to float
        op = core.CreateOperator('ATen2', ['self_', 'non_blocking'], ['out'],
                                 schema='aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor')
        self.assertDeviceChecks(dc, op, [self_, non_blocking], [0])
