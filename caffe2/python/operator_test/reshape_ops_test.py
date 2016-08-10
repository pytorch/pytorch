from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestLengthsToShapeOps(TestCase):
    def test_lengths_to_shape_ops(self):
        workspace.FeedBlob('l', np.array([200, 200, 200], dtype=np.int64))
        workspace.RunOperatorOnce(core.CreateOperator(
            'LengthsToShape', ['l'], ['s']))
        workspace.FeedBlob('res', np.array([3, 200]))
        assert ((workspace.FetchBlob('s') == workspace.FetchBlob('res')).all())

    def test_reshape_ops(self):
        workspace.FeedBlob('res', np.array([[0, 0, 0, 0]], dtype=np.float32))
        workspace.FeedBlob('shape', np.array([1, 4], dtype=np.int32))
        workspace.FeedBlob('input', np.zeros((2, 2), dtype=np.float32))
        workspace.RunOperatorOnce(core.CreateOperator(
            'Reshape', ['input', 'shape'], ['output']))
        assert ((workspace.FetchBlob('output') ==
                 workspace.FetchBlob('res')).all())
