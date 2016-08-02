from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestGatherOps(TestCase):
    def test_gather_ops(self):
        data = np.array(["world", "hello", "!"], dtype='|S')
        ind = np.array([1, 0, 2], dtype=np.int32)
        workspace.FeedBlob('data', data)
        workspace.FeedBlob('ind', ind)
        workspace.RunOperatorOnce(core.CreateOperator(
            'Gather', ['data', 'ind'], ['word']))
        outdata = np.array(["hello", "world", "!"], dtype='|S')
        assert((workspace.FetchBlob('word') == outdata).all())
