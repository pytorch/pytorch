from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


class TestSqueezeOp(TestCase):
    def test_squeeze_all(self):
        # Testing that squeezing without dims works.
        # With dims is covered in hypothesis_test
        data = np.array([[[1]]], dtype=np.int32)
        workspace.FeedBlob('data', data)
        workspace.RunOperatorOnce(core.CreateOperator(
            'Squeeze', ['data'], ['squeezed']))
        result = workspace.FetchBlob('squeezed')
        assert(np.array_equal(result, 1))
