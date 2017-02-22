from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestCounterOps(TestCase):

    def test_stats_ops(self):
        workspace.FeedBlob('k', np.array(['k0', 'k1'], dtype=str))
        workspace.FeedBlob('v', np.array([34, 45], dtype=np.int64))
        for _ in range(2):
            workspace.RunOperatorOnce(core.CreateOperator(
                'StatRegistryUpdate', ['k', 'v'], []))
        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k2', 'v2', 'ts']))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryCreate', [], ['reg']))
        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryUpdate', ['k2', 'v2', 'reg'], []))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', ['reg'], ['k3', 'v3', 'ts']))
        self.assertTrue(len(workspace.FetchBlob('k3')) == 2)
        self.assertTrue(len(workspace.FetchBlob('v3')) == 2)
