# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestCounterOps(TestCase):

    def test_stats_ops(self):
        # The global StatRegistry isn't reset when the workspace is reset,
        #   so there may be existing stats from a previous test
        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['prev_k', 'prev_v', 'prev_ts']))
        previous_keys = workspace.FetchBlob('prev_k')
        existing = len(previous_keys)

        prefix = '/'.join([__name__, 'TestCounterOps', 'test_stats_ops'])
        keys = [
            (prefix + '/key1').encode('ascii'),
            (prefix + '/key2').encode('ascii')
        ]
        values = [34, 45]
        workspace.FeedBlob('k', np.array(keys, dtype=str))
        workspace.FeedBlob('v', np.array(values, dtype=np.int64))
        for _ in range(2):
            workspace.RunOperatorOnce(core.CreateOperator(
                'StatRegistryUpdate', ['k', 'v'], []))
        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k2', 'v2', 't2']))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryCreate', [], ['reg']))
        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryUpdate', ['k2', 'v2', 'reg'], []))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', ['reg'], ['k3', 'v3', 't3']))

        k3 = workspace.FetchBlob('k3')
        v3 = workspace.FetchBlob('v3')
        t3 = workspace.FetchBlob('t3')

        self.assertEqual(len(k3) - existing, 2)
        self.assertEqual(len(v3), len(k3))
        self.assertEqual(len(t3), len(k3))
        for key in keys:
            self.assertIn(key, k3)
