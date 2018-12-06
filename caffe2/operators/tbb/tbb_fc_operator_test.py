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
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestTBBFcOperator(hu.HypothesisTestCase):
    def _run_test(self, n, m, k, multi_dim, gc, dc):
        dtype = np.float32
        X = np.random.rand(m, k).astype(dtype) - 0.5
        if multi_dim:
            W = np.random.rand(n, k, 1, 1).astype(dtype) - 0.5
        else:
            W = np.random.rand(n, k).astype(dtype) - 0.5
        b = np.random.rand(n).astype(dtype) - 0.5

        workspace.FeedBlob('X', X)
        workspace.FeedBlob('W', W)
        workspace.FeedBlob('b', b)

        def fc_op(X, W, b):
            return [np.dot(X, W.reshape(n, k).transpose()) + b.reshape(n)]

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            'Y',
            engine='TBB',
        )

        # Check against numpy reference
        net = core.Net('test_net')
        net.Proto().type = 'parallel'
        net.Proto().num_workers = 1
        net.Proto().op.extend([op])

        workspace.RunNetOnce(net)
        reference_outputs = fc_op(X, W, b)

        for output_index in range(len(reference_outputs)):
            output_blob_name = op.output[output_index]
            output = workspace.FetchBlob(output_blob_name)
            np.testing.assert_allclose(
                output,
                reference_outputs[output_index],
                atol=1e-4,
                rtol=1e-4,
                err_msg='Output {} is not matching the reference'.
                format(output_blob_name)
            )

        op = core.CreateOperator(
            'FCGradient', ['X', 'W', 'dY'], ['dW', 'db', 'dX'],
            engine='TBB')
        op_ref = core.CreateOperator(
            'FCGradient', ['X', 'W', 'dY'], ['dW_ref', 'db_ref', 'dX_ref'])

        workspace.FeedBlob('dY', workspace.FetchBlob('Y'))
        del net.Proto().op[:]
        net.Proto().op.extend([op])

        ref_net = core.Net('ref_test_net')
        net.Proto().op.extend([op_ref])

        workspace.RunNetOnce(net)
        workspace.RunNetOnce(ref_net)

        for output_index in range(len(op.output)):
            output_blob_name = op.output[output_index]
            output = workspace.FetchBlob(output_blob_name)
            ref_output = workspace.FetchBlob('{}_ref'.format(output_blob_name))
            np.testing.assert_allclose(
                output,
                ref_output,
                atol=1e-4,
                rtol=1e-4,
                err_msg='Output {} is not matching the reference'.
                format(output_blob_name)
            )

    @settings(max_examples=1)
    @given(n=st.sampled_from([63]),
           m=st.sampled_from([63]),
           k=st.sampled_from([63]),
           multi_dim=st.sampled_from([False]),
           **hu.gcs_cpu_only)
    def test_fc(self, **kwargs):
        self._run_test(**kwargs)


if __name__ == "__main__":
    import unittest
    unittest.main()
