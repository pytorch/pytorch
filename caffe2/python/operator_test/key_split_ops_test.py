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

import hypothesis.strategies as st

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import numpy as np


class TestKeySplitOps(hu.HypothesisTestCase):
    @given(
        X=hu.arrays(
            dims=[1000],
            dtype=np.int64,
            elements=st.integers(min_value=0, max_value=100)
        ),
        **hu.gcs_cpu_only
    )
    def test_key_split_op(self, X, gc, dc):
        categorical_limit = max(X) + 1
        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X)
        output_blobs = ['Y_%d' % i for i in range(categorical_limit)]
        op = core.CreateOperator(
            'KeySplit', ['X'],
            output_blobs,
            categorical_limit=categorical_limit
        )
        workspace.RunOperatorOnce(op)
        output_vecs = [
            workspace.blobs[output_blobs[i]] for i in range(categorical_limit)
        ]
        expected_output_vecs = [[] for _ in range(categorical_limit)]
        for i, x in enumerate(X):
            expected_output_vecs[x].append(i)
        for i in range(categorical_limit):
            np.testing.assert_array_equal(
                output_vecs[i],
                np.array(expected_output_vecs[i], dtype=np.int32)
            )
