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

from caffe2.python import core
import hypothesis.strategies as st
from hypothesis import given


import caffe2.python.hypothesis_test_util as hu

import numpy as np


class TestFindOperator(hu.HypothesisTestCase):

    @given(n=st.sampled_from([1, 4, 8, 31, 79, 150]),
           idxsize=st.sampled_from([2, 4, 8, 1000, 5000]),
           **hu.gcs)
    def test_find(self, n, idxsize, gc, dc):
        maxval = 10

        def findop(idx, X):
            res = []
            for j in list(X.flatten()):
                i = np.where(idx == j)[0]
                if len(i) == 0:
                    res.append(-1)
                else:
                    res.append(i[-1])

            print("Idx: {} X: {}".format(idx, X))
            print("Res: {}".format(res))
            return [np.array(res).astype(np.int32)]

        X = (np.random.rand(n) * maxval).astype(np.int32)
        idx = (np.random.rand(idxsize) * maxval).astype(np.int32)

        op = core.CreateOperator(
            "Find",
            ["idx", "X"],
            ["y"],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[idx, X],
            reference=findop,
        )
