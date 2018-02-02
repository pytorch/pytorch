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

import hypothesis
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestSparseNormalize(hu.HypothesisTestCase):

    @staticmethod
    def ref_normalize(param_in, use_max_norm, norm):
        param_norm = np.linalg.norm(param_in) + 1e-12
        if (use_max_norm and param_norm > norm) or not use_max_norm:
            param_in = param_in * norm / param_norm
        return param_in

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=2, min_dim=2, max_dim=2),
           use_max_norm=st.booleans(),
           norm=st.floats(min_value=1.0, max_value=4.0),
           data_strategy=st.data(),
           **hu.gcs_cpu_only)
    def test_sparse_normalize(self, inputs, use_max_norm, norm,
                              data_strategy, gc, dc):
        param, grad = inputs
        param += 0.02 * np.sign(param)
        param[param == 0.0] += 0.02

        # Create an indexing array containing values that are lists of indices,
        # which index into grad
        indices = data_strategy.draw(
            hu.tensor(dtype=np.int64, min_dim=1, max_dim=1,
                      elements=st.sampled_from(np.arange(grad.shape[0]))),
        )
        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # For now, the indices must be unique
        hypothesis.assume(np.array_equal(np.unique(indices.flatten()),
                                         np.sort(indices.flatten())))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseNormalize",
            ["param", "indices", "grad"],
            ["param"],
            use_max_norm=use_max_norm,
            norm=norm,
        )

        def ref_sparse_normalize(param, indices, grad):
            param_out = np.copy(param)
            for _, index in enumerate(indices):
                param_out[index] = self.ref_normalize(
                    param[index],
                    use_max_norm,
                    norm,
                )
            return (param_out,)

        # self.assertDeviceChecks(dc, op, [param, indices, grad], [0])
        self.assertReferenceChecks(
            gc, op, [param, indices, grad],
            ref_sparse_normalize
        )
