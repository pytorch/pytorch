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
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestIndexHashOps(hu.HypothesisTestCase):
    @given(
        indices=st.sampled_from([
            np.int32, np.int64
        ]).flatmap(lambda dtype: hu.tensor(min_dim=1, max_dim=1, dtype=dtype)),
        seed=st.integers(min_value=0, max_value=10),
        modulo=st.integers(min_value=100000, max_value=200000),
        **hu.gcs_cpu_only
    )
    def test_index_hash_ops(self, indices, seed, modulo, gc, dc):
        op = core.CreateOperator("IndexHash",
                                 ["indices"], ["hashed_indices"],
                                 seed=seed, modulo=modulo)

        def index_hash(indices):
            dtype = np.array(indices).dtype
            assert dtype == np.int32 or dtype == np.int64
            hashed_indices = []
            for index in indices:
                hashed = dtype.type(0xDEADBEEF * seed)
                indices_bytes = np.array([index], dtype).view(np.int8)
                for b in indices_bytes:
                    hashed = dtype.type(hashed * 65537 + b)
                hashed = (modulo + hashed % modulo) % modulo
                hashed_indices.append(hashed)
            return [hashed_indices]

        self.assertDeviceChecks(dc, op, [indices], [0])
        self.assertReferenceChecks(gc, op, [indices], index_hash)
