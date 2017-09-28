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
from caffe2.proto import caffe2_pb2
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


def _one_hots():
    index_size = st.integers(min_value=1, max_value=5)
    lengths = st.lists(
        elements=st.integers(min_value=0, max_value=5))
    return st.tuples(index_size, lengths).flatmap(
        lambda x: st.tuples(
            st.just(x[0]),
            st.just(x[1]),
            st.lists(
                elements=st.integers(min_value=0, max_value=x[0] - 1),
                min_size=sum(x[1]),
                max_size=sum(x[1]))))


class TestOneHotOps(hu.HypothesisTestCase):
    @given(
        x=hu.tensor(
            min_dim=2, max_dim=2, dtype=np.int32,
            elements=st.integers(min_value=0, max_value=10)),
        **hu.gcs_cpu_only)
    def test_batch_one_hot(self, x, gc, dc):
        d = x.shape[1]
        lens = []
        vals = []
        for i in range(0, d):
            val = np.unique(x[:, i])
            vals.extend(val)
            lens.append(len(val))
        lens = np.array(lens, dtype=np.int32)
        vals = np.array(vals, dtype=np.int32)

        def ref(x, lens, vals):
            output_dim = vals.size
            ret = np.zeros((x.shape[0], output_dim)).astype(x.dtype)
            p = 0
            for i, l in enumerate(lens):
                for j in range(0, l):
                    v = vals[p + j]
                    ret[x[:, i] == v, p + j] = 1
                p += lens[i]
            return (ret, )

        op = core.CreateOperator('BatchOneHot', ["X", "LENS", "VALS"], ["Y"])
        self.assertReferenceChecks(gc, op, [x, lens, vals], ref)

    @given(
        hot_indices=hu.tensor(
            min_dim=1, max_dim=1, dtype=np.int64,
            elements=st.integers(min_value=0, max_value=42)),
        end_padding=st.integers(min_value=0, max_value=2),
        **hu.gcs)
    def test_one_hot(self, hot_indices, end_padding, gc, dc):

        def one_hot_ref(hot_indices, size):
            out = np.zeros([len(hot_indices), size], dtype=float)
            x = enumerate(hot_indices)
            for i, x in enumerate(hot_indices):
                out[i, x] = 1.
            return (out, )

        size = np.array(max(hot_indices) + end_padding + 1, dtype=np.int64)
        if size == 0:
            size = 1
        op = core.CreateOperator('OneHot', ['hot_indices', 'size'], ['output'])
        self.assertReferenceChecks(
            gc,
            op,
            [hot_indices, size],
            one_hot_ref,
            input_device_options={'size': core.DeviceOption(caffe2_pb2.CPU)})

    @given(hot_indices=_one_hots())
    def test_segment_one_hot(self, hot_indices):
        index_size, lengths, indices = hot_indices

        index_size = np.array(index_size, dtype=np.int64)
        lengths = np.array(lengths, dtype=np.int32)
        indices = np.array(indices, dtype=np.int64)

        def segment_one_hot_ref(lengths, hot_indices, size):
            offset = 0
            out = np.zeros([len(lengths), size], dtype=float)
            for i, length in enumerate(lengths):
                for idx in hot_indices[offset:offset + length]:
                    out[i, idx] = 1.
                offset += length
            return (out, )

        op = core.CreateOperator(
            'SegmentOneHot',
            ['lengths', 'hot_indices', 'size'],
            ['output'])
        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [lengths, indices, index_size],
            segment_one_hot_ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
