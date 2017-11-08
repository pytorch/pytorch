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
from hypothesis import strategies as st

import caffe2.python.hypothesis_test_util as hu
import numpy as np


def batched_boarders_and_data(
        data_min_size=5, data_max_size=10,
        examples_min_number=1, examples_max_number=4,
        example_min_size=1, example_max_size=3,
        dtype=np.float32, elements=None):
    dims_ = st.tuples(
        st.integers(min_value=data_min_size,
                    max_value=data_max_size),
        st.integers(min_value=examples_min_number,
                    max_value=examples_max_number),
        st.integers(min_value=example_min_size,
                    max_value=example_max_size),
    )
    return dims_.flatmap(
        lambda dims: st.tuples(
            hu.arrays(
                [dims[1], dims[2], 2], dtype=np.int32,
                elements=st.integers(min_value=0, max_value=dims[0])
            ),
            hu.arrays([dims[0]], dtype, elements)
    ))


@st.composite
def _tensor_splits(draw):
    lengths = draw(st.lists(st.integers(1, 5), min_size=1, max_size=10))
    batch_size = draw(st.integers(1, 5))
    element_pairs = [
        (batch, r) for batch in range(batch_size) for r in range(len(lengths))
    ]
    perm = draw(st.permutations(element_pairs))
    perm = perm[:-1]  # skip one range
    ranges = [[(0, 0)] * len(lengths) for _ in range(batch_size)]
    offset = 0
    for pair in perm:
        ranges[pair[0]][pair[1]] = (offset, lengths[pair[1]])
        offset += lengths[pair[1]]

    data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=offset,
        max_size=offset
    ))
    return (
        np.array(data).astype(np.float32), np.array(ranges), np.array(lengths)
    )


def gather_ranges(data, ranges):
    lengths = []
    output = []
    for example_ranges in ranges:
        length = 0
        for range in example_ranges:
            assert len(range) == 2
            output.extend(data[range[0]:range[0] + range[1]])
            length += range[1]
        lengths.append(length)
    return output, lengths


def gather_ranges_to_dense(data, ranges, lengths):
    outputs = []
    assert len(ranges)
    batch_size = len(ranges)
    assert len(ranges[0])
    num_ranges = len(ranges[0])
    assert ranges.shape[2] == 2
    for i in range(num_ranges):
        out = []
        for j in range(batch_size):
            start, length = ranges[j][i]
            if not length:
                out.append([0] * lengths[i])
            else:
                assert length == lengths[i]
                out.append(data[start:start + length])
        outputs.append(np.array(out))
    return outputs


class TestGatherRanges(hu.HypothesisTestCase):
    @given(boarders_and_data=batched_boarders_and_data(), **hu.gcs_cpu_only)
    def test_gather_ranges(self, boarders_and_data, gc, dc):
        boarders, data = boarders_and_data

        def boarders_to_range(boarders):
            assert len(boarders) == 2
            boarders = sorted(boarders)
            return [boarders[0], boarders[1] - boarders[0]]

        ranges = np.apply_along_axis(boarders_to_range, 2, boarders)

        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator("GatherRanges",
                                   ["data", "ranges"],
                                   ["output", "lengths"]),
            inputs=[data, ranges],
            reference=gather_ranges,
        )

    @given(tensor_splits=_tensor_splits(), **hu.gcs_cpu_only)
    def test_gather_ranges_split(self, tensor_splits, gc, dc):
        data, ranges, lengths = tensor_splits

        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator(
                "GatherRangesToDense",
                ['data', 'ranges'],
                ['X_{}'.format(i) for i in range(len(lengths))],
                lengths=lengths
            ),
            inputs=[data, ranges, lengths],
            reference=gather_ranges_to_dense
        )

if __name__ == "__main__":
    import unittest
    unittest.main()
