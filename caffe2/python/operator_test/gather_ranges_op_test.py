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

if __name__ == "__main__":
    import unittest
    unittest.main()
