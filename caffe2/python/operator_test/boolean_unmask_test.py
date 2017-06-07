from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestUnmaskOp(hu.HypothesisTestCase):
    @given(N=st.integers(min_value=2, max_value=20),
           dtype=st.sampled_from([
               np.bool_,
               np.int8,
               np.int16,
               np.int32,
               np.int64,
               np.uint8,
               np.uint16,
               np.float16,
               np.float32,
               np.float64]))
    def test(self, N, dtype):
        M = np.random.randint(1, N)
        all_value = np.random.rand(N).astype(dtype)
        split = sorted(np.random.randint(1, N, size=(M,)))
        indices = np.array(list(range(N)))
        random.shuffle(indices)
        pieces = np.split(indices, split)
        masks_and_values_name = []
        for i, piece in enumerate(pieces):
            mask = np.zeros(N, dtype=np.bool_)
            mask[piece] = True
            values = all_value[piece]
            mask_name = "mask%d" % i
            value_name = "value%d" % i
            workspace.FeedBlob(mask_name, mask)
            workspace.FeedBlob(value_name, values)
            masks_and_values_name.append(mask_name)
            masks_and_values_name.append(value_name)
        net = core.Net('net')
        net.BooleanUnmask(masks_and_values_name, ["output"])
        workspace.RunNetOnce(net)
        output = workspace.FetchBlob('output')
        self.assertAlmostEqual(
            output.all(),
            all_value.all(),
            delta=1e-4
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
