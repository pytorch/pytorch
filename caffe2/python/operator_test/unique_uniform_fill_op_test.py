



from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest


class TestUniqueUniformFillOp(hu.HypothesisTestCase):
    @given(
        r=st.integers(1000, 10000),
        avoid=st.lists(
            st.integers(1, 1000),
            min_size=1,
            max_size=100,
            unique=True
        ),
        dtypes=st.sampled_from(
            [
                (np.int32, core.DataType.INT32),
                (np.int64, core.DataType.INT64)
            ]
        ),
        s=st.integers(10, 500),
        **hu.gcs_cpu_only
    )
    def test_unique_uniform_int_fill(self, r, avoid, dtypes, s, gc, dc):
        net = core.Net("net")
        workspace.FeedBlob("X", np.array([s], dtype=np.int64))
        workspace.FeedBlob("AVOID", np.array(avoid, dtype=dtypes[0]))
        net.UniqueUniformFill(
            ["X", "AVOID"], ["Y"],
            min=1,
            max=r,
            input_as_shape=True,
            dtype=dtypes[1]
        )
        workspace.RunNetOnce(net)
        y = workspace.FetchBlob("Y")
        self.assertEqual(s, len(y))
        self.assertEqual(s, len(set(y)))
        self.assertEqual(s, len(set(y) - set(avoid)))


if __name__ == "__main__":
    unittest.main()
