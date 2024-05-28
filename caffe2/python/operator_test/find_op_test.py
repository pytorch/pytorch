




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np


class TestFindOperator(serial.SerializedTestCase):

    @given(n=st.sampled_from([1, 4, 8, 31, 79, 150]),
        idxsize=st.sampled_from([2, 4, 8, 1000, 5000]),
        **hu.gcs)
    @settings(deadline=10000)
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
