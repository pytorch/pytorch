from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import unittest
import random


class TestSpecializedSegmentOps(hu.HypothesisTestCase):
    @given(nseg=st.integers(1, 20),
           fptype=st.sampled_from([np.float16, np.float32]),
           fp16asint=st.booleans(),
           bs=st.sampled_from([8, 17, 32, 64, 85, 96, 128, 163]), **hu.gcs)
    def test_sparse_lengths_sum_cpu_fp32(
            self, nseg, fptype, fp16asint, bs, gc, dc):

        tblsize = 300
        if fptype == np.float32:
            X = np.random.rand(tblsize, bs).astype(np.float32)
            atol = 1e-5
        else:
            if fp16asint:
                X = (10.0 * np.random.rand(tblsize, bs)
                     ).round().astype(np.float16)
                atol = 1e-3
            else:
                X = np.random.rand(tblsize, bs).astype(np.float16)
                atol = 1e-1

        ind = random.sample(range(1, 30), nseg)
        Y = np.random.randint(0, tblsize, size=sum(ind)).astype(np.int64)
        Z = np.asarray(ind).astype(np.int32)
        op = core.CreateOperator("SparseLengthsSum", ["X", "Y", "Z"], "out")

        def sparse_lengths_sum_ref(X, Y, Z):
            rptr = np.cumsum(np.insert(Z, [0], [0]))
            out = np.zeros((len(Z), bs))
            for i in range(0, len(rptr[0:-1])):
                out[i] = X[Y[rptr[i]:rptr[i + 1]]].sum(axis=0)
            return [out]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y, Z],
            reference=sparse_lengths_sum_ref,
            atol=atol,
        )


if __name__ == "__main__":
    unittest.main()
