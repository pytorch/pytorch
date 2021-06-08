#!/usr/bin/env python3

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
from hypothesis import given


class TestAsyncNetBarrierOp(hu.HypothesisTestCase):
    @given(
        n=st.integers(1, 5),
        shape=st.lists(st.integers(0, 5), min_size=1, max_size=3),
        **hu.gcs
    )
    def test_async_net_barrier_op(self, n, shape, dc, gc):
        test_inputs = [(100 * np.random.random(shape)).astype(np.float32) for _ in range(n)]
        test_input_blobs = ["x_{}".format(i) for i in range(n)]

        barrier_op = core.CreateOperator(
            "AsyncNetBarrier",
            test_input_blobs,
            test_input_blobs,
            device_option=gc,
        )

        def reference_func(*args):
            self.assertEquals(len(args), n)
            return args

        self.assertReferenceChecks(gc, barrier_op, test_inputs, reference_func)
