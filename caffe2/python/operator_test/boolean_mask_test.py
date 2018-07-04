from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestBooleanMaskOp(hu.HypothesisTestCase):

    @given(x=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_boolean_mask(self, x, gc, dc):
        op = core.CreateOperator("BooleanMask",
                                 ["data", "mask"],
                                 "masked_data")
        mask = np.random.choice(a=[True, False], size=x.shape[0])

        def ref(x, mask):
            return (x[mask],)

        self.assertReferenceChecks(gc, op, [x, mask], ref)
        self.assertDeviceChecks(dc, op, [x, mask], [0])

    @given(x=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_boolean_mask_indices(self, x, gc, dc):
        op = core.CreateOperator("BooleanMask",
                                 ["data", "mask"],
                                 ["masked_data", "masked_indices"])
        mask = np.random.choice(a=[True, False], size=x.shape[0])

        def ref(x, mask):
            return (x[mask], np.where(mask)[0])

        self.assertReferenceChecks(gc, op, [x, mask], ref)
        self.assertDeviceChecks(dc, op, [x, mask], [0])

    @staticmethod
    def _dtype_conversion(x, dtype, gc, dc):
        """SequenceMask only supports fp16 with CUDA."""
        if dtype == np.float16:
            assume(gc.device_type == caffe2_pb2.CUDA)
            dc = [d for d in dc if d.device_type == caffe2_pb2.CUDA]
            x = x.astype(dtype)
        return x, dc

    @given(x=hu.tensor(min_dim=2,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_with_lengths(self, x, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        op = core.CreateOperator("SequenceMask",
                                 ["data", "lengths"],
                                 ["masked_data"],
                                 mode="sequence",
                                 axis=len(x.shape) - 1,
                                 fill_val=fill_val)
        elem_dim = x.shape[-1]
        leading_dim = 1
        for dim in x.shape[:-1]:
            leading_dim *= dim
        lengths = np.random.randint(0, elem_dim, [leading_dim])\
            .astype(np.int32)

        def ref(x, lengths):
            ref = np.reshape(x, [leading_dim, elem_dim])
            for i in range(leading_dim):
                for j in range(elem_dim):
                    if j >= lengths[i]:
                        ref[i, j] = fill_val
            return [ref.reshape(x.shape)]

        self.assertReferenceChecks(gc, op, [x, lengths], ref)
        self.assertDeviceChecks(dc, op, [x, lengths], [0])

    @given(x=hu.tensor(min_dim=2,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_with_window(self, x, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        radius = 2
        op = core.CreateOperator("SequenceMask",
                                 ["data", "centers"],
                                 ["masked_data"],
                                 mode="window",
                                 radius=radius,
                                 axis=len(x.shape) - 1,
                                 fill_val=fill_val)
        elem_dim = x.shape[-1]
        leading_dim = 1
        for dim in x.shape[:-1]:
            leading_dim *= dim
        centers = np.random.randint(0, elem_dim, [leading_dim])\
            .astype(np.int32)

        def ref(x, centers):
            ref = np.reshape(x, [leading_dim, elem_dim])
            for i in range(leading_dim):
                for j in range(elem_dim):
                    if j > centers[i] + radius or j < centers[i] - radius:
                        ref[i, j] = fill_val
            return [ref.reshape(x.shape)]

        self.assertReferenceChecks(gc, op, [x, centers], ref)
        self.assertDeviceChecks(dc, op, [x, centers], [0])

        threshold = 0.4 if dtype == np.float16 else 0.005
        self.assertGradientChecks(gc, op, [x, centers], 0, [0],
                                  threshold=threshold)

    @given(x=hu.tensor(min_dim=2,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           mode=st.sampled_from(['upper', 'lower', 'upperdiag', 'lowerdiag']),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_triangle(self, x, mode, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        op = core.CreateOperator("SequenceMask",
                                 ["data"],
                                 ["masked_data"],
                                 mode=mode,
                                 axis=len(x.shape) - 1,
                                 fill_val=fill_val)
        elem_dim = x.shape[-1]
        leading_dim = 1
        for dim in x.shape[:-1]:
            leading_dim *= dim

        if mode == 'upper':
            def compare(i, j):
                return j > i
        elif mode == 'lower':
            def compare(i, j):
                return j < i
        elif mode == 'upperdiag':
            def compare(i, j):
                return j >= i
        elif mode == 'lowerdiag':
            def compare(i, j):
                return j <= i

        def ref(x):
            ref = np.reshape(x, [leading_dim, elem_dim])
            for i in range(leading_dim):
                for j in range(elem_dim):
                    if compare(i, j):
                        ref[i, j] = fill_val
            return [ref.reshape(x.shape)]

        self.assertReferenceChecks(gc, op, [x], ref)
        self.assertDeviceChecks(dc, op, [x], [0])

        threshold = 0.4 if dtype == np.float16 else 0.005
        stepsize = 0.1 if dtype == np.float16 else 0.05
        self.assertGradientChecks(gc, op, [x], 0, [0],
                                  threshold=threshold, stepsize=stepsize)

    @given(x=hu.tensor(min_dim=2,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_batching_lengths(self, x, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        # choose _different_ batch and axis dimensions, w/ axis != 0.
        axis = 0
        batch = 0
        while axis == 0 or axis < batch:
            inds = np.arange(len(x.shape))
            np.random.shuffle(inds)
            batch = inds[0]
            axis = inds[1]
        op = core.CreateOperator("SequenceMask",
                                 ["data", "lengths"],
                                 ["masked_data"],
                                 mode='sequence',
                                 axis=axis,
                                 fill_val=fill_val,
                                 batch=batch)

        before = int(np.prod(x.shape[:batch + 1]))
        between = int(np.prod(x.shape[batch + 1:axis]))
        after = int(np.prod(x.shape[axis:]))

        lengths = np.random.randint(0, after, [between])\
            .astype(np.int32)

        def ref(z, l):
            w = np.reshape(z, [before, between, after])

            for b in range(before):
                r = w[b, :, :]
                for i in range(between):
                    for j in range(after):
                        if j >= l[i]:
                            r[i, j] = fill_val
            return [w.reshape(z.shape)]

        self.assertReferenceChecks(gc, op, [x, lengths], ref)
        self.assertDeviceChecks(dc, op, [x, lengths], [0])

        threshold = 0.4 if dtype == np.float16 else 0.005
        self.assertGradientChecks(gc, op, [x, lengths], 0, [0],
                                  threshold=threshold)

    @given(x=hu.tensor(min_dim=4,
                       max_dim=4,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_batching_window(self, x, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        radius = 1
        # choose _different_ batch and axis dimensions, w/ axis != 0.
        axis = 0
        batch = 0
        while axis == 0 or axis < batch:
            inds = np.arange(len(x.shape))
            np.random.shuffle(inds)
            batch = inds[0]
            axis = inds[1]
        op = core.CreateOperator("SequenceMask",
                                 ["data", "centers"],
                                 ["masked_data"],
                                 mode='window',
                                 radius=radius,
                                 axis=axis,
                                 fill_val=fill_val,
                                 batch=batch)

        before = int(np.prod(x.shape[:batch + 1]))
        between = int(np.prod(x.shape[batch + 1:axis]))
        after = int(np.prod(x.shape[axis:]))

        centers = np.random.randint(0, after, [between])\
            .astype(np.int32)

        def ref(z, c):
            w = np.reshape(z, [before, between, after])

            for b in range(before):
                r = w[b, :, :]
                for i in range(between):
                    for j in range(after):
                        if j > c[i] + radius or j < c[i] - radius:
                            r[i, j] = fill_val
            return [w.reshape(z.shape)]

        self.assertReferenceChecks(gc, op, [x, centers], ref)
        self.assertDeviceChecks(dc, op, [x, centers], [0])

        threshold = 0.4 if dtype == np.float16 else 0.005
        self.assertGradientChecks(gc, op, [x, centers], 0, [0],
                                  threshold=threshold)

    @given(x=hu.tensor(min_dim=3,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           mode=st.sampled_from(['upper', 'lower', 'upperdiag', 'lowerdiag']),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_batching_triangle(self, x, mode, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        # choose _different_ batch and axis dimensions, w/ axis != 0.
        axis = 0
        batch = 0
        while axis == 0 or axis < batch:
            inds = np.arange(len(x.shape))
            np.random.shuffle(inds)
            batch = inds[0]
            axis = inds[1]
        op = core.CreateOperator("SequenceMask",
                                 ["data"],
                                 ["masked_data"],
                                 mode=mode,
                                 axis=axis,
                                 fill_val=fill_val,
                                 batch=batch)

        if mode == 'upper':
            def compare(i, j):
                return j > i
        elif mode == 'lower':
            def compare(i, j):
                return j < i
        elif mode == 'upperdiag':
            def compare(i, j):
                return j >= i
        elif mode == 'lowerdiag':
            def compare(i, j):
                return j <= i

        def ref(z):
            before = int(np.prod(z.shape[:batch + 1]))
            between = int(np.prod(z.shape[batch + 1:axis]))
            after = int(np.prod(z.shape[axis:]))

            w = np.reshape(z, [before, between, after])

            for b in range(before):
                r = w[b, :, :]
                for i in range(between):
                    for j in range(after):
                        if compare(i, j):
                            r[i, j] = fill_val
            return [w.reshape(z.shape)]

        self.assertReferenceChecks(gc, op, [x], ref)
        self.assertDeviceChecks(dc, op, [x], [0])

        threshold = 0.4 if dtype == np.float16 else 0.005
        stepsize = 0.1 if dtype == np.float16 else 0.05
        self.assertGradientChecks(gc, op, [x], 0, [0],
                                  threshold=threshold, stepsize=stepsize)

    @given(x=hu.tensor(min_dim=3,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_sequence_mask_repeated(self, x, dtype, gc, dc):
        x, dc = self._dtype_conversion(x, dtype, gc, dc)
        # finite fill value needed for gradient check
        fill_val = 1e-3 if dtype == np.float16 else 1e-9
        op = core.CreateOperator("SequenceMask",
                                 ["data", "lengths"],
                                 ["masked_data"],
                                 mode="sequence",
                                 axis=len(x.shape) - 2,
                                 repeat_from_axis=-1,
                                 fill_val=fill_val)

        elem_dim = x.shape[-2]
        leading_dim = 1
        for dim in x.shape[:-2]:
            leading_dim *= dim
        lengths = np.random.randint(0, elem_dim, [leading_dim])\
            .astype(np.int32)

        def ref(x, lengths):
            ref = np.reshape(x, [leading_dim, elem_dim, -1])
            for i in range(leading_dim):
                for j in range(elem_dim):
                    if j >= lengths[i]:
                        ref[i, j, :] = fill_val
            return [ref.reshape(x.shape)]

        self.assertReferenceChecks(gc, op, [x, lengths], ref)
        self.assertDeviceChecks(dc, op, [x, lengths], [0])
