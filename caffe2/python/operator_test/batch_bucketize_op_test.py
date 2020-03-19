from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st


class TestBatchBucketize(serial.SerializedTestCase):
    @serial.given(**hu.gcs_cpu_only)
    def test_batch_bucketize_example(self, gc, dc):
        op = core.CreateOperator('BatchBucketize',
                                 ["FEATURE", "INDICES", "BOUNDARIES", "LENGTHS"],
                                 ["O"])
        float_feature = np.array([[1.42, 2.07, 3.19, 0.55, 4.32],
                                  [4.57, 2.30, 0.84, 4.48, 3.09],
                                  [0.89, 0.26, 2.41, 0.47, 1.05],
                                  [0.03, 2.97, 2.43, 4.36, 3.11],
                                  [2.74, 5.77, 0.90, 2.63, 0.38]], dtype=np.float32)
        indices = np.array([0, 1, 4], dtype=np.int32)
        lengths = np.array([2, 3, 1], dtype=np.int32)
        boundaries = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 2.5], dtype=np.float32)

        def ref(float_feature, indices, boundaries, lengths):
            output = np.array([[2, 1, 1],
                               [2, 1, 1],
                               [1, 0, 0],
                               [0, 2, 1],
                               [2, 3, 0]], dtype=np.int32)
            return (output,)

        self.assertReferenceChecks(gc, op,
                                   [float_feature, indices, boundaries, lengths],
                                   ref)

    @given(
        x=hu.tensor(
            min_dim=2, max_dim=2, dtype=np.float32,
            elements=st.floats(min_value=0, max_value=5),
            min_value=5),
        seed=st.integers(min_value=2, max_value=1000),
        **hu.gcs_cpu_only)
    def test_batch_bucketize(self, x, seed, gc, dc):
        op = core.CreateOperator('BatchBucketize',
                                 ["FEATURE", "INDICES", "BOUNDARIES", "LENGTHS"],
                                 ['O'])
        np.random.seed(seed)
        d = x.shape[1]
        lens = np.random.randint(low=1, high=3, size=d - 3)
        indices = np.random.choice(range(d), d - 3, replace=False)
        indices.sort()
        boundaries = []
        for i in range(d - 3):
            # add [0, 0] as duplicated boundary for duplicated bucketization
            if lens[i] > 2:
                cur_boundary = np.append(
                    np.random.randn(lens[i] - 2) * 5, [0, 0])
            else:
                cur_boundary = np.random.randn(lens[i]) * 5
            cur_boundary.sort()
            boundaries += cur_boundary.tolist()

        lens = np.array(lens, dtype=np.int32)
        boundaries = np.array(boundaries, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

        def ref(x, indices, boundaries, lens):
            output_dim = indices.shape[0]
            ret = np.zeros((x.shape[0], output_dim)).astype(np.int32)
            boundary_offset = 0
            for i, l in enumerate(indices):
                temp_bound = boundaries[boundary_offset : lens[i] + boundary_offset]
                for j in range(x.shape[0]):
                    for k, bound_val in enumerate(temp_bound):
                        if k == len(temp_bound) - 1 and x[j, l] > bound_val:
                            ret[j, i] = k + 1
                        elif x[j, l] > bound_val:
                            continue
                        else:
                            ret[j, i] = k
                            break
                boundary_offset += lens[i]
            return (ret,)

        self.assertReferenceChecks(gc, op, [x, indices, boundaries, lens], ref)
