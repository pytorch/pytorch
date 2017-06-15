from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import hypothesis.strategies as st
import numpy as np
import random

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


class TestTopK(hu.HypothesisTestCase):

    @given(X=hu.tensor(), flatten_indices=st.booleans(), **mu.gcs)
    def test_top_k(self, X, flatten_indices, gc, dc):
        X = X.astype(dtype=np.float32)
        k = random.randint(1, X.shape[-1])
        output_list = ["Values", "Indices"]
        if flatten_indices:
            output_list.append("FlattenIndices")
        op = core.CreateOperator("TopK", ["X"], output_list,
                                 k=k, device_option=gc)

        def top_k_ref(X):
            X_flat = X.reshape((-1, X.shape[-1]))
            indices_ref = np.ndarray(shape=X_flat.shape, dtype=np.int32)
            flatten_indices_ref = np.ndarray(shape=X_flat.shape, dtype=np.int32)
            values_ref = np.ndarray(shape=X_flat.shape, dtype=np.float32)
            global_idx = 0
            for i in range(X_flat.shape[0]):
                od = OrderedDict()
                for j in range(X_flat.shape[1]):
                    val = X_flat[i, j]
                    if val not in od:
                        od[val] = []
                    od[val].append((j, global_idx))
                    global_idx += 1
                j = 0
                for val, idxs in sorted(od.items(), reverse=True):
                    for (idx, flatten_idx) in idxs:
                        indices_ref[i, j] = idx
                        flatten_indices_ref[i, j] = flatten_idx
                        values_ref[i, j] = val
                        j = j + 1

            indices_ref = np.reshape(indices_ref, X.shape)
            flatten_indices_ref = np.reshape(flatten_indices_ref, X.shape)
            values_ref = np.reshape(values_ref, X.shape)

            indices_ref = indices_ref.take(list(range(k)), axis=-1)
            flatten_indices_ref = flatten_indices_ref.take(list(range(k)),
                                                           axis=-1)
            values_ref = values_ref.take(list(range(k)), axis=-1)

            flatten_indices_ref = flatten_indices_ref.flatten()

            if flatten_indices:
                return (values_ref, indices_ref, flatten_indices_ref)
            else:
                return (values_ref, indices_ref)

        self.assertReferenceChecks(hu.cpu_do, op, [X], top_k_ref)

    @given(X=hu.tensor(min_dim=2), **hu.gcs_cpu_only)
    def test_top_k_grad(self, X, gc, dc):
        X = X.astype(np.float32)
        k = random.randint(1, X.shape[-1])

        # this try to make sure adding stepsize (0.05)
        # will not change TopK selections at all
        # since dims max_value = 5 as defined in
        # caffe2/caffe2/python/hypothesis_test_util.py
        for i in range(X.shape[-1]):
            X[..., i] = i * 1.0 / X.shape[-1]

        op = core.CreateOperator(
            "TopK", ["X"], ["Values", "Indices"], k=k, device_option=gc
        )

        self.assertGradientChecks(gc, op, [X], 0, [0])
