from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import numpy as np

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu


class TestFlexibleTopK(hu.HypothesisTestCase):
    def flexible_top_k_ref(self, X, k):
        X_flat = X.reshape((-1, X.shape[-1]))
        indices_ref = np.ndarray(shape=sum(k), dtype=np.int32)
        values_ref = np.ndarray(shape=sum(k), dtype=np.float32)
        offset = 0
        for i in range(X_flat.shape[0]):
            od = OrderedDict()
            for j in range(X_flat.shape[1]):
                val = X_flat[i, j]
                if val not in od:
                    od[val] = []
                od[val].append(j)
            k_ = 0
            for val, idxs in sorted(od.items(), reverse=True):
                for idx in idxs:
                    indices_ref[offset + k_] = idx
                    values_ref[offset + k_] = val
                    k_ += 1
                    if k_ >= k[i]:
                        break
                if k_ >= k[i]:
                    break
            offset += k[i]

        return (values_ref, indices_ref)

    @given(X=hu.tensor(min_dim=2), **hu.gcs_cpu_only)
    def test_flexible_top_k(self, X, gc, dc):
        X = X.astype(dtype=np.float32)
        k_shape = (int(X.size / X.shape[-1]), )
        k = np.random.randint(1, high=X.shape[-1] + 1, size=k_shape)

        output_list = ["Values", "Indices"]
        op = core.CreateOperator("FlexibleTopK", ["X", "k"], output_list,
                                 device_option=gc)

        def bind_ref(X_loc, k):
            ret = self.flexible_top_k_ref(X_loc, k)
            return ret

        self.assertReferenceChecks(gc, op, [X, k], bind_ref)

    @given(X=hu.tensor(min_dim=2), **hu.gcs_cpu_only)
    def test_flexible_top_k_grad(self, X, gc, dc):
        X = X.astype(np.float32)
        k_shape = (int(X.size / X.shape[-1]), )
        k = np.random.randint(1, high=X.shape[-1] + 1, size=k_shape)

        # this try to make sure adding stepsize (0.05)
        # will not change TopK selections at all
        # since dims max_value = 5 as defined in
        # caffe2/caffe2/python/hypothesis_test_util.py
        for i in range(X.shape[-1]):
            X[..., i] = i * 1.0 / X.shape[-1]

        op = core.CreateOperator(
            "FlexibleTopK", ["X", "k"], ["Values", "Indices"], device_option=gc
        )

        self.assertGradientChecks(gc, op, [X, k], 0, [0])
