from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestListwiseL2rOps(hu.HypothesisTestCase):
    def ref_lambda_rank_ndcg_loss(self, y, r):
        n = len(y)

        def get_discounts(v):
            x = np.argsort(v)
            d = [0 for _ in range(n)]
            for i in range(n):
                d[x[i]] = 1. / np.log2(n - i + 1.)
            return d

        def sigm(x):
            return 1 / (1 + np.exp(-x))

        def log_sigm(x):
            return -np.log(1 + np.exp(-x))

        g = [2**r[i] for i in range(n)]
        d = get_discounts(r)
        idcg = sum([g[i] * d[i] for i in range(n)])

        d = get_discounts(y)
        loss = 0
        dy = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lambda_weight = np.abs((2**r[i] - 2**r[j]) * (d[i] - d[j]))
                rank_loss = -log_sigm(
                    y[i] - y[j] if r[i] > r[j] else y[j] - y[i]
                )
                rank_dy = (0. if r[i] > r[j] else 1.) - sigm(-y[i] + y[j])
                loss += lambda_weight * rank_loss / idcg
                dy[i] += lambda_weight * rank_dy / idcg
        return loss, dy

    @given(n=st.integers(1, 20), k=st.integers(2, 5))
    def test_lambda_rank_ndcg_loss(self, n, k):
        y = np.random.rand(n).astype(np.float32)
        r = np.random.randint(k, size=n).astype(np.float32)
        dloss = np.random.random(1).astype(np.float32)

        workspace.blobs['y'] = y
        workspace.blobs['r'] = r
        workspace.blobs['dloss'] = dloss

        op = core.CreateOperator('LambdaRankNdcg', ['y', 'r'], ['loss', 'dy'])
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs['loss']
        dy = workspace.blobs['dy']
        ref_loss, ref_dy = self.ref_lambda_rank_ndcg_loss(y, r)
        self.assertAlmostEqual(np.asscalar(loss), ref_loss, delta=1e-4)
        np.testing.assert_allclose(dy, ref_dy, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            'LambdaRankNdcgGradient', ['y', 'dy', 'dloss'], ['dy_back']
        )
        workspace.RunOperatorOnce(op)
        dy_back = workspace.blobs['dy_back']
        np.testing.assert_allclose(
            dy_back, np.asscalar(dloss) * ref_dy, rtol=1e-5, atol=1e-6
        )
