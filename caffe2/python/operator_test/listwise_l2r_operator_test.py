

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given


class TestListwiseL2rOps(hu.HypothesisTestCase):
    def ref_lambda_rank_loss(
        self, y, r, use_ndcg_as_loss, use_idcg_normalization, use_exp_gain
    ):
        n = len(y)

        def get_discounts(v):
            x = np.argsort(v)
            d = [0 for _ in range(n)]
            for i in range(n):
                d[x[i]] = 1.0 / np.log2(n - i + 1.0)
            return d

        def sigm(x):
            return 1 / (1 + np.exp(-x))

        def log_sigm(x):
            return -np.log(1 + np.exp(-x))

        dy = np.zeros(n)
        loss = 0
        if np.sum(np.abs(r)) < 1e-6:
            return loss, dy
        if use_ndcg_as_loss and (not use_exp_gain):
            g = [r[i] for i in range(n)]
        else:
            g = [2 ** r[i] for i in range(n)]
        d = get_discounts(r)
        idcg = sum([g[i] * d[i] for i in range(n)])

        if use_idcg_normalization:
            session_weight = max(idcg, 1e-5)
        else:
            session_weight = 1

        d = get_discounts(y)

        if use_ndcg_as_loss:
            dcg = sum(g[i] * d[i] for i in range(n))
            loss = (idcg - dcg) / session_weight
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lambda_weight = np.abs((g[i] - g[j]) * (d[i] - d[j]))
                rank_loss = -log_sigm(y[i] - y[j] if r[i] > r[j] else y[j] - y[i])
                rank_dy = (0.0 if r[i] > r[j] else 1.0) - sigm(-y[i] + y[j])
                if not use_ndcg_as_loss:
                    loss += lambda_weight * rank_loss / session_weight
                dy[i] += lambda_weight * rank_dy / session_weight

        return loss, dy

    @given(n=st.integers(1, 20), k=st.integers(2, 5), m=st.integers(3, 5))
    def test_lambda_rank_loss(self, n, k, m):
        y = np.random.rand(n * m).astype(np.float32)
        r = np.random.randint(k, size=n * m).astype(np.float32)
        # m sessions of length n
        session_lengths = np.repeat(n, m).astype(np.int32)
        ref_loss = np.empty(0)
        ref_ndcg_loss = np.empty(0)
        ref_ndcg_loss_no_exp = np.empty(0)
        ref_dcg_loss = np.empty(0)
        ref_dcg_loss_no_exp = np.empty(0)
        ref_dy = np.empty(0)
        ref_dy_no_exp = np.empty(0)
        ref_dcg_dy = np.empty(0)
        ref_dcg_dy_no_exp = np.empty(0)
        for i in range(m):
            r_loss, r_dy = self.ref_lambda_rank_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], False, True, False
            )
            r_ndcg_loss, _ = self.ref_lambda_rank_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], True, True, True
            )
            r_ndcg_loss_no_exp, r_dy_no_exp = self.ref_lambda_rank_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], True, True, False
            )
            r_dcg_loss, r_dcg_dy = self.ref_lambda_rank_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], True, False, True
            )
            r_dcg_loss_no_exp, r_dcg_dy_no_exp = self.ref_lambda_rank_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], True, False, False
            )
            ref_loss = np.append(ref_loss, r_loss)
            ref_dy = np.append(ref_dy, r_dy)
            ref_ndcg_loss = np.append(ref_ndcg_loss, r_ndcg_loss)

            ref_ndcg_loss_no_exp = np.append(ref_ndcg_loss_no_exp, r_ndcg_loss_no_exp)
            ref_dy_no_exp = np.append(ref_dy_no_exp, r_dy_no_exp)

            ref_dcg_loss = np.append(ref_dcg_loss, r_dcg_loss)
            ref_dcg_dy = np.append(ref_dcg_dy, r_dcg_dy)

            ref_dcg_loss_no_exp = np.append(ref_dcg_loss_no_exp, r_dcg_loss_no_exp)
            ref_dcg_dy_no_exp = np.append(ref_dcg_dy_no_exp, r_dcg_dy_no_exp)

        dloss = np.random.random(m).astype(np.float32)

        workspace.blobs["y"] = y
        workspace.blobs["r"] = r
        workspace.blobs["session_lengths"] = session_lengths
        workspace.blobs["dloss"] = dloss

        op = core.CreateOperator(
            "LambdaRankNdcg",
            ["y", "r", "session_lengths"],
            ["loss", "dy"],
            use_ndcg_as_loss=False,
            use_idcg_normalization=True,
            use_exp_gain=False,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dy"]
        np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dy, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            "LambdaRankNdcg",
            ["y", "r", "session_lengths"],
            ["loss", "dy"],
            use_ndcg_as_loss=True,
            use_idcg_normalization=True,
            use_exp_gain=True,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dy"]
        np.testing.assert_allclose(loss, ref_ndcg_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dy, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            "LambdaRankNdcgGradient",
            ["y", "session_lengths", "dy", "dloss"],
            ["dy_back"],
        )
        workspace.RunOperatorOnce(op)
        dy_back = workspace.blobs["dy_back"]
        for i in range(m):
            np.testing.assert_allclose(
                dy_back[i * n : (i + 1) * n],
                dloss[i] * ref_dy[i * n : (i + 1) * n],
                rtol=1e-5,
                atol=1e-6,
            )

        op = core.CreateOperator(
            "LambdaRankNdcg",
            ["y", "r", "session_lengths"],
            ["loss", "dy"],
            use_ndcg_as_loss=True,
            use_idcg_normalization=True,
            use_exp_gain=False,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dy"]
        np.testing.assert_allclose(loss, ref_ndcg_loss_no_exp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dy_no_exp, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            "LambdaRankNdcgGradient",
            ["y", "session_lengths", "dy", "dloss"],
            ["dy_back"],
        )
        workspace.RunOperatorOnce(op)
        dy_back = workspace.blobs["dy_back"]
        for i in range(m):
            np.testing.assert_allclose(
                dy_back[i * n : (i + 1) * n],
                dloss[i] * ref_dy_no_exp[i * n : (i + 1) * n],
                rtol=1e-5,
                atol=1e-6,
            )

        op = core.CreateOperator(
            "LambdaRankNdcg",
            ["y", "r", "session_lengths"],
            ["loss", "dy"],
            use_ndcg_as_loss=True,
            use_idcg_normalization=False,
            use_exp_gain=True,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dy"]
        np.testing.assert_allclose(loss, ref_dcg_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dcg_dy, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            "LambdaRankNdcgGradient",
            ["y", "session_lengths", "dy", "dloss"],
            ["dy_back"],
        )
        workspace.RunOperatorOnce(op)
        dy_back = workspace.blobs["dy_back"]
        for i in range(m):
            np.testing.assert_allclose(
                dy_back[i * n : (i + 1) * n],
                dloss[i] * ref_dcg_dy[i * n : (i + 1) * n],
                rtol=1e-5,
                atol=1e-6,
            )

        op = core.CreateOperator(
            "LambdaRankNdcg",
            ["y", "r", "session_lengths"],
            ["loss", "dy"],
            use_ndcg_as_loss=True,
            use_idcg_normalization=False,
            use_exp_gain=False,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dy"]
        np.testing.assert_allclose(loss, ref_dcg_loss_no_exp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dcg_dy_no_exp, rtol=1e-5, atol=1e-6)

        op = core.CreateOperator(
            "LambdaRankNdcgGradient",
            ["y", "session_lengths", "dy", "dloss"],
            ["dy_back"],
        )
        workspace.RunOperatorOnce(op)
        dy_back = workspace.blobs["dy_back"]
        for i in range(m):
            np.testing.assert_allclose(
                dy_back[i * n : (i + 1) * n],
                dloss[i] * ref_dcg_dy_no_exp[i * n : (i + 1) * n],
                rtol=1e-5,
                atol=1e-6,
            )
