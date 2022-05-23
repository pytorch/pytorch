import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given


class TestMarginLossL2rOps(hu.HypothesisTestCase):
    def ref_margin_loss(self, y, r, margin):
        n = len(y)
        dy = np.zeros(n)
        loss = 0
        if np.sum(np.abs(r)) < 1e-6:
            return loss, dy

        for i in range(n):
            for j in range(i + 1, n):
                weight = 1.0 / n
                diff = 1 if r[i] - r[j] > 0 else -1
                if (margin > (y[i] - y[j]) * diff) and (r[i] != r[j]):
                    loss += weight * (margin - (y[i] - y[j]) * diff)
                    dy[i] += -diff * weight
                    dy[j] += diff * weight
        return loss, dy

    @given(
        n=st.integers(10, 10),
        k=st.integers(2, 5),
        m=st.integers(1, 5),
        **hu.gcs_cpu_only
    )
    def test_session_margin_loss(self, n, k, m, gc, dc):
        y = np.random.rand(n * m).astype(np.float32)
        r = np.random.randint(k, size=n * m).astype(np.float32)
        # m sessions of length n
        session_lengths = np.repeat(n, m).astype(np.int32)
        ref_loss = np.empty(0)
        ref_scale_loss = np.empty(0)
        ref_dy = np.empty(0)
        ref_scale_dy = np.empty(0)
        for i in range(m):
            r_loss, r_dy = self.ref_margin_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], 0.06
            )
            r_scale_loss, r_scale_dy = self.ref_margin_loss(
                y[(i) * n : (i + 1) * n], r[(i) * n : (i + 1) * n], 0.04
            )
            ref_loss = np.append(ref_loss, r_loss)
            ref_dy = np.append(ref_dy, r_dy)
            ref_scale_loss = np.append(ref_scale_loss, r_scale_loss)
            ref_scale_dy = np.append(ref_scale_dy, r_scale_dy)

        dloss = np.random.random(m).astype(np.float32)

        workspace.blobs["pred"] = y
        workspace.blobs["label"] = r
        workspace.blobs["session_lengths"] = session_lengths
        workspace.blobs["dloss"] = dloss

        # Test scale = 1
        op = core.CreateOperator(
            "SessionMarginLoss",
            ["pred", "label", "session_lengths"],
            ["loss", "dpred"],
            margin=0.06,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dpred"]
        np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_dy, rtol=1e-5, atol=1e-6)
        name = op.output[0]
        arr = workspace.FetchBlob(name)
        self.assertGradientChecks(
            gc, op, [y, r, session_lengths], 0, [0], stepsize=1e-3, threshold=2e-1
        )

        # Test scale > 1
        op = core.CreateOperator(
            "SessionMarginLoss",
            ["pred", "label", "session_lengths"],
            ["loss", "dpred"],
            margin=0.04,
        )
        workspace.RunOperatorOnce(op)
        loss = workspace.blobs["loss"]
        dy = workspace.blobs["dpred"]
        np.testing.assert_allclose(loss, ref_scale_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dy, ref_scale_dy, rtol=1e-5, atol=1e-6)
        self.assertGradientChecks(
            gc, op, [y, r, session_lengths], 0, [0], stepsize=1e-3, threshold=2e-1
        )
