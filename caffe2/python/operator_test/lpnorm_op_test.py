




import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given, settings
import hypothesis.strategies as st


class LpnormTest(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=1,
                             min_dim=1,
                             max_dim=3,
                             dtype=np.float32),
           **hu.gcs)
    @settings(deadline=10000)
    def test_Lp_Norm(self, inputs, gc, dc):
        X = inputs[0]
        # avoid kinks by moving away from 0
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.ws.create_blob("X").feed(X)
        op = core.CreateOperator(
            'LpNorm',
            ['X'],
            ['l1_norm'],
            p=1,
        )
        self.ws.run(op)

        np.testing.assert_allclose(self.ws.blobs[("l1_norm")].fetch(),
                                     np.linalg.norm((X).flatten(), ord=1),
                                    rtol=1e-4, atol=1e-4)

        self.assertDeviceChecks(dc, op, [X], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=1e-2, threshold=1e-2)

        op = core.CreateOperator(
            'LpNorm',
            ['X'],
            ['l2_norm'],
            p=2,
        )
        self.ws.run(op)

        np.testing.assert_allclose(
            self.ws.blobs[("l2_norm")].fetch(),
            np.linalg.norm((X).flatten(), ord=2)**2,
            rtol=1e-4,
            atol=1e-4
        )

        self.assertDeviceChecks(dc, op, [X], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=1e-2, threshold=1e-2)

        op = core.CreateOperator(
            'LpNorm',
            ['X'],
            ['l2_averaged_norm'],
            p=2,
            average=True
        )
        self.ws.run(op)

        np.testing.assert_allclose(
            self.ws.blobs[("l2_averaged_norm")].fetch(),
            np.linalg.norm((X).flatten(), ord=2)**2 / X.size,
            rtol=1e-4,
            atol=1e-4
        )

    @given(x=hu.tensor(
        min_dim=1, max_dim=10, dtype=np.float32,
        elements=st.integers(min_value=-100, max_value=100)),
        p=st.integers(1, 2),
        average=st.integers(0, 1)
    )
    def test_lpnorm_shape_inference(self, x, p, average):
        workspace.FeedBlob('x', x)

        net = core.Net("lpnorm_test")
        result = net.LpNorm(['x'], p=p, average=bool(average))
        (shapes, types) = workspace.InferShapesAndTypes([net])
        workspace.RunNetOnce(net)

        self.assertEqual(shapes[result], list(workspace.blobs[result].shape))
        self.assertEqual(types[result], core.DataType.FLOAT)
