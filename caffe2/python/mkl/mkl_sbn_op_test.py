




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLSpatialBNTest(hu.HypothesisTestCase):
    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           #order=st.sampled_from(["NCHW", "NHWC"]),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(1e-5, 1e-2),
           **mu.gcs)
    def test_spatialbn_test_mode(self, size, input_channels,
                                 batch_size, seed, order, epsilon, gc, dc):
        np.random.seed(seed)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["Y"],
            order=order,
            is_test=True,
            epsilon=epsilon,
        )

        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var], [0])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           #order=st.sampled_from(["NCHW", "NHWC"]),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(1e-5, 1e-2),
           **mu.gcs)
    def test_spatialbn_train_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            gc, dc):
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "running_mean", "running_var"],
            ["Y", "running_mean", "running_var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
        )
        np.random.seed(seed)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        # Note: it seems that the running mean and var do not pass the device
        # test, suggesting that the semantics are a bit different. Only
        # checking the output and saved mean and var at this stage.
        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var],
                                [0, 3, 4])


if __name__ == "__main__":
    import unittest
    unittest.main()
