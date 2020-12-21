




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given, settings
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import unittest


class TestChannelBackpropStats(serial.SerializedTestCase):
    @given(
        size=st.integers(7, 10),
        inputChannels=st.integers(1, 10),
        batchSize=st.integers(1, 3),
        **hu.gcs
    )
    @settings(deadline=10000)
    def testChannelBackpropStats(self, size, inputChannels, batchSize, gc, dc):

        op = core.CreateOperator(
            "ChannelBackpropStats",
            ["X", "mean", "invStdDev", "outputGrad"],
            ["scaleGrad", "biasGrad"],
        )

        def referenceChannelBackpropStatsTest(X, mean, invStdDev, outputGrad):
            scaleGrad = np.zeros(inputChannels)
            biasGrad = np.zeros(inputChannels)
            for n in range(batchSize):
                for c in range(inputChannels):
                    for h in range(size):
                        for w in range(size):
                            biasGrad[c] += outputGrad[n, c, h, w]
                            scaleGrad[c] += (
                                X[n, c, h, w] - mean[c]
                            ) * invStdDev[c] * outputGrad[n, c, h, w]
            return scaleGrad, biasGrad

        X = np.random.rand(batchSize, inputChannels, size, size)\
                     .astype(np.float32) - 0.5
        sums = np.sum(X, axis=(0, 2, 3), keepdims=False)
        numPixels = size * size * batchSize
        mean = sums / numPixels
        sumsq = np.sum(X**2, axis=(0, 2, 3), keepdims=False)
        var = ((sumsq -
                (sums * sums) / numPixels) / numPixels).astype(np.float32)
        invStdDev = 1 / np.sqrt(var)
        outputGrad = np.random.rand(batchSize, inputChannels, size, size)\
            .astype(np.float32) - 0.5
        self.assertReferenceChecks(
            gc, op, [X, mean, invStdDev, outputGrad],
            referenceChannelBackpropStatsTest
        )


if __name__ == "__main__":
    unittest.main()
