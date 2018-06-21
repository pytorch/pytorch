from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import struct

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestFloatToFusedRandRowwiseQuantized(hu.HypothesisTestCase):
    @given(X=hu.tensor(min_dim=2, max_dim=2,
    min_value=1, max_value=17),  # only matrix is supported
           bitwidth_=st.sampled_from([1, 2, 4, 8]),
           random_=st.sampled_from([False]),  # only deterministic supported in unittest
           **hu.gcs)
    def test_rand_quantization(self, X, bitwidth_, random_, gc, dc):

        def rand_quantization_ref(X):
            in_shape = X.shape
            data_per_byte = 8 // bitwidth_
            output_cols = 10 + in_shape[1] // data_per_byte
            tail = 0
            if in_shape[1] % data_per_byte:
                output_cols += 1
                tail = data_per_byte - in_shape[1] % data_per_byte
            segment = output_cols - 10
            out = np.zeros((in_shape[0], output_cols), dtype=np.uint8)
            for r in range(0, in_shape[0]):
                out[r][0] = bitwidth_
                out[r][1] = tail
                min_fval = np.amin(X[r])
                max_fval = np.amax(X[r])
                min_bvals = bytearray(struct.pack("f", min_fval))
                max_bvals = bytearray(struct.pack("f", max_fval))
                for idx in range(0, 4):
                    out[r][2 + idx] = min_bvals[idx]
                    out[r][6 + idx] = max_bvals[idx]
                for c in range(0, in_shape[1]):
                    fval = X[r][c]
                    gap = (max_fval - min_fval) / ((1 << bitwidth_) - 1)
                    thetimes = (fval - min_fval) / (gap + 1e-8)
                    thetimes = round(thetimes)
                    out[r][10 + c % segment] = \
                        (out[r][10 + c % segment] << bitwidth_) + thetimes
            print("pY:\n{}\n".format(out))

            return (out,)

        op = core.CreateOperator(
            "FloatToFusedRandRowwiseQuantized",
            ["X"], ["Y"],
            bitwidth=bitwidth_,
            random=random_)
        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        print("X:\n{}\n".format(workspace.FetchBlob("X")))
        Y = workspace.FetchBlob("Y")
        print("Y:\n{}\n".format(Y))

        pY = rand_quantization_ref(X)[0]

        # The equality check of encoded floating values in bytes may occasionally fail
        # because of precision.
        # Refer to the format of floating values here:
        #   https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        ## self.assertReferenceChecks(gc, op, [X], rand_quantization_ref)
        # Instead of using the above assertReferenceChecks,
        # we replace the first byte (the least significant byte) of encoded
        # floating values with zeros, and use assert_array_equal
        Y[:, 2] = 0
        Y[:, 6] = 0
        pY[:, 2] = 0
        pY[:, 6] = 0
        np.testing.assert_array_equal(Y, pY)

        # Check over multiple devices -- CUDA implementation is pending
        # self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
