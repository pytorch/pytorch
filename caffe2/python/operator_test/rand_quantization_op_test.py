from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import struct
import unittest
import os

from hypothesis import given, example
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

np.set_printoptions(precision=6)

class TestFloatToFusedRandRowwiseQuantized(hu.HypothesisTestCase):
    @given(X=hu.tensor(min_dim=2, max_dim=2,
                        min_value=1, max_value=17),  # only matrix is supported
           bitwidth_=st.sampled_from([1, 2, 4, 8]),
           random_=st.booleans(),
           **hu.gcs)
    @example(X=np.array([[0., 0., 0., 0.264019]]).astype(np.float32),
            bitwidth_=2,
            random_=False,
            **hu.gcs)
    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/28550")
    def test_rand_quantization(self, X, bitwidth_, random_, gc, dc):

        # python reference of encoder
        def quantization_ref(X):
            in_shape = X.shape
            data_per_byte = 8 // bitwidth_
            output_cols = 10 + in_shape[1] // data_per_byte
            tail = 0
            if in_shape[1] % data_per_byte:
                output_cols += 1
                tail = data_per_byte - in_shape[1] % data_per_byte
            segment = output_cols - 10
            out = np.zeros((in_shape[0], output_cols), dtype=np.uint8)
            for r in range(in_shape[0]):
                out[r][0] = bitwidth_
                out[r][1] = tail
                min_fval = np.amin(X[r])
                max_fval = np.amax(X[r])
                min_bvals = bytearray(struct.pack("f", min_fval))
                max_bvals = bytearray(struct.pack("f", max_fval))
                for idx in range(4):
                    out[r][2 + idx] = min_bvals[idx]
                    out[r][6 + idx] = max_bvals[idx]
                for c in range(in_shape[1]):
                    fval = X[r][c]
                    gap = (max_fval - min_fval) / ((1 << bitwidth_) - 1)
                    thetimes = (fval - min_fval) / (gap + 1e-8)
                    thetimes = np.around(thetimes).astype(np.uint8)
                    out[r][10 + c % segment] += \
                        (thetimes << (bitwidth_ * (c // segment)))
            print("pY:\n{}\n".format(out))
            return (out,)

        # the maximum quantization error
        def get_allowed_errors(X):
            out = np.zeros_like(X)
            for r in range(X.shape[0]):
                min_fval = np.amin(X[r])
                max_fval = np.amax(X[r])
                gap = (max_fval - min_fval) / (2 ** bitwidth_ - 1) + 1e-8
                out[r] = gap
            return out

        # python reference of decoder
        def dec_ref(Y):
            in_shape = Y.shape
            bitwidth = Y[0][0]
            tail = Y[0][1]
            mask = np.array([(1 << bitwidth) - 1]).astype(np.uint8)[0]
            data_per_byte = 8 // bitwidth
            output_cols = (in_shape[1] - 10) * data_per_byte - tail
            segment = in_shape[1] - 10
            out = np.zeros((in_shape[0], output_cols), dtype=np.float)
            for r in range(in_shape[0]):
                min_fval = struct.unpack('f', Y[r][2:6])[0]
                max_fval = struct.unpack('f', Y[r][6:10])[0]
                print(min_fval, max_fval)
                gap = (max_fval - min_fval) / (2 ** bitwidth - 1.) + 1e-8
                for out_c in range(output_cols):
                    bit_start = (out_c // segment) * bitwidth
                    out[r][out_c] = min_fval + gap * \
                        ((Y[r][10 + out_c % segment] >> bit_start) & mask)
            print("pdecX:\n{}\n".format(out))
            return (out,)

        enc_op = core.CreateOperator(
            "FloatToFusedRandRowwiseQuantized",
            ["X"], ["Y"],
            bitwidth=bitwidth_,
            random=random_)
        dec_op = core.CreateOperator(
            "FusedRandRowwiseQuantizedToFloat",
            ["Y"], ["decX"])
        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(enc_op)
        print("X:\n{}\n".format(workspace.FetchBlob("X")))
        Y = workspace.FetchBlob("Y")
        print("Y:\n{}\n".format(Y))

        pY = quantization_ref(X)[0]
        pdecX = dec_ref(Y)[0]

        # The equality check of encoded floating values in bytes may occasionally fail
        # because of precision.
        # Refer to the format of floating values here:
        #   https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        # Instead of using self.assertReferenceChecks(gc, op, [X], quantization_ref),
        # we do the following
        if not random_:
            for r in range(Y.shape[0]):
                # compare min
                np.testing.assert_almost_equal(
                    struct.unpack('f', Y[r][2:6])[0],
                    struct.unpack('f', pY[r][2:6])[0])
                # compare max
                np.testing.assert_almost_equal(
                    struct.unpack('f', Y[r][6:10])[0],
                    struct.unpack('f', pY[r][6:10])[0])
            np.testing.assert_array_equal(Y[:, 0:2], pY[:, 0:2])
            np.testing.assert_array_equal(Y[:, 10:], pY[:, 10:])

        # check decoded floating values are within error thresholds
        workspace.RunOperatorOnce(dec_op)
        decX = workspace.FetchBlob("decX")
        print("decX:\n{}\n".format(decX))
        if random_:
            err_thre = get_allowed_errors(X) + 1e-6
        else:
            err_thre = get_allowed_errors(X) / 2.0 + 1e-6
        err = decX - X
        print("err_thre:\n{}\n".format(err_thre))
        print("err:\n{}\n".format(err))
        np.testing.assert_almost_equal(decX, pdecX, decimal=6)

        np.testing.assert_array_less(
            np.absolute(err),
            err_thre)

        # test the expectation of stochastic quantized values
        # are near to the floating values
        # Warning: it can fail the unit test with small probability
        test_stochastic_quantization = True
        if random_ and test_stochastic_quantization:
            X_sum = np.zeros_like(X)
            test_times = 2000
            for _ in range(test_times):
                workspace.RunOperatorOnce(enc_op)
                workspace.RunOperatorOnce(dec_op)
                X_sum += workspace.FetchBlob("decX")
            X_avg = X_sum / test_times
            print("X_avg:\n{}".format(X_avg))
            print("X    :\n{}".format(X))
            np.testing.assert_array_less(
                np.absolute(X_avg - X),
                5e-2 * get_allowed_errors(X) + 1e-4
            )

        # Check over multiple devices -- CUDA implementation is pending
        # self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
