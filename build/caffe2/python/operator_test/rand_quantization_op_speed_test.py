from __future__ import absolute_import, division, print_function, unicode_literals

import time

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given


np.set_printoptions(precision=6)


class TestSpeedFloatToFusedRandRowwiseQuantized(hu.HypothesisTestCase):
    @given(
        bitwidth_=st.sampled_from([1, 2, 4, 8]),
        random_=st.sampled_from([True, False]),
        data_shape_=st.sampled_from(
            [
                np.array([32, 512]),
                np.array([1, 1024]),
                np.array([1024, 1024]),
                np.array([1024, 1224]),
                np.array([512, 969]),
            ]
        ),
        **hu.gcs
    )
    def test_speed_of_rand_quantization(self, bitwidth_, random_, data_shape_, gc, dc):
        X1 = np.random.rand(data_shape_[0], data_shape_[1]).astype(np.float32)
        X2 = np.random.rand(data_shape_[0], data_shape_[1]).astype(np.float32)

        sub_scale_sum_net = core.Net("sub_scale_sum")
        sub_op = core.CreateOperator("Sub", ["X1", "X2"], ["dX"])
        scale_op = core.CreateOperator("Scale", ["dX"], ["dX"], scale=0.023)
        sum_op = core.CreateOperator("Sum", ["X2", "dX"], ["X2"])
        sub_scale_sum_net.Proto().op.extend([sub_op, scale_op, sum_op])

        enc_net = core.Net("enc")
        enc_op = core.CreateOperator(
            "FloatToFusedRandRowwiseQuantized",
            ["dX"],
            ["Y"],
            bitwidth=bitwidth_,
            random=random_,
        )
        enc_net.Proto().op.extend([enc_op])

        dec_net = core.Net("dec")
        dec_op = core.CreateOperator(
            "FusedRandRowwiseQuantizedToFloat", ["Y"], ["decX"]
        )
        dec_net.Proto().op.extend([dec_op])

        workspace.FeedBlob("X1", X1)
        workspace.FeedBlob("X2", X2)

        workspace.CreateNet(sub_scale_sum_net)
        workspace.CreateNet(enc_net)
        workspace.CreateNet(dec_net)
        workspace.RunNet(sub_scale_sum_net)
        workspace.RunNet(enc_net)
        workspace.RunNet(dec_net)

        sub_scale_sum_time = 0
        enc_time = 0
        dec_time = 0
        times = 10
        for _ in range(times):
            start = time.time()
            workspace.RunNet(sub_scale_sum_net)
            end = time.time()
            sub_scale_sum_time += end - start

            start = time.time()
            workspace.RunNet(enc_net)
            end = time.time()
            enc_time += end - start

            start = time.time()
            workspace.RunNet(dec_net)
            end = time.time()
            dec_time += end - start

        print("Sub+Scale+Sum time: {} ms".format(sub_scale_sum_time / times * 1000))
        print(
            "Quantizing time: {} ms ({}X)".format(
                enc_time / times * 1000, enc_time / sub_scale_sum_time
            )
        )
        print(
            "De-quantizing time: {} ms ({}X)".format(
                dec_time / times * 1000, dec_time / sub_scale_sum_time
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
