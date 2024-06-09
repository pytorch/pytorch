import unittest

# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa
import datetime
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=0",
        "--glow_global_fused_scale_offset_fp16=0",
        "--glow_global_force_sls_fp16_accum=0",
    ]
)
GLOW_MATMUL_ATOL = 1e-5
GLOW_MATMUL_RTOL = 1e-3

class SparseLengthsSum8BitFakeNNPIFp32Test(serial.SerializedTestCase):
    @given(
        seed=st.integers(0, 65535),
        num_rows=st.integers(2, 20),
        embedding_dim=st.sampled_from([8, 12, 16, 24, 32, 54, 64, 128]),
        batch_size=st.integers(1, 5),
        max_weight=st.integers(0, 100),
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_slws_fused_8bit_rowwise_acc32_nnpi(
        self, seed, num_rows, embedding_dim, batch_size, max_weight
    ):
        workspace.GlobalInit(
            [
                "caffe2",
                "--glow_global_fp16=0",
                "--glow_global_fused_scale_offset_fp16=0",
                "--glow_global_force_sls_fp16_accum=0",
            ]
        )

        workspace.ResetWorkspace()
        np.random.seed(seed)
        data = np.random.rand(num_rows, embedding_dim).astype(np.float32)
        lengths = np.random.choice(np.arange(1, num_rows), batch_size).astype(np.int32)

        _indices = []
        for length in lengths:
            _indices.extend(np.random.choice(np.arange(1, num_rows), length))
        indices = np.asarray(_indices).astype(np.int64)

        weights = np.random.uniform(
            low=0,
            high=max_weight,
            size=[len(indices)]
        ).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized",
                ["data"],
                ["quantized_data"]
            )
        )
        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=batch_size,
            max_seq_size=np.max(lengths),
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in onnxified_net.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            print_test_debug_info(
                "test_slws_fused_8bit_rowwise_acc32_nnpi",
                {
                    "seed": seed,
                    "num_rows": num_rows,
                    "embedding_dim": embedding_dim,
                    "batch_size": batch_size,
                    "indices": indices,
                    "data": data.shape,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0


    @given(seed=st.integers(0, 65535))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_small_sls_acc32(self, seed):
        workspace.GlobalInit(
            [
                "caffe2",
                "--glow_global_fp16=0",
                "--glow_global_fused_scale_offset_fp16=0",
                "--glow_global_force_sls_fp16_accum=0",
            ]
        )
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 2
        DIM = 3
        data = 4 * (np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        lengths = np.array([n], dtype=np.int32)
        indices = np.array(range(n), dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[n]).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP32NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused8BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )

        quantized_data = workspace.FetchBlob("quantized_data")

        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=1,
            max_seq_size=n,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in onnxified_net.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_ref = workspace.FetchBlob("Y")

        diff = np.abs((Y_ref - Y_glow) / (Y_ref + 1e-8))
        max_err = np.max(diff, axis=1)
        num_offenders = (max_err > 0).sum()
        if num_offenders > 0:
            np.set_printoptions(precision=12)
            print(
                "ref",
                Y_ref.astype(np.float16).astype(np.float32),
                "glow",
                Y_glow.astype(np.float16).astype(np.float32),
            )
            print_test_debug_info(
                "test_small_sls_acc32",
                {
                    "seed": seed,
                    "indices": indices,
                    "data": data,
                    "quantized_data": quantized_data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_glow": Y_glow,
                    "Y_ref": Y_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0


if __name__ == '__main__':
    unittest.main()
