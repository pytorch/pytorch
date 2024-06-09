import unittest
from typing import Dict, Any

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
        "--glow_global_fp16=1",
        "--glow_global_fused_scale_offset_fp16=1",
        "--glow_global_force_sls_fp16_accum=1",
    ]
)
GLOW_MATMUL_ATOL = 1e-5
GLOW_MATMUL_RTOL = 1e-3


class SparseLengthsSum8BitFakeNNPIFp16Test(serial.SerializedTestCase):
    def Skip_test_SLS_NonQuantized_fp16(self):
        N = 20000
        DIM = 64
        D = (4 * np.random.random_sample((N, DIM)) + 1).astype(np.float32)
        I = (np.random.randint(0, N, size=12)).astype(np.int64)
        L = np.asarray([4, 4, 4]).astype(np.int32)
        workspace.FeedBlob("D", D)

        ref_c2_net = core.Net("test_ref_c2")
        ref_c2_net.SparseLengthsSum(["D", "I", "L"], "ref_out")
        ref_c2_net.Proto().external_input.extend(["D", "I", "L"])
        ref_c2_net.Proto().external_output.extend(["ref_out"])

        fp16_c2_net = core.Net("test_fp16_c2")
        fp16_c2_net.SparseLengthsSumFakeFP16AccFP16(["D", "I", "L"], "fp16_out")

        input_dict : Dict[Any, Any] = {}

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(["D", "I", "L"])
        pred_net.external_output.append("glow_out")
        pred_net.op.add().CopyFrom(
            core.CreateOperator("SparseLengthsSum", ["D", "I", "L"], ["glow_out"])
        )

        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            input_dict,
            max_batch_size=3,
            max_seq_size=16,
            debug=True,
            adjust_batch=False,
            use_onnx=False,
        )

        num_onnxified_ops = sum(
            1 if op.type == "Onnxifi" else 0 for op in onnxified_net.op
        )
        print(onnxified_net)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("I", I)
        workspace.FeedBlob("L", L)

        workspace.RunNetOnce(ref_c2_net)
        ref_c2_out = workspace.FetchBlob("ref_out")

        workspace.RunNetOnce(fp16_c2_net)
        fp16_c2_out = workspace.FetchBlob("fp16_out")

        np.testing.assert_allclose(fp16_c2_out, ref_c2_out, atol=1e-3, rtol=1e-3)

        workspace.RunNetOnce(onnxified_net)
        fp16_glow_out = workspace.FetchBlob("glow_out")

        if not np.allclose(fp16_glow_out, fp16_c2_out):
            diff = np.abs(fp16_glow_out - fp16_c2_out)
            print_test_debug_info(
                "sls",
                {
                    "indices": I,
                    "data": D,
                    "lengths": L,
                    "Y_c2": fp16_c2_out,
                    "Y_glow": fp16_glow_out,
                    "diff": diff,
                    "rowwise_diff": diff[:, 0],
                },
            )
            assert 0

    @given(seed=st.integers(0, 65535))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_slws_fused_8bit_rowwise_all_same(self, seed):
        # Comment out for predictable debugging
        np.random.seed(seed)
        workspace.ResetWorkspace()
        n = 1
        m = 2
        data = np.ones((n, m)).astype(np.float32) * 0.2 - 0.1

        max_segments = 5
        max_segment_length = 200
        num_lengths = np.random.randint(1, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(0, max_segment_length + 1, size=num_lengths).astype(
            np.int32
        )
        num_indices = np.sum(lengths)
        indices = np.zeros(num_indices, dtype=np.int64)
        weights = np.random.uniform(low=-0.5, high=0.5, size=[len(indices)]).astype(
            np.float32
        )

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
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
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
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )

        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(ref_net)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob("Y")

        workspace.RunNet(ref_net.name)
        Y_c2 = workspace.FetchBlob("Y")

        if not np.allclose(Y_c2, Y_glow):
            print_test_debug_info(
                "slws_fused_8bit_rowwise",
                {
                    "seed": seed,
                    "indices": indices,
                    "data": data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_c2": Y_c2,
                    "Y_glow": Y_glow,
                    "diff": Y_glow - Y_c2,
                    "rowwise_diff": (Y_glow - Y_c2)[:, 0],
                },
            )
            assert 0

    @given(
        seed=st.integers(0, 65535),
        num_rows=st.integers(2, 20),
        embedding_dim=st.sampled_from([8, 12, 16, 24, 32, 54, 64, 128]),
        batch_size=st.integers(1, 5),
        max_weight=st.integers(0, 100),
    )
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_slws_fused_8bit_rowwise(self, seed, num_rows, embedding_dim, batch_size, max_weight):
        np.random.seed(seed)
        workspace.ResetWorkspace()

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

        assert(len(weights) < 64000)

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
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
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
                "slws_fused_8bit_rowwise_inv_scale",
                {
                    "seed": seed,
                    "num_rows": num_rows,
                    "embedding_dim": embedding_dim,
                    "batch_size": batch_size,
                    "max_weight": max_weight,
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

    # Simple test to aid debugging order of operations
    # Minimize the case to an SLS that adds two rows
    @given(seed=st.integers(0, 65535))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_small_sls(self, seed):
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
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
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
                "slws_fused_8bit_rowwise_inv_scale",
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

    @given(seed=st.integers(0, 65535))
    @settings(deadline=datetime.timedelta(seconds=10))
    def test_sls_layernorm(self, seed):
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
        pred_net.external_output.append("Y_norm")
        pred_net.external_output.append("Y_mean")
        pred_net.external_output.append("Y_std")

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNorm",
                ["Y"],
                ["Y_norm", "Y_mean", "Y_std"],
                epsilon=1e-4,
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"]
        )
        ref_net.external_output.append("Y_norm")
        ref_net.external_output.append("Y_mean")
        ref_net.external_output.append("Y_std")

        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "LayerNormFakeFP16NNPI",
                ["Y"],
                ["Y_norm", "Y_mean", "Y_std"],
                epsilon=1e-4,
                axis=1,
                elementwise_affine=False
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
        print("before", pred_net)
        print("after", onnxified_net)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(onnxified_net)
        workspace.CreateNet(ref_net)

        workspace.RunNet(onnxified_net.name)
        Y_glow = workspace.FetchBlob("Y_norm")
        Y_mean_glow = workspace.FetchBlob("Y_mean")
        Y_std_glow = workspace.FetchBlob("Y_std")

        workspace.RunNet(ref_net.name)
        Y = workspace.FetchBlob("Y")
        print("pre normalization", Y)
        Y_ref = workspace.FetchBlob("Y_norm")
        Y_mean_ref = workspace.FetchBlob("Y_mean")
        Y_std_ref = workspace.FetchBlob("Y_std")

        # print(Y_ref, Y_glow)
        # print(Y_ref.shape, Y_glow.shape)

        diff = np.abs(Y_ref - Y_glow)
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
                "slws_fused_8bit_rowwise_inv_scale",
                {
                    "seed": seed,
                    "indices": indices,
                    "data": data,
                    "quantized_data": quantized_data,
                    "lengths": lengths,
                    "weights": weights,
                    "Y_norm_glow": Y_glow,
                    "Y_norm_ref": Y_ref,
                    "Y_mean_glow": Y_mean_glow,
                    "Y_std_glow": Y_std_glow,
                    "Y_mean_ref": Y_mean_ref,
                    "Y_std_ref": Y_std_ref,
                    "diff": diff,
                    "rowwise_diff": np.max(diff, axis=1),
                },
            )
            assert 0


if __name__ == '__main__':
    unittest.main()
