from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time

# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, dyndep
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
from caffe2.python.fakelowp.test_utils import print_test_debug_info

workspace.GlobalInit(["caffe2", "--glow_global_fp16=1",
                      "--glow_global_fused_scale_offset_fp16=1",
                      "--glow_global_force_sls_fp16_accum=1"])


class SparseLengthsSumTest(TestCase):
    def test_slws_fused_4bit_rowwise_all_same(self):
        # Comment out for predictable debugging
        seed = int(time.time())
        np.random.seed(seed)
        workspace.ResetWorkspace()
        n = 1
        m = 2
        data = np.ones((n, m)).astype(np.float32) * 0.2 - 0.1

        max_segments = 5
        max_segment_length = 100
        num_lengths = np.random.randint(1, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(0, max_segment_length + 1,
                                    size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)
        indices = np.zeros(num_indices, dtype=np.int64)
        weights = np.random.uniform(low=-0.5, high=0.5,
            size=[len(indices)]).astype(np.float32)
        weights = np.ones(len(indices)).astype(np.float32)

        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"])
        pred_net.external_output.append("Y")
        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused4BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        ref_net = caffe2_pb2.NetDef()
        ref_net.name = "ref"
        ref_net.external_input.extend(
            ["quantized_data", "weights", "indices", "lengths"])
        ref_net.external_output.append("Y")
        ref_net.op.add().CopyFrom(
            core.CreateOperator(
                "SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused4BitRowwiseQuantized",
                ['data'],
                ['quantized_data']
            )
        )

        print("quantized", workspace.FetchBlob("quantized_data"))
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segments * max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False
        )

        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op)
        np.testing.assert_equal(num_onnxified_ops, 1)

        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("weights", weights)

        workspace.CreateNet(pred_net_onnxified)
        workspace.CreateNet(ref_net)

        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchBlob('Y')

        workspace.RunNet(ref_net.name)
        Y_c2 = workspace.FetchBlob('Y')

        if not np.allclose(Y_c2, Y_glow):
            print_test_debug_info(
                "slws_fused_4bit_rowwise",
                {"seed": seed,
                 "indices": indices,
                 "data": data,
                 "lengths": lengths,
                 "weights": weights,
                 "Y_c2": Y_c2,
                 "Y_glow": Y_glow,
                 "diff": Y_glow - Y_c2,
                 "rowwise_diff": (Y_glow - Y_c2)[:, 0]})
            assert(0)

    def test_slws_fused_4bit_rowwise(self):
        # Comment out for predictable debugging
        seed = int(time.time() * 1000) % 2 ** 16
        print(seed)
        np.random.seed(seed)
        workspace.ResetWorkspace()

        n = 20000
        DIM = 6
        data = (4 * np.random.random_sample((n, DIM)) + 1).astype(np.float32)

        max_segments = 200
        max_segment_length = 200
        num_lengths = np.random.randint(0, max_segments + 1)
        # number of segments to run
        lengths = np.random.randint(2, max_segment_length + 1, size=num_lengths).astype(
            np.int32
        )
        num_indices = np.sum(lengths)
        indices = np.random.randint(low=0, high=n, size=num_indices, dtype=np.int64)
        weights = np.random.uniform(low=0.01, high=0.5, size=[len(indices)]).astype(
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
                "SparseLengthsWeightedSumFused4BitRowwise",
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
                "SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI",
                ["quantized_data", "weights", "indices", "lengths"],
                ["Y"],
            )
        )

        workspace.FeedBlob("data", data)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FloatToFused4BitRowwiseQuantized", ["data"], ["quantized_data"]
            )
        )
        onnxified_net = onnxifi_caffe2_net(
            pred_net,
            {},
            max_batch_size=max_segments,
            max_seq_size=max_segments * max_segment_length,
            debug=True,
            adjust_batch=True,
            use_onnx=False,
        )
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
                "slws_fused_4bit",
                {
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
