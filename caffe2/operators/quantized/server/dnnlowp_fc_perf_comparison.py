from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import operator

import numpy as np
from caffe2.python import core, dyndep, workspace


dyndep.InitOpsLibrary("@/caffe2/caffe2/fb/operators:fully_connected_op_skinny")
dyndep.InitOpsLibrary("//caffe2/caffe2/operators/quantized/server:dnnlowp_ops")


def GetArgumentParser():
    parser = argparse.ArgumentParser(
        description="Caffe2 benchmark. Extra args will be passed to Caffe2"
    )
    parser.add_argument("--batch_size", type=int, default=20, help="The batch size.")
    parser.add_argument(
        "--input_dim", type=int, default=463, help="The input dense dimension."
    )
    parser.add_argument(
        "--output_dim", type=int, default=128, help="The output dense dimension."
    )
    parser.add_argument(
        "--l3_cache_size",
        type=int,
        default=30 * 2 ** 20 // 4,
        help="used to wipe caches between runs",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1000, help="how many times to run each op"
    )
    parser.add_argument(
        "--all_shapes",
        default=False,
        action="store_true",
        help="use all collected input shapes.",
    )
    parser.add_argument(
        "--quantize_input",
        default=False,
        action="store_true",
        help="Use int8 inputs in FC",
    )
    parser.add_argument(
        "--quantize_output",
        default=False,
        action="store_true",
        help="Do not dequantize outputs in FC",
    )
    return parser


# {M,    N,    K}
# NMT
input_shapes_nmt = np.array(
    [
        [1, 128, 512],
        [1, 1024, 256],
        [1, 2048, 512],
        [1, 4096, 1024],
        [6, 256, 1024],
        [6, 256, 2048],
        [6, 512, 512],
        [6, 1024, 256],
        [6, 2048, 256],
        [6, 2048, 512],
        [6, 4096, 256],
        [6, 4096, 1024],
        [6, 4096, 2048],
        [10, 2048, 256],
        [10, 4096, 1024],
        [20, 2048, 256],
        [20, 4096, 1024],
        [102, 1024, 512],
        [102, 2323, 256],
        [102, 512, 256],
    ]
)

# Speech
input_shapes_speech = np.array([[1, 800, 3200], [1, 800, 8000]])


# Ads
input_shapes_ads = np.array(
    [
        [16, 256, 1500],
        [16, 256, 1567],
        [1, 128, 2876],
        [16, 128, 1567],
        [1, 128, 2722],
        [16, 256, 512],
    ]
)


def main():
    parser = GetArgumentParser()
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(
        [
            "dnnlowp_fc_perf_bench",
            "--caffe2_logging_operator_dyno_sampling_rate=0",
        ]
        + extra_args
    )
    if args.all_shapes:
        for input_shape in input_shapes_nmt:
            compare_fcs(input_shape[0], input_shape[2], input_shape[1], args)
        for input_shape in input_shapes_speech:
            compare_fcs(input_shape[0], input_shape[2], input_shape[1], args)
        for input_shape in input_shapes_ads:
            compare_fcs(input_shape[0], input_shape[2], input_shape[1], args)
    else:
        compare_fcs(args.batch_size, args.input_dim, args.output_dim, args)


def compare_fcs(M, K, N, args):
    X = np.random.rand(M, K).astype(np.float32) - 0.5
    W = np.random.rand(N, K).astype(np.float32) - 0.5
    b = np.random.rand(N).astype(np.float32) - 0.5

    def fc(X, W, b):
        return np.dot(X, np.transpose(W)) + b

    ground = np.array(fc(X, W, b))
    Y_scale = (ground.max() - ground.min()) / 255
    print("min ", ground.min(), " max ", ground.max(), " scale ", Y_scale)
    print("l3_cache_size ", args.l3_cache_size * 4 / 2 ** 20, "MB")
    workspace.FeedBlob("X", X)
    workspace.FeedBlob("W", W)
    workspace.FeedBlob("WT", W.T)
    workspace.FeedBlob("b", b)
    workspace.FeedBlob(
        "huge_blob", np.random.randn(args.l3_cache_size).astype(np.float32)
    )

    net = core.Net("test")

    net.FC(["X", "W", "b"], "Y_default")
    net.FCTransposed(["X", "WT", "b"], "Y_pretranspose")

    if args.quantize_input:
        quantize_X = core.CreateOperator("Quantize", ["X"], ["X_q"], engine="DNNLOWP")
        net.Proto().op.extend([quantize_X])
        quantize_W = core.CreateOperator("Quantize", ["W"], ["W_q"], engine="DNNLOWP")
        net.Proto().op.extend([quantize_W])

    fc_i8_rowwise = core.CreateOperator(
        "Int8FCRowWise",
        ["X_q", "W", "b"] if args.quantize_input else ["X", "W", "b"],
        "Y_rowwise_dnnlowp",
        dequantize_output=0 if args.quantize_output else 1,
        Y_scale=Y_scale,
        Y_zero_point=0,
        engine="DNNLOWP",
    )
    net.Proto().op.extend([fc_i8_rowwise])

    fc_i8 = core.CreateOperator(
        "Int8FC",
        ["X_q", "W_q", "b"] if args.quantize_input else ["X", "W", "b"],
        "Y_dnnlowp",
        dequantize_output=0 if args.quantize_output else 1,
        Y_scale=Y_scale,
        Y_zero_point=0,
        engine="DNNLOWP",
    )
    net.Proto().op.extend([fc_i8])

    pack_w = core.CreateOperator("FbGemmPack", ["W"], "W_packed")
    net.Proto().op.extend([pack_w])
    fc_fp16 = core.CreateOperator("FbFCPacked", ["X", "W_packed", "b"], ["Y_fp16"])
    net.Proto().op.extend([fc_fp16])

    ops = [op for op in net.Proto().op]
    del net.Proto().op[:]
    for op in ops:
        net.Proto().op.extend([op])
        # wipe caches
        net.Scale("huge_blob", "huge_blob_2x", value=2.0)

    workspace.CreateNet(net)
    workspace.RunNet(net)

    # makes sure that results are good.
    outputs = [op.output[0] for op in net.Proto().op if "FC" in op.type]
    for Y in outputs:
        if "i8" in Y or "fp16" in Y or "dnnlowp" in Y:
            continue
        np.testing.assert_allclose(
            workspace.FetchBlob(outputs[0]),
            workspace.FetchBlob(Y),
            atol=1e-2,
            rtol=1e-2,
        )

    runtimes = workspace.BenchmarkNet(
        net.Name(), 1, args.num_runs, True  # warmup  # run induvidual ops
    )[1:]

    results = {
        op.output[0]: runtime
        for op, runtime in zip(net.Proto().op, runtimes)
        if "FC" in op.type
    }

    def get_gflops(time, m, k, n):
        return round(m * n * k * 2 / time / 10 ** 6 * 10) / 10

    results = [
        (out, time, "{} GFLOPS".format(get_gflops(time, M, K, N)))
        for out, time in results.items()
    ]
    # results = sorted(results, key=operator.itemgetter(1))

    print("input shape M, N, K: {} {} {}".format(M, N, K))
    for output, time, flops in results:
        print("{}: {:.4f} ms {}".format(output, time, flops))


if __name__ == "__main__":
    main()
