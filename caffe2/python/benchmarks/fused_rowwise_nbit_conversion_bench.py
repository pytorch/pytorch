from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime

import numpy as np
from caffe2.fb.python.operator_test.fused_nbit_rowwise_test_helper import (
    _compress_uniform_simplified,
    param_search_greedy,
)
from caffe2.python import core, dyndep, workspace


def main(bit_rate):
    # uncomment for debugging
    # np.random.seed(0)
    batchsize = 10 * 1000
    blocksize = 64
    print(batchsize, blocksize)
    input_data = np.random.rand(batchsize, blocksize).astype(np.float32)

    workspace.FeedBlob("input_data", input_data)

    net = core.Net("bench")
    op = core.CreateOperator(
        "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
        "input_data",
        "quantized_data",
        engine="GREEDY",
    )
    net.Proto().op.extend([op])
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
    workspace.CreateNet(net)
    iterations = 10
    workspace.BenchmarkNet(net.Proto().name, 1, iterations, True)

    net2 = core.Net("bench2")
    op = core.CreateOperator(
        "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
        "input_data",
        "quantized_data",
    )
    net2.Proto().op.extend([op])

    workspace.CreateNet(net2)
    workspace.BenchmarkNet(net2.Proto().name, 1, iterations, True)

    start = datetime.datetime.now()
    for n in range(10):
        for i in range(input_data.shape[0]):
            xmin, xmax = param_search_greedy(
                input_data[i, :], 4, n_bins=200, ratio=0.16
            )
            X_q_ref = _compress_uniform_simplified(input_data[i, :], 4, xmin, xmax)
    end = datetime.datetime.now()
    print((end - start).total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark for row-wise 2/4-bit quantization."
    )
    parser.add_argument("--bit-rate", type=int, default=4)
    args = parser.parse_args()
    main(args.bit_rate)
