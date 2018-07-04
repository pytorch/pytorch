from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import datetime

from caffe2.python import core, workspace

DTYPES = {
    'uint8': np.uint8,
    'uint8_fused': np.uint8,
    'float': np.float32,
    'float16': np.float16,
}


def benchmark_sparse_lengths_sum(
        dtype_str,
        categorical_limit,
        embedding_size,
        average_len,
        batch_size,
        iterations):
    print('Preparing lookup table. ' + str(datetime.datetime.now()))

    # We will use a constant, but non-trivial value so we save initialization
    # time.
    data = np.ones([categorical_limit, embedding_size], dtype=np.float32)
    data *= 17.01

    if dtype_str == 'uint8':
        scale_bias = np.random.rand(categorical_limit, 2).astype(np.float32)
        workspace.FeedBlob("scale_bias", scale_bias.astype(np.float32))
    elif dtype_str == 'uint8_fused':
        scale_bias = np.random.randint(255, size=(categorical_limit, 8))
        data = np.concatenate([data, scale_bias], axis=1)

    print('Data has shape {} {}'.format(data.shape, datetime.datetime.now()))
    workspace.FeedBlob("X", data.astype(DTYPES[dtype_str]))

    # In order to produce truly random lengths and indices, we will embed a
    # Python operator in the net to generate them.
    def f(_, outputs):
        lengths = np.random.randint(
            int(average_len * 0.75),
            int(average_len * 1.25),
            batch_size).astype(np.int32)
        indices = np.random.randint(
            0, categorical_limit, np.sum(lengths)).astype(np.int64)
        outputs[0].feed(indices)
        outputs[1].feed(lengths)

    net = core.Net("mynet")
    net.Python(f)([], ["indices", "lengths", ])
    if dtype_str == "uint8":
        net.SparseLengthsSum8BitsRowwise(["X", "indices", "lengths", "scale_bias"], "Y")
    elif dtype_str == "uint8_fused":
        net.SparseLengthsSumFused8BitRowwise(["X", "indices", "lengths"], "Y")
    else:
        net.SparseLengthsSum(["X", "indices", "lengths"], "Y")
    workspace.CreateNet(net)

    # Set random seed, so that repeated runs will keep the same sequence of
    # random indices.
    np.random.seed(1701)

    print('Preparation finished. ' + str(datetime.datetime.now()))

    workspace.BenchmarkNet(net.Name(), 1, iterations, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum.")
    parser.add_argument(
        '-d', "--dtype", choices=list(DTYPES.keys()), default="float",
        help="The data type for the input lookup table.")
    parser.add_argument(
        '-e', "--embedding-size", type=int, default=6000000,
        help="Lookup table size.")
    parser.add_argument(
        "--embedding-dim", type=int, default=128,
        help="Embedding dimension.")
    parser.add_argument(
        "--average_len", type=int, default=27,
        help="Sparse feature average lengths, default is 27")
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="The batch size.")
    parser.add_argument(
        '-i', "--iteration", type=int, default=100000,
        help="The number of iterations.")
    args, extra_args = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_sparse_lengths_sum(
        args.dtype,
        args.embedding_size,
        args.embedding_dim,
        args.average_len,
        args.batch_size,
        args.iteration)
