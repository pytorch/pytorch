from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import time

import numpy as np
from caffe2.python import core, dyndep, workspace


DTYPES = {"float": np.float32, "float16": np.float16}

dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


def benchmark_sparse_lengths_sum(
    dtype_str,
    categorical_limit,
    embedding_dim,
    average_len,
    batch_size,
    iterations,
    engine,
    num_workers,
):
    print("Preparing lookup table. " + str(datetime.datetime.now()))

    # We will use a constant, but non-trivial value so we save initialization
    # time.
    data = np.ones([categorical_limit, embedding_dim], dtype=np.float32)
    data *= 17.01

    print("Data has shape {} {}".format(data.shape, datetime.datetime.now()))
    workspace.FeedBlob("X", data.astype(DTYPES[dtype_str]))

    # Set random seed, so that repeated runs will keep the same sequence of
    # random indices.
    np.random.seed(1701)

    lengths = np.random.randint(
        int(average_len * 0.75), int(average_len * 1.25), batch_size
    ).astype(np.int32)
    indices = np.random.randint(0, categorical_limit, np.sum(lengths)).astype(np.int32)
    workspace.FeedBlob("indices", indices)
    workspace.FeedBlob("lengths", lengths)

    net = core.Net("mynet")
    net.Proto().type = (
        "async_scheduling" if engine == "INTRA_OP_PARALLEL" else "parallel"
    )
    net.Proto().num_workers = num_workers
    net.SparseLengthsSum(["X", "indices", "lengths"], "Y", engine=engine)

    workspace.CreateNet(net)

    print("Preparation finished. " + str(datetime.datetime.now()))

    niter_outer = 4

    bytes_per_iteration = (
        embedding_dim * len(indices) * np.dtype(DTYPES[dtype_str]).itemsize
        + indices.nbytes
        + lengths.nbytes
    )

    for _ in range(niter_outer):
        t = time.time()
        workspace.RunNet(net, iterations)
        dt = time.time() - t
        print(
            "Intra-op parallel SparseLengthsSum effective BW: {} GB/s".format(
                iterations * bytes_per_iteration / dt / 1e9
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="minimal benchmark for sparse lengths sum."
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=list(DTYPES.keys()),
        default="float",
        help="The data type for the input lookup table.",
    )
    parser.add_argument(
        "-e", "--embedding-size", type=int, default=6000000, help="Lookup table size."
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--average-len",
        type=int,
        default=27,
        help="Sparse feature average lengths, default is 27",
    )
    parser.add_argument("--batch-size", type=int, default=4096, help="The batch size.")
    parser.add_argument(
        "-i", "--iteration", type=int, default=16, help="The number of iterations."
    )
    parser.add_argument("--engine", type=str, default="TBB")
    parser.add_argument("--num-workers", type=int, default=1)
    args, extra_args = parser.parse_known_args()
    global_options = ["caffe2", "--caffe2_cpu_numa_enabled=1"]
    if args.engine == "TBB":
        global_options += ["--caffe2_task_graph_engine=tbb"]
    workspace.GlobalInit(global_options + extra_args)
    assert workspace.IsNUMAEnabled()
    benchmark_sparse_lengths_sum(
        args.dtype,
        args.embedding_size,
        args.embedding_dim,
        args.average_len,
        args.batch_size,
        args.iteration,
        args.engine,
        args.num_workers,
    )
