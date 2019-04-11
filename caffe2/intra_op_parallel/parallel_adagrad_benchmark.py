from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import time

import numpy as np
from caffe2.python import core, dyndep, workspace


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


def benchmark_parallel_adagrad(args, engine):
    h = args.nrows
    w = args.embedding_dim
    n = args.ninputs

    param = np.zeros([h, w]).astype(np.float32)
    momentum = np.zeros([h, w]).astype(np.float32)
    lr = np.asarray([1]).astype(np.float32)

    if args.distribution == "zipf":
        a = np.minimum(np.random.zipf(2.0, n) * h, h - 1)
    else:
        a = np.random.rand(n) * (h - 1)
    indices = np.round(a).astype(np.int)
    grads = np.zeros([h, w]).astype(np.float32)
    sparse_grads = np.zeros([n, w]).astype(np.float32)

    workspace.FeedBlob("param", param)
    workspace.FeedBlob("momentum", momentum)
    workspace.FeedBlob("momentum_row_wise", momentum[:, 0])
    workspace.FeedBlob("indices", indices)
    workspace.FeedBlob("grads", grads)
    workspace.FeedBlob("sparse_grads", sparse_grads)
    workspace.FeedBlob("lr", lr)

    adagrad_net = core.Net("adagrad_net")
    adagrad_net.Adagrad(
        ["param", "momentum", "grads", "lr"], ["param", "momentum"], engine=engine
    )

    sparse_adagrad_net = core.Net("sparse_adagrad_net")
    sparse_adagrad_net.SparseAdagrad(
        ["param", "momentum", "indices", "sparse_grads", "lr"],
        ["param", "momentum"],
        engine=engine,
    )

    row_wise_sparse_adagrad_net = core.Net("row_wise_sparse_adagrad_net")
    row_wise_sparse_adagrad_net.RowWiseSparseAdagrad(
        ["param", "momentum_row_wise", "indices", "sparse_grads", "lr"],
        ["param", "momentum_row_wise"],
        engine=engine,
    )

    for net in [adagrad_net, sparse_adagrad_net, row_wise_sparse_adagrad_net]:
        net.Proto().type = (
            "async_scheduling" if engine == "INTRA_OP_PARALLEL" else "parallel"
        )
        net.Proto().num_workers = args.num_workers
        workspace.CreateNet(net)

    nwarmup = 1
    niter = 10

    # *2 for reading and writing
    bytes_per_adagrad_iteration = 2 * (param.nbytes + momentum.nbytes) + grads.nbytes
    runtimes = workspace.BenchmarkNet(adagrad_net.Name(), nwarmup, niter, True)
    print(
        "Intra-op parallel Adagrad effective BW: {} GB/s".format(
            bytes_per_adagrad_iteration / runtimes[0] / 1e6
        )
    )

    # First 4 is for reading/writing param/momentum
    # Second 4 is for sizeof(float)
    bytes_per_sparse_adagrad_iteration = (
        4 * w * len(indices) * 4 + indices.nbytes + sparse_grads.nbytes
    )
    runtimes = workspace.BenchmarkNet(sparse_adagrad_net.Name(), nwarmup, niter, True)
    print(
        "Intra-op parallel SparseAdagrad effective BW: {} GB/s".format(
            bytes_per_sparse_adagrad_iteration / runtimes[0] / 1e6
        )
    )

    # *2 for reading/writing param/momentum
    # *4 for sizeof(float)
    bytes_per_row_wise_sparse_adagrad_iteration = (
        2 * (w + 1) * len(indices) * 4 + indices.nbytes + sparse_grads.nbytes
    )
    runtimes = workspace.BenchmarkNet(row_wise_sparse_adagrad_net.Name(), nwarmup, niter, True)
    print(
        "Intra-op parallel RowWiseSparseAdagrad effective BW: {} GB/s".format(
            bytes_per_row_wise_sparse_adagrad_iteration / runtimes[0] / 1e6
        )
    )


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser(
        description="Benchmark intra op parallel SparseAdagrad"
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--nrows", type=int, default=1024 * 1024)
    parser.add_argument("--ninputs", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--engine", type=str, default="TBB")
    args, extra_args = parser.parse_known_args()

    global_options = [
        "caffe2",
        "--caffe2_cpu_numa_enabled=1",
        "--caffe2_intra_op_parallel_max_num_tasks={}".format(args.num_workers),
    ]
    if args.engine == "TBB":
        global_options += ["--caffe2_task_graph_engine=tbb"]

    if extra_args:
        global_options += extra_args
    workspace.GlobalInit(global_options)
    assert workspace.IsNUMAEnabled()

    benchmark_parallel_adagrad(args, args.engine)
