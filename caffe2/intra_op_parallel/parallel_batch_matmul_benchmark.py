from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import inspect
import time

import libfb.py.mkl
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")

parser = argparse.ArgumentParser(
    description="Caffe2 benchmark. Extra args will be passed to Caffe2"
)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--engine", type=str, default="TBB")
args, extra_args = parser.parse_known_args()
global_options = [
    "parallel_batch_matmul_benchmark",
    "--caffe2_cpu_numa_enabled=1",
]
if args.engine == "TBB":
    global_options += ["--caffe2_task_graph_engine=tbb"]
workspace.GlobalInit(global_options + extra_args)
assert workspace.IsNUMAEnabled()

M = N = 512  # num of embedding tables
K = 512  # Embedding dim
C = 20  # Batch size
trans_a = 0
trans_b = 0
dtype = np.float32

X = np.random.rand(C, M, K).astype(dtype) - 0.5
if trans_a:
    X = X.swapaxes(-1, -2)
Y = np.random.rand(C, K, N).astype(dtype) - 0.5
if trans_b:
    Y = Y.swapaxes(-1, -2)

workspace.FeedBlob("X", X)
workspace.FeedBlob("Y", Y)

net = core.Net("test_net")
net.BatchMatMul(
    ["X", "Y"], "out", trans_a=trans_a, trans_b=trans_b, engine=args.engine
)
net.Proto().type = "async_scheduling" if args.engine == "INTRA_OP_PARALLEL" else "parallel"
net.Proto().num_workers = args.num_workers
workspace.CreateNet(net)

ref_net = core.Net("ref_test_net")
ref_net.BatchMatMul(["X", "Y"], "out_ref", trans_a=trans_a, trans_b=trans_b)
workspace.CreateNet(ref_net)

nwarmup = 1
niter = 10
runtimes = workspace.BenchmarkNet(net.Name(), nwarmup, niter, True)

runtimes_ref = workspace.BenchmarkNet(ref_net.Name(), nwarmup, niter, True)

gflops_per_iteration = M * N * K * C * 2 * 1e-9

print(
    "Intra-op parallel BatchMatMul: {} ms/iter with {} workers; {} GF/s".format(
        runtimes[0], args.num_workers, gflops_per_iteration / runtimes[0] * 1e3
    )
)
print(
    "Reference BatchMatMul time: {} ms/iter; {} GF/s".format(
        runtimes_ref[0], gflops_per_iteration / runtimes_ref[0] * 1e3
    )
)

output = workspace.FetchBlob("out")
ref_output = workspace.FetchBlob("out_ref")
np.testing.assert_allclose(
    output,
    ref_output,
    atol=1e-4,
    rtol=1e-4,
    err_msg="Out is not matching the reference",
)
