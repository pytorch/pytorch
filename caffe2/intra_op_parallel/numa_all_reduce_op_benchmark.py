# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import time

from caffe2.proto import caffe2_pb2
from caffe2.python import core, dyndep, workspace


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


# Parse command line arguments
parser = argparse.ArgumentParser(description="NUMA all-reduce benchmark")
parser.add_argument("--num-numa-nodes", type=int, default=1)
parser.add_argument("--len", type=int, default=33554432)  # 32M
parser.add_argument("--enable-numa", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--engine", type=str, default="TBB")
args, extra_args = parser.parse_known_args()

# Set global options
global_options = ["caffe2"]
if args.enable_numa:
    global_options += ["--caffe2_cpu_numa_enabled=1"]
if extra_args:
    global_options += extra_args
core.GlobalInit(global_options)
if args.enable_numa:
    assert workspace.IsNUMAEnabled()
assert args.num_numa_nodes <= workspace.GetNumNUMANodes()

# Create nets with appropriate type and num_workers
init_net = core.Net("allreduce_bench_init")
init_net.Proto().type = "async_scheduling"
init_net.Proto().num_workers = args.num_workers

net = core.Net("allreduce_bench")
net.Proto().type = "async_scheduling" if args.engine == "" else "parallel"
net.Proto().num_workers = args.num_workers

# Initialize weights with numa aware allocation (if the option is on)
blob_names = []
for numa_node_id in range(args.num_numa_nodes):
    numa_device_option = caffe2_pb2.DeviceOption()
    numa_device_option.device_type = caffe2_pb2.CPU
    numa_device_option.numa_node_id = numa_node_id

    blob_name = "weight_{}".format(numa_node_id)
    blob_names.append(blob_name)

    init_net.XavierFill(
        [], blob_name, shape=[args.len], device_option=numa_device_option
    )
workspace.RunNetOnce(init_net)

# Create net with all-reduce operator
net.NUMAAllReduce(blob_names, blob_names, engine=args.engine)
workspace.CreateNet(net)

# Run the actual benchmark and report performance
niter_outer = 4
niter_inner = 128

bytes_per_iteration = 2.0 * 4 * args.len * (args.num_numa_nodes - 1)

for _ in range(niter_outer):
    t = time.time()
    workspace.RunNet(net.Name(), niter_inner)
    dt = time.time() - t
    print(
        "All-reduce aggregate effective BW: {} GB/s".format(
            niter_inner * bytes_per_iteration / dt / 1e9
        )
    )
