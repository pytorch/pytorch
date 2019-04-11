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
import copy
import time

import libfb.py.mkl
from caffe2.proto import caffe2_pb2
from caffe2.python import core, dyndep, model_helper, workspace
from caffe2.python.fb.net_transforms.mutable_proto import (
    MutableInitTrainProtos,
    MutableNetProto,
)
from caffe2.python.fb.net_transforms.parallelize import (
    PARALLEL_DEVICE,
    Partition,
    parallelize_net,
)


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


# Build intra_op_parallel benchmark net with numa setting
def build_model(net_name, engine):
    model = model_helper.ModelHelper(net_name)

    # Create a serial net
    init_net = core.Net(net_name + "_init")
    net = core.Net(net_name)

    par_dev = core.DeviceOption(caffe2_pb2.CPU, PARALLEL_DEVICE)

    desired_blob_states = {}
    with core.DeviceScope(par_dev):
        input_blob_name = "activation_{}".format(engine)
        init_net.XavierFill([], input_blob_name, shape=[args.batch_size, args.dim])
        # desired_blob_states[input_blob_name] = BlobState.REPLICATED

        for layer in range(args.num_layers):
            output_dim = args.dim if (layer < args.num_layers - 1) else 1
            weight_blob_name = "weight_{}_{}".format(layer, engine)
            bias_blob_name = "bias_{}_{}".format(layer, engine)
            output_blob_name = "activation_{}_{}".format(layer + 1, engine)

            init_net.XavierFill([], weight_blob_name, shape=[output_dim, args.dim])
            init_net.XavierFill([], bias_blob_name, shape=[output_dim])
            # desired_blob_states[input_blob_name] = BlobState.REPLICATED
            # desired_blob_states[bias_blob_name] = BlobState.REPLICATED

            net.FC(
                [input_blob_name, weight_blob_name, bias_blob_name],
                [output_blob_name],
                engine=engine,
            )

            input_blob_name = output_blob_name

    model.param_init_net.AppendNet(init_net)
    model.net.AppendNet(net)

    # Add gradient operators
    if not args.forward_only:
        model.AddGradientOperators(
            ["activation_{}_{}".format(args.num_layers, engine)]
        )

    worker_devs = [
        core.DeviceOption(
            caffe2_pb2.CPU,
            numa_node_id=i,
            node_name=None
            if engine != "" or args.num_workers == 1
            else "first_step",
        )
        for i in range(args.num_numa_nodes)
    ]

    protos = MutableInitTrainProtos(
        MutableNetProto(model.param_init_net.Proto()),
        MutableNetProto.clone_from_net(model.net),
    )

    protos = parallelize_net(
        protos,
        par_dev=core.DeviceOption(caffe2_pb2.CPU, PARALLEL_DEVICE),
        worker_devs=worker_devs,
        desired_blob_states=desired_blob_states,
    )

    par_init_net, par_net = protos.to_nets()

    if engine == "" and args.num_workers > 1:
        from caffe2.python.fb.net_transforms import multi_parallelize

        intra_partitions = []
        for worker_dev in worker_devs:
            intra_worker_dev = copy.deepcopy(worker_dev)
            intra_worker_dev.ClearField("node_name")
            intra_partitions.append(
                Partition(
                    par_dev=worker_dev,
                    worker_devs=[intra_worker_dev] * args.num_workers,
                )
            )
        intra_multi_partition = multi_parallelize.MultiPartition(intra_partitions)

        def clear_node_names(net):
            for op in net.Proto().op:
                if (
                    op.type not in ["FC", "FCGradient"]
                    and op.device_option.node_name is not None
                ):
                    op.device_option.ClearField("node_name")

        clear_node_names(par_init_net)
        clear_node_names(par_net)

        par_init_net = multi_parallelize.parallelize_net(
            intra_multi_partition, MutableNetProto.clone_from_net(par_init_net)
        ).to_net()
        par_net = multi_parallelize.parallelize_net(
            intra_multi_partition, MutableNetProto.clone_from_net(par_net)
        ).to_net()

    if args.print_model:
        print(par_init_net.Proto())
        print(par_net.Proto())

    return par_init_net, par_net


# Parse command line arguments
parser = argparse.ArgumentParser(description="Intra-op parallel FC operator benchmark")
parser.add_argument("--num-numa-nodes", type=int, default=1)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1024,
    help="total batch size across all numa nodes",
)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--forward-only", action="store_true")
parser.add_argument("--print-model", action="store_true")
parser.add_argument("--niter-outer", type=int, default=2)
parser.add_argument("--niter-inner", type=int, default=16)
parser.add_argument("--engine", type=str, default="TBB")
args, extra_args = parser.parse_known_args()

# Set global options
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

# Create nets
baseline_init_net, baseline_net = build_model("baseline", "")
intra_op_parallel_init_net, intra_op_parallel_net = build_model(
    "intra_op_parallel", args.engine
)
for net in [
    baseline_init_net,
    baseline_net,
    intra_op_parallel_init_net,
    intra_op_parallel_net,
]:
    net.Proto().num_workers = args.num_workers
    net.Proto().type = (
        "async_scheduling" if args.engine == "INTRA_OP_PARALLEL" else "parallel"
    )

workspace.RunNetOnce(baseline_init_net)
workspace.RunNetOnce(intra_op_parallel_init_net)

workspace.CreateNet(baseline_net)
workspace.CreateNet(intra_op_parallel_net)

# Run the actual benchmark and report performance

# FC multiplies a (batch_size x dim) matrix with a (dim x dim) matrix,
# except the last FC layer that multiplies a (batch_size x dim) matrix with a
# (dim x 1) matrix.
# FCGradient does two such matrix multiplications
flops_per_iteration = (
    (1.0 if args.forward_only else 3.0)
    * 2
    * args.batch_size
    * ((args.num_layers - 1) * args.dim + 1)
    * args.dim
)

for _ in range(args.niter_outer):
    t = time.time()
    workspace.RunNet(baseline_net, args.niter_inner)
    dt = time.time() - t
    print(
        "Data-parallel net transform time {}, aggregated {} GF/s".format(
            dt, args.niter_inner * flops_per_iteration / dt / 1e9
        )
    )

    t = time.time()
    workspace.RunNet(intra_op_parallel_net, args.niter_inner)
    dt = time.time() - t
    print(
        "Intra-op parallel time {}, aggregated {} GF/s:".format(
            dt, args.niter_inner * flops_per_iteration / dt / 1e9
        )
    )
