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

from caffe2.intra_op_parallel.parallel_benchmark_utils import parallelize_model
from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, dyndep, model_helper, workspace
from caffe2.python.modeling import parameter_info


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")


# Build intra_op_parallel benchmark net with numa setting
def build_serial_model(net_name, engine):
    model = model_helper.ModelHelper(net_name)

    # Build forward path
    init_net = core.Net(net_name + "_init")
    net = core.Net(net_name)

    default_device_option = caffe2_pb2.DeviceOption(numa_node_id=0)
    input_blob_name = "activation"
    init_net.XavierFill(
        [],
        input_blob_name,
        shape=[args.batch_size, args.dim],
        device_option=default_device_option,
    )

    for layer in range(args.num_layers):
        output_dim = args.dim if (layer < args.num_layers - 1) else 1
        weight_blob_name = "weight_{}".format(layer)
        bias_blob_name = "bias_{}".format(layer)
        output_blob_name = "activation_{}".format(layer + 1)

        weight_blob = init_net.XavierFill(
            [],
            weight_blob_name,
            shape=[output_dim, args.dim],
            device_option=default_device_option,
        )
        bias_blob = init_net.XavierFill(
            [], bias_blob_name, shape=[output_dim], device_option=default_device_option
        )

        net.FC(
            [input_blob_name, weight_blob_name, bias_blob_name],
            [output_blob_name],
            device_option=default_device_option,
        )

        model.AddParameter(weight_blob, parameter_info.ParameterTags.WEIGHT)
        model.AddParameter(bias_blob, parameter_info.ParameterTags.BIAS)

        input_blob_name = output_blob_name

    model.param_init_net.AppendNet(init_net)
    model.net.AppendNet(net)

    if not args.forward_only:
        # Add gradient operators
        model.AddGradientOperators(["activation_{}".format(args.num_layers)])

        # Add adagrad
        iter = brew.iter(model, "iter")
        lr = model.net.LearningRate([iter], "lr", base_lr=1.0, policy="fixed")

        for param in model.GetParams():
            param_grad = model.param_to_grad[param]
            param_momentum = model.param_init_net.ConstantFill(
                [param],
                param + "_momentum",
                value=0.0,
                device_option=default_device_option,
            )

            model.net.Adagrad(
                [param, param_momentum, param_grad, lr],
                [param, param_momentum],
                epsilon=0.9,
                max_num_tasks=4,
                device_option=default_device_option,
            )

    # Set INTRA_OP_PARALLEL engine
    for op in model.net.Proto().op:
        if op.type in ["FC", "FCGradient", "Adagrad"]:
            op.engine = engine

    return model


def build_model(net_name, engine):
    serial_model = build_serial_model(net_name, engine)
    par_init_net, par_net = parallelize_model(serial_model, engine, args.num_numa_nodes)
    if args.print_model:
        print(par_init_net.Proto())
        print(par_net.Proto())
    return par_init_net, par_net


# Parse command line arguments
parser = argparse.ArgumentParser(description="Intra-op parallel MLP operator benchmark")
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
parser.add_argument("--print-model", action="store_true")
parser.add_argument("--niter-outer", type=int, default=2)
parser.add_argument("--niter-inner", type=int, default=16)
parser.add_argument("--engine", type=str, default="TBB")
parser.add_argument("--forward-only", action="store_true")
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
init_net, predict_net = build_model("benchmark", args.engine)
for net in [init_net, predict_net]:
    net.Proto().num_workers = args.num_workers
    net.Proto().type = (
        "async_scheduling" if args.engine == "INTRA_OP_PARALLEL" else "parallel"
    )

workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)

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
    # warm up
    workspace.RunNet(predict_net)
    t = time.time()
    workspace.RunNet(predict_net, args.niter_inner)
    dt = time.time() - t
    print(
        "Intra-op parallel time {}, aggregated {} GF/s:".format(
            dt, args.niter_inner * flops_per_iteration / dt / 1e9
        )
    )
