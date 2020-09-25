




from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.models.download import ModelDownloader
import numpy as np
import argparse
import time
import os.path


def GetArgumentParser():
    parser = argparse.ArgumentParser(description="Caffe2 benchmark.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size."
    )
    parser.add_argument("--model", type=str, help="The model to benchmark.")
    parser.add_argument(
        "--order",
        type=str,
        default="NCHW",
        help="The order to evaluate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="device to evaluate on."
    )
    parser.add_argument(
        "--cudnn_ws",
        type=int,
        help="The cudnn workspace size."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run the network."
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warm-up iterations before benchmarking."
    )
    parser.add_argument(
        "--forward_only",
        action='store_true',
        help="If set, only run the forward pass."
    )
    parser.add_argument(
        "--layer_wise_benchmark",
        action='store_true',
        help="If True, run the layer-wise benchmark as well."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="",
        help="If set, blindly prefer the given engine(s) for every op.")
    parser.add_argument(
        "--dump_model",
        action='store_true',
        help="If True, dump the model prototxts to disk."
    )
    parser.add_argument("--net_type", type=str, default="simple")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use-nvtx", default=False, action='store_true')
    parser.add_argument("--htrace_span_log_path", type=str)
    return parser


def benchmark(args):
    print('Batch size: {}'.format(args.batch_size))
    mf = ModelDownloader()
    init_net, pred_net, value_info = mf.get_c2_model(args.model)
    input_shapes = {k : [args.batch_size] + v[-1][1:] for (k, v) in value_info.items()}
    print("input info: {}".format(input_shapes))
    external_inputs = {}
    for k, v in input_shapes.items():
        external_inputs[k] = np.random.randn(*v).astype(np.float32)

    if args.device == 'CPU':
        device_option = core.DeviceOption(caffe2_pb2.CPU)
    elif args.device == 'MKL':
        device_option = core.DeviceOption(caffe2_pb2.MKLDNN)
    elif args.device == 'IDEEP':
        device_option = core.DeviceOption(caffe2_pb2.IDEEP)
    else:
        raise Exception("Unknown device: {}".format(args.device))
    print("Device option: {}, {}".format(args.device, device_option))
    pred_net.device_option.CopyFrom(device_option)
    for op in pred_net.op:
        op.device_option.CopyFrom(device_option)

    # Hack to initialized weights into MKL/IDEEP context
    workspace.RunNetOnce(init_net)
    bb = workspace.Blobs()
    weights = {}
    for b in bb:
        weights[b] = workspace.FetchBlob(b)
    for k, v in external_inputs.items():
        weights[k] = v
    workspace.ResetWorkspace()

    with core.DeviceScope(device_option):
        for name, blob in weights.items():
            #print("{}".format(name))
            workspace.FeedBlob(name, blob, device_option)
        workspace.CreateNet(pred_net)
        start = time.time()
        res = workspace.BenchmarkNet(pred_net.name,
                                     args.warmup_iterations,
                                     args.iterations,
                                     args.layer_wise_benchmark)
        print("FPS: {:.2f}".format(1/res[0]*1000*args.batch_size))

if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()
    if (
        not args.batch_size or not args.model or not args.order
    ):
        GetArgumentParser().print_help()
    benchmark(args)
