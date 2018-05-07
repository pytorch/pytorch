from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.models.download import downloadFromURLToFile
import numpy as np
import argparse
import time
import os.path
import json

class ModelFetcher:
    def _model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('CAFFE2_HOME', '~/.caffe2'))
        models_dir = os.getenv('CAFFE2_MODELS', os.path.join(caffe2_home, 'models'))
        return os.path.join(models_dir, model)

    def _download(self, model):
        model_dir = self._model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'init_net.pb', 'value_info.json']:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                try:
                    downloadFromURLToFile(url, dest,
                                          show_progress=False)
                except TypeError:
                    # show_progress not supported prior to
                    # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                    # (Sep 17, 2017)
                    downloadFromURLToFile(url, dest)
            except Exception as e:
                print("Abort: {reason}".format(reason=e))
                print("Cleaning up...")
                deleteDirectory(model_dir)
                exit(1)

    def get_c2_model(self, model_name):
        model_dir = self._model_dir(model_name)
        if not os.path.exists(model_dir):
            self._download(model_name)
        c2_predict_pb = os.path.join(model_dir, 'predict_net.pb')
        c2_predict_net = caffe2_pb2.NetDef()
        with open(c2_predict_pb, 'rb') as f:
            c2_predict_net.ParseFromString(f.read())
        c2_predict_net.name = model_name

        c2_init_pb = os.path.join(model_dir, 'init_net.pb')
        c2_init_net = caffe2_pb2.NetDef()
        with open(c2_init_pb, 'rb') as f:
            c2_init_net.ParseFromString(f.read())
        c2_init_net.name = model_name + '_init'

        value_info = json.load(open(os.path.join(model_dir, 'value_info.json')))
        return c2_init_net, c2_predict_net, value_info

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
    mf = ModelFetcher()
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
        res = workspace.BenchmarkNet(pred_net.name, args.warmup_iterations, args.iterations, False)
        print("FPS: {:.2f}".format(1/res[0]*1000*args.batch_size))

if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()
    if (
        not args.batch_size or not args.model or not args.order
    ):
        GetArgumentParser().print_help()
    benchmark(args)
