from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import google.protobuf.text_format as protobuftx

from caffe2.python import workspace, core

from caffe2.proto import caffe2_pb2
from caffe2.python import model_helper


def extract(args):
    # Load
    orig_net = caffe2_pb2.NetDef()
    protobuftx.Merge(open(args.net, "r").read(), orig_net)
    predict_net = model_helper.ExtractPredictorNet(
        orig_net,
        args.input_blobs,
        args.output_blobs,
        device=core.DeviceOption(caffe2_pb2.CPU, 0),
        renames={},
        disabled=args.disabled_blobs,
    )

    open(args.net + ".converted", "w").write(str(predict_net.Proto()))
    print("Converted model:" + args.net + ".converted")


def main():
    parser = argparse.ArgumentParser(
        description="Caffe2: Extract predictor models from nets"
    )
    parser.add_argument("--net", type=str, required=True,
                        help="Path to the net proto")
    parser.add_argument("--output_blobs", type=str, nargs='*',
                        default=[])
    parser.add_argument("--input_blobs", type=str, nargs='*',
                        default=[])
    parser.add_argument("--disabled_blobs", type=str, nargs='*',
                        default=[])
    args = parser.parse_args()

    extract(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
