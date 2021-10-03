#!/usr/bin/env python3





import string

import argparse

import numpy as np

from caffe2.python.model_helper import ModelHelper
from caffe2.python.predictor import mobile_exporter
from caffe2.python import core, workspace, brew, utils


def parse_kwarg(kwarg_str):
    key, value = map(string.strip, kwarg_str.split("=", 1))
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return key, value


def main(args):
    # User defined keyword arguments
    kwargs = {"order": "NCHW"}
    kwargs.update(dict(args.kwargs))

    model = ModelHelper(name=args.benchmark_name)

    op_type = args.operator  # assumes a brew type op name
    input_name = args.input_name
    output_name = args.output_name

    iters = int(args.iters)
    for i in range(iters):
        input_blob_name = input_name + (str(i) if i > 0 and args.chain else '')
        output_blob_name = output_name + str(i + 1)
        add_op = getattr(brew, op_type)
        add_op(model, input_blob_name, output_blob_name, **kwargs)
        if args.chain:
            input_name, output_name = output_name, input_name

    workspace.RunNetOnce(model.param_init_net)
    extra_init_net_ops = []

    def make_blob_on_context(blob_name, blob_data, context):
        if context.upper() != "CPU":
            blob_name_modified = "{}_CPU".format(blob_name)
        else:  # CPU case is simple
            blob_name_modified = blob_name

        fill_op = core.CreateOperator(
            "GivenTensorFill", [], [blob_name_modified],
            arg=[
                utils.MakeArgument("shape", blob_data.shape),
                utils.MakeArgument("values", blob_data)
            ]
        )
        extra_init_net_ops.append(fill_op)

        # We need to create CPU blobs and add some copy operations in
        # the init_net
        if context.upper() == "OPENGL":
            copy_op = core.CreateOperator("CopyToOpenGL", [blob_name_modified],
                                          [blob_name])
            extra_init_net_ops.append(copy_op)

    for unparsed_blob in args.blob:
        name, unparsed_dims = unparsed_blob.split('=')
        dims = [int(d) for d in unparsed_dims.split(',')]
        np_input = np.random.rand(*dims).astype(np.float32)
        make_blob_on_context(name, np_input, args.context)

    init_net, predict_net = mobile_exporter.Export(
        workspace, model.net, model.params
    )
    init_net.op.extend(extra_init_net_ops)

    # Handle manual rewrite
    if args.context.upper() == "OPENGL":
        old_ops = [op for op in predict_net.op]
        del predict_net.op[:]
        for op in old_ops:
            op.type = 'OpenGL{}'.format(op.type)
        predict_net.op.extend(old_ops)

    if args.debug:
        print("init_net:")
        for op in init_net.op:
            print(" ", op.type, op.input, "-->", op.output)
        print("predict_net:")
        for op in predict_net.op:
            print(" ", op.type, op.input, "-->", op.output)

    with open(args.predict_net, 'wb') as f:
        f.write(predict_net.SerializeToString())
    with open(args.init_net, 'wb') as f:
        f.write(init_net.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilitity to generate Caffe2 benchmark models.")
    parser.add_argument("operator", help="Caffe2 operator to benchmark.")
    parser.add_argument("-b", "--blob",
                        help="Instantiate a blob --blob name=dim1,dim2,dim3",
                        action='append')
    parser.add_argument("--context", help="Context to run on.", default="CPU")
    parser.add_argument("--kwargs", help="kwargs to pass to operator.",
                        nargs="*", type=parse_kwarg, default=[])
    parser.add_argument("--init_net", help="Output initialization net.",
                        default="init_net.pb")
    parser.add_argument("--predict_net", help="Output prediction net.",
                        default="predict_net.pb")
    parser.add_argument("--benchmark_name",
                        help="Name of the benchmark network",
                        default="benchmark")
    parser.add_argument("--input_name", help="Name of the input blob.",
                        default="data")
    parser.add_argument("--output_name", help="Name of the output blob.",
                        default="output")
    parser.add_argument("--iters",
                        help="Number of iterations to run the operator.",
                        default="1")
    parser.add_argument("-d", "--debug", help="Print debug information.",
                        action='store_true')
    parser.add_argument("-c", "--chain",
                        help="Chain ops together (create data dependencies)",
                        action='store_true')
    args = parser.parse_args()
    main(args)
