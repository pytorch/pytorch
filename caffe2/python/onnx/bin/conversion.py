## @package onnx
# Module caffe2.python.onnx.bin.conversion





import json

from caffe2.proto import caffe2_pb2
import click
import numpy as np
from onnx import checker, ModelProto

from caffe2.python.onnx.backend import Caffe2Backend as c2
import caffe2.python.onnx.frontend as c2_onnx


@click.command(
    help='convert caffe2 net to onnx model',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('caffe2_net', type=click.File('rb'))
@click.option('--caffe2-net-name',
              type=str,
              help="Name of the caffe2 net")
@click.option('--caffe2-init-net',
              type=click.File('rb'),
              help="Path of the caffe2 init net pb file")
@click.option('--value-info',
              type=str,
              help='A json string providing the '
              'type and shape information of the inputs')
@click.option('-o', '--output', required=True,
              type=click.File('wb'),
              help='Output path for the onnx model pb file')
def caffe2_to_onnx(caffe2_net,
                   caffe2_net_name,
                   caffe2_init_net,
                   value_info,
                   output):
    c2_net_proto = caffe2_pb2.NetDef()
    c2_net_proto.ParseFromString(caffe2_net.read())
    if not c2_net_proto.name and not caffe2_net_name:
        raise click.BadParameter(
            'The input caffe2 net does not have name, '
            '--caffe2-net-name must be provided')
    c2_net_proto.name = caffe2_net_name or c2_net_proto.name
    if caffe2_init_net:
        c2_init_net_proto = caffe2_pb2.NetDef()
        c2_init_net_proto.ParseFromString(caffe2_init_net.read())
        c2_init_net_proto.name = '{}_init'.format(caffe2_net_name)
    else:
        c2_init_net_proto = None

    if value_info:
        value_info = json.loads(value_info)

    onnx_model = c2_onnx.caffe2_net_to_onnx_model(
        predict_net=c2_net_proto,
        init_net=c2_init_net_proto,
        value_info=value_info)

    output.write(onnx_model.SerializeToString())


@click.command(
    help='convert onnx model to caffe2 net',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('onnx_model', type=click.File('rb'))
@click.option('-o', '--output', required=True,
              type=click.File('wb'),
              help='Output path for the caffe2 net file')
@click.option('--init-net-output',
              required=True,
              type=click.File('wb'),
              help='Output path for the caffe2 init net file')
def onnx_to_caffe2(onnx_model, output, init_net_output):
    onnx_model_proto = ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())

    init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model_proto)
    init_net_output.write(init_net.SerializeToString())
    output.write(predict_net.SerializeToString())
