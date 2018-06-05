## @package onnx
# Module caffe2.python.onnx.helper
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from onnx.backend.base import namedtupledict

from caffe2.python.onnx.workspace import Workspace
import caffe2.python._import_c_extension as C

import io
import logging
import time


log = logging.getLogger(__name__)


def c2_native_run_op(op_def, inputs):
    ws = Workspace()
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            ws.FeedBlob(key, value, op_def.device_option)
    else:
        assert(len(op_def.input) == len(inputs))
        for key, value in zip(op_def.input, inputs):
            ws.FeedBlob(key, value, op_def.device_option)

    ws.RunOperatorOnce(op_def)

    output_names = op_def.output
    output_values = [ws.FetchBlob(name) for name in output_names]
    return ws, namedtupledict('Outputs', output_names)(*output_values)


def c2_native_run_net(init_net, predict_net, inputs):
    ws = Workspace()
    if init_net:
        ws.RunNetOnce(init_net)

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            ws.FeedBlob(key, value, predict_net.device_option)
    else:
        uninitialized = [input_name
                         for input_name in predict_net.external_input
                         if not ws.HasBlob(input_name)]
        if len(uninitialized) == len(inputs):
            for key, value in zip(uninitialized, inputs):
                ws.FeedBlob(key, value, predict_net.device_option)
        else:
            # If everything is initialized,
            # we just initialized the first len(inputs) external_input.
            assert(len(inputs) <= len(predict_net.external_input))
            for i in range(len(inputs)):
                ws.FeedBlob(predict_net.external_input[i], inputs[i],
                            predict_net.device_option)

    ws.RunNetOnce(predict_net)

    output_names = predict_net.external_output
    output_values = [ws.FetchBlob(name) for name in output_names]
    return ws, namedtupledict('Outputs', output_names)(*output_values)


def load_caffe2_net(file):
    net = caffe2_pb2.NetDef()
    with open(file, "rb") as f:
        net.ParseFromString(f.read())
    return net


def save_caffe2_net(net, file, output_txt=False):
    with open(file, "wb") as f:
        f.write(net.SerializeToString())
    if output_txt:
        with open(file + "txt", "w") as f:
            f.write(str(net))


def benchmark_caffe2_model(init_net, predict_net, warmup_iters=3, main_iters=10, layer_details=True):
    '''
        Run the benchmark net on the target model.
        Return the execution time per iteration (millisecond).
    '''
    ws = Workspace()
    if init_net:
        ws.RunNetOnce(init_net)
    ws.CreateNet(predict_net)
    results = ws.BenchmarkNet(predict_net.name, warmup_iters, main_iters, layer_details)
    del ws
    return results[0]


def benchmark_pytorch_model(model, inputs, training=False, warmup_iters=3,
                            main_iters=10, verbose=False):
    '''
        Run the model several times, and measure the execution time.
        Return the execution time per iteration (millisecond).
    '''
    for _i in range(warmup_iters):
        model(*inputs)
    total_pytorch_time = 0.0
    for _i in range(main_iters):
        ts = time.time()
        model(*inputs)
        te = time.time()
        total_pytorch_time += te - ts
    log.info("The PyTorch model execution time per iter is {} milliseconds, "
             "{} iters per second.".format(total_pytorch_time / main_iters * 1000,
                                           main_iters / total_pytorch_time))
    return total_pytorch_time * 1000 / main_iters
