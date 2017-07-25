from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from caffe2.proto import caffe2_pb2
from caffe2.python import core


def rewrite_init_net_simple(net):
    for op in net.op:
        op.device_option.device_type = caffe2_pb2.MKLDNN


def rewrite_run_net_simple(net):
    # Simple rewrite for now - assume entire graph can be executed
    # with MKL, so just insert copy ops for external_input[0] and
    # external_output[0]
    def mkl_tmp(name):
        return "{}__MKL__".format(name)

    input_blob = net.external_input[0]
    (output_blob,) = net.external_output
    if input_blob != net.op[0].input[0]:
        raise Exception(
            "Input blob: {} is not consumed by first op: {}".format(
                input_blob, net.op[0]))
    if output_blob not in net.op[-1].output:
        raise Exception(
            "Output blob: {} is not produced by last op: {}".format(
                output_blob, net.op[-1].output[0]))

    # Modify input/outputs to point to copied MKL blobs.

    copy_input_op = core.CreateOperator(
        "CopyCPUToMKL", input_blob, mkl_tmp(input_blob))
    net.op[0].input[0] = mkl_tmp(input_blob)
    copy_output_op = core.CreateOperator(
        "CopyMKLToCPU", mkl_tmp(output_blob), output_blob)
    net.op[-1].output[0] = mkl_tmp(output_blob)
    ops = [copy_input_op] + net.op[:] + [copy_output_op]
    del net.op[:]
    net.op.extend(ops)
    for op in net.op:
        op.device_option.device_type = caffe2_pb2.MKLDNN


def rewrite_model_helper_simple(model):
    model = copy.deepcopy(model)
    # All parameter initialization should run on MKL
    rewrite_init_net_simple(model.param_init_net.Proto())
    rewrite_run_net_simple(model.net.Proto())
    return model
