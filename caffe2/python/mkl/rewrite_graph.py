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

def last_producer(ops, blob):
    for (i, op) in reversed(list(enumerate(ops))):
        if blob in op.output:
            return i
    raise ValueError("Failed to find last producer of blob, %s", blob)


def rewrite_run_net_simple(net):
    # Simple rewrite for now - assume entire graph can be executed
    # with MKL, so just insert copy ops for external_input[0] and
    # external_output[0]
    def mkl_tmp(name):
        return "{}__MKL__".format(name)

    input_blob = net.external_input[0]
    if input_blob != net.op[0].input[0]:
        raise Exception(
            "Input blob: {} is not consumed by first op: {}".format(
                input_blob, net.op[0]))
    # Modify input/outputs to point to copied MKL blobs.
    copy_input_op = core.CreateOperator(
        "CopyCPUToMKL", input_blob, mkl_tmp(input_blob))
    net.op[0].input[0] = mkl_tmp(input_blob)

    copy_output_ops = [
        core.CreateOperator("CopyMKLToCPU", mkl_tmp(output_blob), output_blob)
        for output_blob in net.external_output]

    for output_blob in net.external_output:
        last_producer_idx = last_producer(net.op, output_blob)
        renamed_outputs = [blob if blob != output_blob else mkl_tmp(blob)
                           for blob in net.op[last_producer_idx].output]
        net.op[last_producer_idx].output[:] = renamed_outputs
        # Rename any subsequent consumers of an output blob.
        for op in net.op[last_producer_idx + 1:]:
            renamed_input = [blob if blob != output_blob else mkl_tmp(blob)
                             for blob in op.input]
            op.input[:] = renamed_input

    ops = [copy_input_op] + net.op[:] + copy_output_ops
    del net.op[:]
    net.op.extend(ops)
    for op in net.op:
        op.device_option.MergeFrom(
            core.DeviceOption(device_type=caffe2_pb2.MKLDNN))
        op.engine = ""


def rewrite_model_helper_simple(model):
    model = copy.deepcopy(model)
    # All parameter initialization should run on MKL
    rewrite_init_net_simple(model.param_init_net.Proto())
    rewrite_run_net_simple(model.net.Proto())
    return model
