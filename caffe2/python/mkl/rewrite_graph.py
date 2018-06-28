from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python._import_c_extension as C


def rewrite_init_net_simple(net, ideep=True):
    device = caffe2_pb2.IDEEP if ideep else caffe2_pb2.MKLDNN
    for op in net.op:
        op.device_option.device_type = device

def last_producer(ops, blob):
    for (i, op) in reversed(list(enumerate(ops))):
        if blob in op.output:
            return i
    raise ValueError("Failed to find last producer of blob, %s", blob)


def fix_BoxWithNMSLimit(net):
    outputs = set()
    for op in net.op:
        if op.type == 'BoxWithNMSLimit':
            outputs.add(op.output[0])
            outputs.add(op.output[1])
            outputs.add(op.output[2])
    for op in net.op:
        if op.type == 'CopyIDEEPToCPU':
            if op.input[0] in outputs:
                print("Chaning CopyIDEEPToCPU to Copy for {}".format(op.input[0]))
                op.type = 'Copy'
                op.device_option.device_type = caffe2_pb2.CPU


def rewrite_run_net_simple(net, ideep=True):
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
    from_cpu = "CopyCPUToIDEEP" if ideep else "CopyCPUToMKL"
    to_cpu = "CopyIDEEPToCPU" if ideep else "CopyMKLToCPU"
    copy_input_op = core.CreateOperator(
        from_cpu, input_blob, mkl_tmp(input_blob))
    net.op[0].input[0] = mkl_tmp(input_blob)

    copy_output_ops = [
        core.CreateOperator(to_cpu, mkl_tmp(output_blob), output_blob)
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
    device = caffe2_pb2.IDEEP if ideep else caffe2_pb2.MKLDNN
    for op in net.op:
        op.device_option.MergeFrom(
            core.DeviceOption(device_type=device))
        op.engine = ""

    if ideep:
        # Temporarily disbale conv+relu fusion until we verify further
        # net.ParseFromString(
        #     C.transform_optimizeForIDEEP(net.SerializeToString()))
        fix_BoxWithNMSLimit(net)


def rewrite_model_helper_simple(model, ideep=True):
    model = copy.deepcopy(model)
    # All parameter initialization should run on MKL
    rewrite_init_net_simple(model.param_init_net.Proto(), ideep)
    rewrite_run_net_simple(model.net.Proto(), ideep)
    return model
