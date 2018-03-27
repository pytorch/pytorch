## @package data_parallel_model_utils
# Module caffe2.python.data_parallel_model_utils
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from future.utils import viewitems, viewkeys, viewvalues

import logging

from caffe2.python import core
from caffe2.python.data_parallel_model import stripBlobName

log = logging.getLogger("data_parallel_model_utils")
log.setLevel(logging.INFO)


def GetActivationBlobs(model):
    # Hacky way to get activations, think of a better way
    activations = []
    first_gpu_prefix = "{}_{}/".format(model._device_prefix, model._devices[0])

    all_inputs = set()
    for op in model.net.Proto().op:
        for inp in op.input:
            all_inputs.add(inp)

    params = set(model.GetParams(''))

    for op in model.net.Proto().op:
        for b in op.output:
            if b.startswith(first_gpu_prefix) and not b.endswith("_grad"):
                if b in all_inputs and b not in params and b + "_grad" in all_inputs:
                    activations.append(stripBlobName(b))
    return activations


def _ShiftActivationDevices(model, activations, from_device, to_device):
    prefix = "{}_{}/".format(model._device_prefix, from_device)
    activations = set([prefix + a for a in activations])
    all_activations = set([prefix + a for a in GetActivationBlobs(model)])
    ops = list(op for op in model.net.Proto().op if
               op.device_option.cuda_gpu_id == from_device)
    device_mapping = {a: to_device for a in activations}
    device_mapping.update({b: from_device for b in all_activations if
                           b not in activations})

    # Assign each blob to a device in a label propagation manner. activations
    # override, and if multiple activations in same op, the output activations
    # determine.
    for op in ops:
        op_device = None
        for b in list(op.input) + list(op.output):
            if b in device_mapping:
                if b in all_activations or op_device is None:
                    op_device = device_mapping[b]
        if op_device is None:
            op_device = op.device_option.cuda_gpu_id
        for b in list(op.input) + list(op.output):
            if b not in device_mapping and b.startswith(prefix):
                device_mapping[b] = op_device
        op.device_option.cuda_gpu_id = op_device

    # Change param_init_net accordingly
    for op in model.param_init_net.Proto().op:
        if op.output[0] in device_mapping:
            op.device_option.cuda_gpu_id = device_mapping[op.output[0]]


def ShiftActivationDevices(model, activations, shifts):
    '''
    Function to enable simple model-parallellism for data_parallel_model
    models. 'shifts' is a dictionary from_gpu -> to_gpu, and activations is
    a list of activation blobs (wout gpu_x/ prefix -- use GetActivationBlobs()).

    Operators handling these activations are shifted to the gpu declared in
    'shifts'. Also related operators such as gradient operators will be moved.
    Appropriate copy-ops are inserted.

    This allows shifting memory usage from one gpu to another, enabling bigger
    models to be trained.
    '''
    assert set(viewvalues(shifts)).intersection(set(viewkeys(shifts))) == set()
    for from_device, to_device in viewitems(shifts):
        log.info(
            "Shifting {} activations from {} --> {}".
            format(len(activations), from_device, to_device)
        )
        _ShiftActivationDevices(model, activations, from_device, to_device)

    param_init_net, blob_to_device = core.InjectCrossDeviceCopies(model.param_init_net)
    net, _blob_to_device = core.InjectCrossDeviceCopies(model.net, blob_to_device)
    model.param_init_net = param_init_net
    model.net = net
