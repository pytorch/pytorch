## @package mobile_exporter
# Module caffe2.python.mobile_exporter

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2
import numpy as np


def add_tensor(net, name, blob):
    ''' Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    '''
    kTypeNameMapper = {
        np.dtype('float32'): "GivenTensorFill",
        np.dtype('int32'): "GivenTensorIntFill",
        np.dtype('int64'): "GivenTensorInt64Fill",
        np.dtype('uint8'): "GivenTensorByteStringToUInt8Fill",
        np.dtype('O'): "GivenTensorStringFill"
    }

    shape = blob.shape
    values = blob
    # pass array of uint8 as a string to save storage
    # storing uint8_t has a large overhead for now
    if blob.dtype == np.dtype('uint8'):
        shape = blob.shape
        values = [blob.tobytes()]
    # Only allow string arrays as objects.
    # The only intended use case for this is to store arrays of strings in the
    # model which can be used for post processing results in subsequent ops.
    if blob.dtype == np.dtype('O'):
        for blob_val in blob:
            assert(isinstance(blob_val, bytes))

    op = core.CreateOperator(
        kTypeNameMapper[blob.dtype],
        [], [name],
        arg=[
            utils.MakeArgument("shape", shape),
            utils.MakeArgument("values", values),
        ]
    )
    net.op.extend([op])


def Export(workspace, net, params):
    """Returns init_net and predict_net suitable for writing to disk
       and loading into a Predictor"""
    proto = net if isinstance(net, caffe2_pb2.NetDef) else net.Proto()
    predict_net = caffe2_pb2.NetDef()
    predict_net.CopyFrom(proto)
    init_net = caffe2_pb2.NetDef()
    # Populate the init_net.
    ssa, blob_versions = core.get_ssa(net)
    inputs = []
    for versioned_inputs, _ in ssa:
        inputs += [name for name, _ in versioned_inputs]

    input_blobs = [blob_name for blob_name, version in
                   blob_versions.items()
                   if version == 0 and blob_name not in params]
    # Blobs that are never used as an input to another layer,
    # i.e. strictly output blobs.
    output_blobs = [blob_name for blob_name, version in
                    blob_versions.items()
                    if version != 0 and blob_name not in inputs]

    for blob_ref in params:
        blob_name = str(blob_ref)
        blob = workspace.FetchBlob(blob_name)
        add_tensor(init_net, blob_name, blob)
    # We have to make sure the blob exists in the namespace
    # and we can do so with fake data. (Which is immediately overwritten
    # by any typical usage)
    for blob_name in input_blobs:
        init_net.op.extend(
            [
                core.CreateOperator(
                    "GivenTensorFill", [], [blob_name],
                    arg=[
                        utils.MakeArgument("shape", [1, 1]),
                        utils.MakeArgument("values", [0.0])
                    ]
                )
            ]
        )

    # Now we make input/output_blobs line up with what Predictor expects.
    del predict_net.external_input[:]

    new_external_inputs = input_blobs
    for external_input in proto.external_input:
        if external_input not in new_external_inputs:
            new_external_inputs.append(external_input)

    # For populating weights
    predict_net.external_input.extend(new_external_inputs)
    # Ensure the output is also consistent with what we want
    del predict_net.external_output[:]
    predict_net.external_output.extend(output_blobs)
    return init_net, predict_net
