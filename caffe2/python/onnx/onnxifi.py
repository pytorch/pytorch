## @package onnx
#Module caffe2.python.onnx.onnxifi

"""
ONNXIFI a Caffe2 net
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python._import_c_extension as C
import numpy as np


def _infer_shapes(pred_net, inputs):
    workspace.RunNetOnce(pred_net)
    hints = {}
    for op in pred_net.op:
        for o in op.output:
            if o not in hints:
                blob = workspace.FetchBlob(o)
                if hasattr(blob, 'shape'):
                    hints[o] = blob.shape
        for i in op.input:
            if i not in hints:
                blob = workspace.FetchBlob(i)
                if hasattr(blob, 'shape'):
                    hints[i] = blob.shape

    return hints


def onnxifi_caffe2_net(
        pred_net,
        input_shapes,
        populate_shapes=False,
        debug=False):
    """
    Transfrom the caffe2_net by collapsing ONNXIFI-runnable nodes into Onnxifi c2 ops
    """
    # Hacky way to infer shapes as not all our operators have shape inference function.
    # Normally this is not needed
    shape_hints = {}
    if populate_shapes:
        input_data = {}
        for k, v in input_shapes.items():
            input_data[k] = np.random.randn(*v).astype(np.float32)
        shape_hints = _infer_shapes(pred_net, input_data)

    for k, v in input_shapes.items():
        shape_hints[k] = v
    pred_net_str = C.onnxifi(pred_net.SerializeToString(),
                                       shape_hints,
                                       debug)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return pred_net_cut
