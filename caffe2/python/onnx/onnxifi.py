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


def onnxifi_caffe2_net(
        pred_net,
        input_shapes,
        max_batch_size=1,
        max_seq_size=1,
        debug=False,
        use_onnx=True):
    """
    Transform the caffe2_net by collapsing ONNXIFI-runnable nodes into Onnxifi c2 ops
    """
    shape_hints = {}
    for k, v in input_shapes.items():
        shape_hints[k] = v
    pred_net_str = C.onnxifi(pred_net.SerializeToString(),
                             shape_hints,
                             max_batch_size,
                             max_seq_size,
                             debug,
                             use_onnx)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return pred_net_cut
