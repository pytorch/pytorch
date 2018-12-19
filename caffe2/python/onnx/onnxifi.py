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
        infer_shapes=False,
        debug=False):
    """
    Transform the caffe2_net by collapsing ONNXIFI-runnable nodes into Onnxifi c2 ops
    """
    # Inject an fake input tensor to help popluate the shape if we
    # do not do shape inference
    shape_hints = {}
    external_inputs = []
    if not infer_shapes:
        for k, v in input_shapes.items():
            need_input_tensor = True
            if workspace.HasBlob(k):
                itensor = workspace.FetchBlob(k)
                if itensor.shape == v:
                    need_input_tensor = False
            if need_input_tensor:
                workspace.FeedBlob(k, np.random.randn(*v).astype(np.float32))
                external_inputs.append(k)

    for k, v in input_shapes.items():
        shape_hints[k] = v
    pred_net_str = C.onnxifi(pred_net.SerializeToString(),
                             external_inputs,
                             shape_hints,
                             infer_shapes,
                             debug)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return pred_net_cut
