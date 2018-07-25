## @package onnx
#Module caffe2.python.trt.transform

"""
TensorRT related transformation
Note that ONNX-TRT enforce an NCHW input!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python.onnx.helper import c2_native_run_net, c2_native_run_op
from caffe2.python import core, workspace
import caffe2.python.onnx.frontend as c2_front
import caffe2.python._import_c_extension as C
import numpy as np

def _dim_values_to_list(dim_values):
    return [x.dim_value for x in dim_values]


def _get_output_shapes(output_value_infos):
    names = [x.name for x in output_value_infos]
    shapes = [_dim_values_to_list(x.type.tensor_type.shape.dim) for x in output_value_infos]
    return dict(zip(names, shapes))


def check_gpu_():
    try:
        C.get_cuda_version()
    except Exception as _:
       raise Exception("TensorRT related functions require CUDA support")

def convert_onnx_model_to_trt_op(onnx_model,
        max_batch_size=50,
        max_workspace_size=2*1024*1024,
        verbosity=1,
        debug_builder=False):
    """
    Convert the whole ONNX model to a TensorRT C2 op
    """
    check_gpu_()
    trt_str = C.onnx_to_trt_op(onnx_model.SerializeToString(),
                               _get_output_shapes(onnx_model.graph.output),
                               max_batch_size,
                               max_workspace_size,
                               verbosity,
                               debug_builder)
    op = caffe2_pb2.OperatorDef()
    op.ParseFromString(trt_str)
    return op


# Assume the workspace is already filled with init weights
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


def transform_caffe2_net(
        pred_net,
        input_shapes,
        populate_shapes = False,
        max_batch_size=50,
        max_workspace_size=2*1024*1024,
        verbosity=1,
        debug_builder=False,
        build_serializable_op=True):
    """
    Transfrom the caffe2_net by collapsing TRT-runnable nodes into trt c2 ops
    """
    check_gpu_()

    # Hacky way to infer shapes as not all our operators have shape inference function.
    # Normally this is not needed
    shape_hints = {}
    if populate_shapes:
        input_data = {}
        for k,v in input_shapes.items():
            input_data[k] = np.random.randn(*v).astype(np.float32)
        shape_hints = _infer_shapes(pred_net, input_data)

    for k,v in input_shapes.items():
        shape_hints[k] = v
    pred_net_str = C.transform_trt(pred_net.SerializeToString(),
                                   shape_hints,
                                   max_batch_size,
                                   max_workspace_size,
                                   verbosity,
                                   debug_builder,
                                   build_serializable_op)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return pred_net_cut

