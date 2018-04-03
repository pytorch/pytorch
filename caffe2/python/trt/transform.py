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

def _infer_shapes(init_net, pred_net, inputs):
    ws, outputs = c2_native_run_net(init_net, pred_net, inputs)
    hints = {}
    for op in pred_net.op:
        for o in op.output:
            if o not in hints:
                blob = ws.FetchBlob(o)
                if hasattr(blob, 'shape'):
                    hints[o] = blob.shape
        for i in op.input:
            if i not in hints:
                blob = ws.FetchBlob(i)
                if hasattr(blob, 'shape'):
                    hints[i] = blob.shape

    return hints

def _ssa_rewrite_input(i):
    return i + "_0";

def transform_caffe2_net(init_net,
        pred_net,
        input_shapes,
        populate_shapes = False,
        max_batch_size=50,
        max_workspace_size=2*1024*1024,
        verbosity=1,
        debug_builder=False):
    """
    Transfrom the caffe2_net by collapsing TRT-runnable nodes into trt c2 ops
    """
    check_gpu_()
    c2_front.ssa_rewrite(pred_net, init_net, value_info=[])
    input_data = {}
    for k,v in input_shapes.iteritems():
        input_data[_ssa_rewrite_input(k)] = np.random.randn(*v).astype(np.float32)

    # Hacky way to infer shapes as not all our operators have shape inference function.
    # Normally this is not needed
    if populate_shapes:
        shape_hints = _infer_shapes(init_net, pred_net, input_data)

    shape_hints = {}
    for k,v in input_shapes.iteritems():
        shape_hints[_ssa_rewrite_input(k)] = v
    init_net_str, pred_net_str = C.transform_trt(init_net.SerializeToString(),
                                                 pred_net.SerializeToString(),
                                                 shape_hints,
                                                 max_batch_size,
                                                 max_workspace_size,
                                                 verbosity,
                                                 debug_builder)
    init_net_cut = caffe2_pb2.NetDef()
    init_net_cut.ParseFromString(init_net_str)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return init_net_cut, pred_net_cut

