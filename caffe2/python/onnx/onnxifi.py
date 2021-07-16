## @package onnx
#Module caffe2.python.onnx.onnxifi

"""
ONNXIFI a Caffe2 net
"""

from caffe2.proto import caffe2_pb2
import caffe2.python._import_c_extension as C


def onnxifi_set_option(option_name, option_value):
    """
    Set onnxifi option
    """
    return C.onnxifi_set_option(option_name, str(option_value))


def onnxifi_get_option(option_name):
    """
    Get onnxifi option
    """
    return C.onnxifi_get_option(option_name)

def onnxifi_caffe2_net(
        pred_net,
        input_shapes,
        max_batch_size=1,
        max_seq_size=1,
        debug=False,
        use_onnx=True,
        merge_fp32_inputs_into_fp16=False,
        adjust_batch=True,
        block_list=None,
        weight_names=None,
        net_ssa_rewritten=False,
        timeout=0):
    """
    Transform the caffe2_net by collapsing ONNXIFI-runnable nodes into Onnxifi c2 ops
    """
    shape_hints = caffe2_pb2.TensorBoundShapes()
    if type(input_shapes) is caffe2_pb2.TensorBoundShapes:
        shape_hints = input_shapes
    elif type(input_shapes) is dict:
        for k, v in input_shapes.items():
            tbs = caffe2_pb2.TensorBoundShape()
            tbs.name = k
            tbs.shape.dims.extend(v)
            tbs.dim_type.extend([caffe2_pb2.TensorBoundShape.CONSTANT] * len(tbs.shape.dims))
            tbs.dim_type[0] = caffe2_pb2.TensorBoundShape.BATCH
            shape_hints.shapes.extend([tbs])
        shape_hints.max_batch_size = max_batch_size
        shape_hints.max_feature_len = max_seq_size
    pred_net_str = C.onnxifi(pred_net.SerializeToString(),
                             shape_hints.SerializeToString(),
                             block_list if block_list else [],
                             weight_names if weight_names is not None else [],
                             max_batch_size,
                             max_seq_size,
                             timeout,
                             adjust_batch,
                             debug,
                             merge_fp32_inputs_into_fp16,
                             net_ssa_rewritten,
                             use_onnx)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    return pred_net_cut
