import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes
PYTORCH_ONNX_CAFFE2_BUNDLE = _C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"


class ExportTypes:
    PROTOBUF_FILE = 1
    ZIP_ARCHIVE = 2
    COMPRESSED_ZIP_ARCHIVE = 3
    DIRECTORY = 4


def _export(*args, **kwargs):
    from torch.onnx import utils
    result = utils._export(*args, **kwargs)
    return result


def export(*args, **kwargs):
    from torch.onnx import utils
    return utils.export(*args, **kwargs)


def export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils.export_to_pretty_string(*args, **kwargs)


def _export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils._export_to_pretty_string(*args, **kwargs)


def _optimize_trace(trace, operator_export_type):
    from torch.onnx import utils
    trace.set_graph(utils._optimize_graph(trace.graph(), operator_export_type))


def set_training(*args, **kwargs):
    from torch.onnx import utils
    return utils.set_training(*args, **kwargs)


def _run_symbolic_function(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_function(*args, **kwargs)


def _run_symbolic_method(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_method(*args, **kwargs)


def is_in_onnx_export():
    from torch.onnx import utils
    return utils.is_in_onnx_export()
