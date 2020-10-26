#include <torch/csrc/jit/passes/utils/onnx_utils.h>
#include <onnx/onnx_pb.h>

namespace torch { namespace jit {

namespace onnx = ::ONNX_NAMESPACE;

onnx::TensorProto_DataType ATenTypeToOnnxType(at::ScalarType at_type) {
  switch (at_type) {
    case at::kDouble:
      return onnx::TensorProto_DataType_DOUBLE;
    case at::kFloat:
      return onnx::TensorProto_DataType_FLOAT;
    case at::kHalf:
      return onnx::TensorProto_DataType_FLOAT16;
    case at::kByte:
      return onnx::TensorProto_DataType_UINT8;
    case at::kChar:
      return onnx::TensorProto_DataType_INT8;
    case at::kShort:
      return onnx::TensorProto_DataType_INT16;
    case at::kInt:
      return onnx::TensorProto_DataType_INT32;
    case at::kLong:
      return onnx::TensorProto_DataType_INT64;
    case at::kBool:
      return onnx::TensorProto_DataType_BOOL;
    case at::kQInt8:
      return onnx::TensorProto_DataType_INT8;
    case at::kQUInt8:
      return onnx::TensorProto_DataType_UINT8;
    case at::kQInt32:
      return onnx::TensorProto_DataType_INT32;
    default:
      AT_ERROR("unexpected tensor scalar type");
  }
}

}} // namespace torch::jit