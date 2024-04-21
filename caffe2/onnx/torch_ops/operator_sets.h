#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
    PyTorch,
    1,
    SparseLengthsSumFused8BitRowwise);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, SparseLengthsSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, SparseLengthsWeightedSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, BatchGather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, DotProduct);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, FCTransposed);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, BatchMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, ExpandDims);

// Iterate over schema from ai.onnx.pytorch domain opset 1
class OpSet_PyTorch_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, SparseLengthsSumFused8BitRowwise)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, SparseLengthsSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, SparseLengthsWeightedSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, BatchGather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, DotProduct)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, FCTransposed)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, BatchMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, ExpandDims)>());
  }
};

inline void RegisterPyTorchOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_PyTorch_ver1>();
}

} // namespace ONNX_NAMESPACE
