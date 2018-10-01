#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(PyTorch, 1, DUMMY_TEST_ONLY);

// Iterate over schema from ai.onnx.pytorch domain opset 1
class OpSet_PyTorch_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           PyTorch, 1, DUMMY_TEST_ONLY)>());
  }
};

inline void RegisterPyTorchOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_PyTorch_ver1>();
}

} // namespace ONNX_NAMESPACE
