#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace ONNX_NAMESPACE {
enum TensorProto_DataType : int;
}

namespace torch {
namespace jit {

// Utility functions for PyTorch to ONNX conversion.

TORCH_API ONNX_NAMESPACE::TensorProto_DataType ATenTypeToOnnxType(
    at::ScalarType at_type);

} // namespace jit
} // namespace torch
