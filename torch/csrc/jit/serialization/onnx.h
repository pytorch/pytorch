#pragma once

#include <onnx/onnx_pb.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model);

} // namespace torch::jit
